import numpy as np
import time
import random
from math import comb

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    def build_indices(n):
        """Build triple index mapping and violation index arrays."""
        triples = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triples.append((a, b, c))
        num_triples = len(triples)
        
        triple_index = {}
        for i, t in enumerate(triples):
            triple_index[t] = i
        
        # Precompute 4-sets and their triple indices (red violations: all 4 triples red)
        red_sets = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        idxs = [
                            triple_index[(a,b,c)],
                            triple_index[(a,b,d)],
                            triple_index[(a,c,d)],
                            triple_index[(b,c,d)]
                        ]
                        red_sets.append(idxs)
        red_indices = np.array(red_sets, dtype=np.int32) if red_sets else np.empty((0,4), dtype=np.int32)
        
        # Precompute 5-sets and their triple indices (blue violations: all 10 triples blue)
        blue_sets = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            idxs = [
                                triple_index[(a,b,c)],
                                triple_index[(a,b,d)],
                                triple_index[(a,b,e)],
                                triple_index[(a,c,d)],
                                triple_index[(a,c,e)],
                                triple_index[(a,d,e)],
                                triple_index[(b,c,d)],
                                triple_index[(b,c,e)],
                                triple_index[(b,d,e)],
                                triple_index[(c,d,e)]
                            ]
                            blue_sets.append(idxs)
        blue_indices = np.array(blue_sets, dtype=np.int32) if blue_sets else np.empty((0,10), dtype=np.int32)
        
        return num_triples, red_indices, blue_indices
    
    def algebraic_init(n, seed_val):
        """Initialize coloring using algebraic structure."""
        random.seed(seed_val)
        num_triples = comb(n, 3)
        colors = np.zeros(num_triples, dtype=np.int8)
        
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # Use modular arithmetic for base pattern
                    val = (a * 7 + b * 11 + c * 13 + seed_val) % 7
                    if val < 3:
                        colors[idx] = 0  # red
                    else:
                        colors[idx] = 1  # blue
                    # Random perturbation
                    if random.random() < 0.2:
                        colors[idx] = 1 - colors[idx]
                    idx += 1
        return colors
    
    def compute_violations(colors, red_indices, blue_indices):
        """Compute number of red and blue violations."""
        if len(red_indices) == 0:
            red_viol = 0
        else:
            red_sums = np.sum(colors[red_indices], axis=1)
            red_viol = np.sum(red_sums == 0)
        
        if len(blue_indices) == 0:
            blue_viol = 0
        else:
            blue_sums = np.sum(colors[blue_indices], axis=1)
            blue_viol = np.sum(blue_sums == 10)
        
        return int(red_viol), int(blue_viol)
    
    def simulated_annealing(n, num_triples, red_indices, blue_indices, deadline, seed_val):
        """Run simulated annealing until deadline or valid solution found."""
        random.seed(seed_val)
        
        # Initialize
        colors = algebraic_init(n, seed_val)
        
        T_start = 1.0
        T_end = 0.005
        alpha = 0.9995  # Slower cooling for more exploration
        
        T = T_start
        best_colors = colors.copy()
        best_rv, best_bv = compute_violations(colors, red_indices, blue_indices)
        best_energy = best_rv + best_bv
        
        if best_energy == 0:
            return colors.tolist()
        
        step = 0
        max_steps = 100000
        stall_count = 0
        stall_limit = 5000
        
        while T > T_end and step < max_steps:
            if time.time() > deadline:
                break
            
            # Pick a random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            
            old_rv, old_bv = compute_violations(colors, red_indices, blue_indices)
            old_energy = old_rv + old_bv
            
            # Flip
            colors[t_idx] = 1 - colors[t_idx]
            
            new_rv, new_bv = compute_violations(colors, red_indices, blue_indices)
            new_energy = new_rv + new_bv
            delta_e = new_energy - old_energy
            
            # Accept or reject
            if delta_e < 0:
                # Accept improvement
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_rv = new_rv
                    best_bv = new_bv
                    best_colors = colors.copy()
                    stall_count = 0
                    if best_energy == 0:
                        return colors.tolist()
                else:
                    stall_count += 1
            else:
                # Accept with probability exp(-delta_e / T)
                if random.random() < np.exp(-delta_e / max(T, 1e-10)):
                    stall_count += 1
                else:
                    # Reject: undo flip
                    colors[t_idx] = 1 - colors[t_idx]
            
            # Cool down
            T *= alpha
            step += 1
            
            # If stuck too long, restart with current best
            if stall_count > stall_limit:
                colors = best_colors.copy()
                stall_count = 0
                T = max(T, 0.1)  # Reheat slightly
        
        if best_energy == 0:
            return best_colors.tolist()
        return None
    
    # Try n values from high to low
    for n in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 18, 16, 14, 12, 10]:
        if time.time() - start_time > time_limit - 3:
            break
        
        try:
            num_triples, red_indices, blue_indices = build_indices(n)
        except Exception:
            continue
        
        if n < 4:
            return {'n': n, 'colors': [0] * num_triples}
        
        # Try multiple seeds with simulated annealing
        found = None
        num_seeds = 10
        for seed_val in range(num_seeds):
            if time.time() - start_time > time_limit - 3:
                break
            
            remaining_time = time_limit - 3 - (time.time() - start_time)
            if remaining_time < 2:
                break
            
            # Allocate time per attempt
            attempt_deadline = min(time.time() + remaining_time, start_time + time_limit - 2)
            
            result = simulated_annealing(n, num_triples, red_indices, blue_indices, 
                                          attempt_deadline, seed_val)
            if result is not None:
                found = result
                break
        
        if found is not None:
            return {'n': n, 'colors': found}
    
    # Final fallback: small n with deterministic construction
    n = 9
    num_triples = comb(n, 3)
    triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triples.append((a, b, c))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
