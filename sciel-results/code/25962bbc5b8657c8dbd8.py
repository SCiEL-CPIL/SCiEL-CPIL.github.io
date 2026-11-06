import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_precomputed(n):
        triples = list(combinations(range(n), 3))
        t_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        t_to_k4 = [[] for _ in triples]
        t_to_k5 = [[] for _ in triples]
        
        for i, k4 in enumerate(k4s):
            ts = [t_idx[t] for t in combinations(k4, 3)]
            for t in ts:
                t_to_k4[t].append(i)
                
        for i, k5 in enumerate(k5s):
            ts = [t_idx[t] for t in combinations(k5, 3)]
            for t in ts:
                t_to_k5[t].append(i)
                
        # Convert to numpy arrays for speed
        t_to_k4_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in t_to_k4]
        t_to_k5_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in t_to_k5]
        
        return triples, t_idx, t_to_k4_np, t_to_k5_np, len(k4s), len(k5s), k4s, k5s

    def modular_init(n, seed=0):
        """Generate a coloring based on modular arithmetic."""
        triples = list(combinations(range(n), 3))
        rng = random.Random(seed)
        
        # Try different modular patterns
        patterns = []
        for mod in [3, 4, 5, 6, 7, 8]:
            for offset in range(mod):
                patterns.append((mod, offset))
        rng.shuffle(patterns)
        
        best_colors = None
        best_viol = float('inf')
        
        for mod, offset in patterns[:20]:  # Try first 20 patterns
            colors = np.zeros(len(triples), dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                val = (a * mod + b * (mod + 1) + c * (mod + 2) + offset) % (2 * mod)
                colors[i] = 0 if val < mod else 1
            
            # Quick violation count
            # Count red K4s and blue K5s
            # This is expensive, so we'll do a rough estimate
            # Actually, let's just use the first pattern and improve
            if best_colors is None:
                best_colors = colors
                best_viol = 1e9
        
        # Use a simpler approach: random with bias
        colors = np.random.RandomState(seed).randint(0, 2, size=len(triples))
        return colors

    def count_violations(colors, t_to_k4_np, t_to_k5_np, num_k4, num_k5):
        """Count total violations (red K4s + blue K5s)."""
        red_count_k4 = np.zeros(num_k4, dtype=np.int32)
        blue_count_k5 = np.zeros(num_k5, dtype=np.int32)
        
        for t in range(len(colors)):
            if colors[t] == 0:
                idxs = t_to_k4_np[t]
                if len(idxs) > 0:
                    red_count_k4[idxs] += 1
            else:
                idxs = t_to_k5_np[t]
                if len(idxs) > 0:
                    blue_count_k5[idxs] += 1
        
        viol_k4 = np.sum(red_count_k4 == 4)
        viol_k5 = np.sum(blue_count_k5 == 10)
        return viol_k4 + viol_k5, red_count_k4, blue_count_k5

    def greedy_violation_elimination(n, time_budget):
        """Start from modular heuristic, then greedily eliminate violations."""
        if time.time() > deadline - 5:
            return None
            
        triples, t_idx, t_to_k4_np, t_to_k5_np, num_k4, num_k5, k4s, k5s = get_precomputed(n)
        num_t = len(triples)
        
        # Multiple restarts with different seeds
        best_result = None
        best_viol = float('inf')
        
        num_restarts = max(3, int(time_budget / 5))
        time_per_restart = time_budget / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 3:
                break
                
            t_end = time.time() + time_per_restart
            
            # Initialize with modular pattern
            seed = r * 1000 + n
            colors = modular_init(n, seed)
            
            # Count violations
            total_viol, red_count_k4, blue_count_k5 = count_violations(
                colors, t_to_k4_np, t_to_k5_np, num_k4, num_k5
            )
            
            if total_viol == 0:
                return colors.tolist()
            
            best_viol_in_restart = total_viol
            best_colors = colors.copy()
            
            # Greedy elimination loop
            steps = 0
            max_steps = 500000
            stuck_count = 0
            
            while steps < max_steps and time.time() < t_end and time.time() < deadline - 3:
                # Find a violating clique
                viol_k4_indices = np.where(red_count_k4 == 4)[0]
                viol_k5_indices = np.where(blue_count_k5 == 10)[0]
                
                if len(viol_k4_indices) == 0 and len(viol_k5_indices) == 0:
                    # No violations!
                    return colors.tolist()
                
                # Pick a random violation to fix
                if len(viol_k4_indices) > 0:
                    # Fix a red K4: flip one of its 4 triples to blue
                    k4_idx = viol_k4_indices[random.randint(0, len(viol_k4_indices) - 1)]
                    k4 = k4s[k4_idx]
                    candidate_triples = [t_idx[t] for t in combinations(k4, 3)]
                    
                    # For each triple in this K4, compute delta if flipped R->B
                    best_delta = float('inf')
                    best_t = None
                    
                    for t in candidate_triples:
                        if colors[t] != 0:
                            continue  # Already blue, can't flip
                        # Flipping R->B: 
                        # - Red K4 containing t: decrease red_count by 1
                        #   If red_count was 4, violation removed (-1)
                        #   If red_count was 3, no change
                        # - Blue K5 containing t: increase blue_count by 1
                        #   If blue_count was 9, new violation (+1)
                        #   If blue_count was 10, was already violating, stays violating (0)
                        
                        delta = 0
                        idxs_k4 = t_to_k4_np[t]
                        if len(idxs_k4) > 0:
                            counts = red_count_k4[idxs_k4]
                            delta -= np.sum(counts == 4)
                        
                        idxs_k5 = t_to_k5_np[t]
                        if len(idxs_k5) > 0:
                            counts = blue_count_k5[idxs_k5]
                            delta += np.sum(counts == 9)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_t = t
                    
                    if best_t is None:
                        stuck_count += 1
                        if stuck_count > 100:
                            break
                        continue
                    
                    t = best_t
                    old_c = colors[t]
                    colors[t] = 1 - old_c
                    total_viol += best_delta
                    
                    # Update counts
                    idxs = t_to_k4_np[t]
                    if len(idxs) > 0:
                        if old_c == 0:
                            red_count_k4[idxs] -= 1
                        else:
                            red_count_k4[idxs] += 1
                            
                    idxs = t_to_k5_np[t]
                    if len(idxs) > 0:
                        if old_c == 0:
                            blue_count_k5[idxs] += 1
                        else:
                            blue_count_k5[idxs] -= 1
                    
                    stuck_count = 0
                    
                else:
                    # Fix a blue K5: flip one of its 10 triples to red
                    k5_idx = viol_k5_indices[random.randint(0, len(viol_k5_indices) - 1)]
                    k5 = k5s[k5_idx]
                    candidate_triples = [t_idx[t] for t in combinations(k5, 3)]
                    
                    # For each triple in this K5, compute delta if flipped B->R
                    best_delta = float('inf')
                    best_t = None
                    
                    for t in candidate_triples:
                        if colors[t] != 1:
                            continue  # Already red, can't flip
                        # Flipping B->R:
                        # - Red K4 containing t: increase red_count by 1
                        #   If red_count was 3, new violation (+1)
                        #   If red_count was 4, was already violating, stays violating (0)
                        # - Blue K5 containing t: decrease blue_count by 1
                        #   If blue_count was 10, violation removed (-1)
                        #   If blue_count was 9, no change
                        
                        delta = 0
                        idxs_k4 = t_to_k4_np[t]
                        if len(idxs_k4) > 0:
                            counts = red_count_k4[idxs_k4]
                            delta += np.sum(counts == 3)
                        
                        idxs_k5 = t_to_k5_np[t]
                        if len(idxs_k5) > 0:
                            counts = blue_count_k5[idxs_k5]
                            delta -= np.sum(counts == 10)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_t = t
                    
                    if best_t is None:
                        stuck_count += 1
                        if stuck_count > 100:
                            break
                        continue
                    
                    t = best_t
                    old_c = colors[t]
                    colors[t] = 1 - old_c
                    total_viol += best_delta
                    
                    # Update counts
                    idxs = t_to_k4_np[t]
                    if len(idxs) > 0:
                        if old_c == 0:
                            red_count_k4[idxs] -= 1
                        else:
                            red_count_k4[idxs] += 1
                            
                    idxs = t_to_k5_np[t]
                    if len(idxs) > 0:
                        if old_c == 0:
                            blue_count_k5[idxs] += 1
                        else:
                            blue_count_k5[idxs] -= 1
                    
                    stuck_count = 0
                
                if total_viol < best_viol_in_restart:
                    best_viol_in_restart = total_viol
                    best_colors = colors.copy()
                
                steps += 1
            
            if best_viol_in_restart < best_viol:
                best_viol = best_viol_in_restart
                best_result = best_colors.tolist()
            
            if best_viol == 0:
                break
        
        return best_result

    # Try n from 40 down to 34
    time_alloc = {
        40: 35.0,
        39: 20.0,
        38: 15.0,
        37: 10.0,
        36: 10.0,
        35: 10.0,
        34: 15.0
    }
    
    for n in range(40, 33, -1):
        if time.time() > deadline - 5:
            break
        budget = time_alloc.get(n, 10.0)
        if time.time() + budget > deadline - 5:
            budget = deadline - 5 - time.time()
            if budget < 3:
                continue
                
        result = greedy_violation_elimination(n, budget)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    n = 34
    result = greedy_violation_elimination(n, deadline - 2 - time.time())
    if result is not None:
        return {'n': n, 'colors': result}
        
    # Last resort: small valid coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
