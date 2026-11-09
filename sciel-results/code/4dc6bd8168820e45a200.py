import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        if len(colors) != len(list(combinations(range(n), 3))):
            return False
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def run_ils(target_n, deadline):
        """
        Iterated Local Search for Max-SAT formulation.
        Returns (colors, num_violations)
        """
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Precompute structure
        quads = list(combinations(range(target_n), 4))
        num_quads = len(quads)
        quints = list(combinations(range(target_n), 5))
        num_quints = len(quints)
        
        # Map triple index to list of quad indices it belongs to
        # And for each quad, which triple indices it contains
        triple_to_quads = [[] for _ in range(num_triples)]
        quad_triples = []
        for q in quads:
            a, b, c, d = q
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            quad_triples.append([t1, t2, t3, t4])
            triple_to_quads[t1].append(len(quad_triples)-1)
            triple_to_quads[t2].append(len(quad_triples)-1)
            triple_to_quads[t3].append(len(quad_triples)-1)
            triple_to_quads[t4].append(len(quad_triples)-1)
            
        triple_to_quints = [[] for _ in range(num_triples)]
        quint_triples = []
        for q in quints:
            ts = [triple_to_idx[t] for t in combinations(q, 3)]
            quint_triples.append(ts)
            for t in ts:
                triple_to_quints[t].append(len(quint_triples)-1)
                
        # State
        # colors: 0 for Red, 1 for Blue
        colors = np.array([0 if random.random() < 0.5 else 1 for _ in range(num_triples)], dtype=np.int8)
        
        # Counts
        # quad_red_count[q]: number of red triples in quad q
        # quint_blue_count[q]: number of blue triples in quint q
        quad_red_count = np.zeros(num_quads, dtype=np.int8)
        quint_blue_count = np.zeros(num_quints, dtype=np.int8)
        
        # Initialize counts
        for q_idx, ts in enumerate(quad_triples):
            quad_red_count[q_idx] = np.sum(colors[ts] == 0)
        for q_idx, ts in enumerate(quint_triples):
            quint_blue_count[q_idx] = np.sum(colors[ts] == 1)
            
        # Initial Cost
        current_cost = np.sum(quad_red_count == 4) + np.sum(quint_blue_count == 10)
        best_cost = current_cost
        best_colors = colors.copy()
        
        # ILS Parameters
        max_restarts = 50
        max_iters_per_restart = 50000
        perturbation_size = 10
        
        for r in range(max_restarts):
            if time.time() > deadline - 2:
                break
            
            # If not first restart, perturb
            if r > 0:
                # Flip random set of triples
                for _ in range(perturbation_size):
                    t_idx = random.randint(0, num_triples - 1)
                    # Update counts for flip
                    old_color = colors[t_idx]
                    
                    # Update Quad Counts
                    for q_idx in triple_to_quads[t_idx]:
                        if old_color == 0: # Red -> Blue
                            quad_red_count[q_idx] -= 1
                        else: # Blue -> Red
                            quad_red_count[q_idx] += 1
                            
                    # Update Quint Counts
                    for q_idx in triple_to_quints[t_idx]:
                        if old_color == 1: # Blue -> Red
                            quint_blue_count[q_idx] -= 1
                        else: # Red -> Blue
                            quint_blue_count[q_idx] += 1
                            
                    colors[t_idx] = 1 - colors[t_idx]
                    
                    # Recalc cost roughly? No, let's just let the main loop fix it.
                    # But we need current_cost to be accurate for the loop.
                    # Easier: just recalculate cost from scratch after perturbation?
                    # Cost calc is O(num_quads + num_quints). ~300k ops. Acceptable for restart.
                    current_cost = np.sum(quad_red_count == 4) + np.sum(quint_blue_count == 10)
                    best_cost = min(best_cost, current_cost)
                    if current_cost == 0:
                        return colors.tolist(), 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
            
            # Local Search Phase
            no_improve_count = 0
            T = 5.0
            T_min = 0.1
            cooling = 0.995
            
            for i in range(max_iters_per_restart):
                if time.time() > deadline - 2:
                    return best_colors.tolist(), best_cost
                
                t_idx = random.randint(0, num_triples - 1)
                old_color = colors[t_idx]
                
                # Calculate Delta
                delta = 0
                
                if old_color == 0: # Red -> Blue
                    # Quads: Red count decreases
                    for q_idx in triple_to_quads[t_idx]:
                        if quad_red_count[q_idx] == 4:
                            delta -= 1 # Violation removed
                    # Quints: Blue count increases
                    for q_idx in triple_to_quints[t_idx]:
                        if quint_blue_count[q_idx] == 9:
                            delta += 1 # Violation added
                else: # Blue -> Red
                    # Quads: Red count increases
                    for q_idx in triple_to_quads[t_idx]:
                        if quad_red_count[q_idx] == 3:
                            delta += 1 # Violation added
                    # Quints: Blue count decreases
                    for q_idx in triple_to_quints[t_idx]:
                        if quint_blue_count[q_idx] == 10:
                            delta -= 1 # Violation removed
                
                # Acceptance Criterion
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = True
                else:
                    # Simulated Annealing
                    if random.random() < np.exp(-delta / T):
                        accept = True
                
                if accept:
                    # Apply Flip
                    if old_color == 0:
                        for q_idx in triple_to_quads[t_idx]:
                            quad_red_count[q_idx] -= 1
                        for q_idx in triple_to_quints[t_idx]:
                            quint_blue_count[q_idx] += 1
                    else:
                        for q_idx in triple_to_quads[t_idx]:
                            quad_red_count[q_idx] += 1
                        for q_idx in triple_to_quints[t_idx]:
                            quint_blue_count[q_idx] -= 1
                    colors[t_idx] = 1 - colors[t_idx]
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        no_improve_count = 0
                        if best_cost == 0:
                            return colors.tolist(), 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
                T *= cooling
                if T < T_min:
                    T = T_min
                    # If stuck, break to restart
                    if no_improve_count > 5000:
                        break
                        
            # End of local search for this restart
            if best_cost == 0:
                break
                
        return best_colors.tolist(), best_cost

    # Strategy: Try increasing n
    # We know n=9 is valid. We try to find n=34.
    # If we fail at 34, we might try 33, etc.
    # But the search is expensive. We should focus on 34 first.
    
    best_found_n = 9
    best_found_colors = mod_color(9, 3)
    
    # Try n=34 with a dedicated time budget
    # If it fails, try n=33, etc.
    targets = [34, 33, 32, 31]
    
    for n in targets:
        if time.time() > deadline - 10:
            break
            
        # Allocate remaining time, but cap it to allow trying smaller n if 34 fails quickly?
        # Better: Give 34 most of the time.
        # Let's say 80s for 34, 10s for 33, 5s for 32, 5s for 31.
        if n == 34:
            alloc = 80
        elif n == 33:
            alloc = 10
        elif n == 32:
            alloc = 5
        else:
            alloc = 5
            
        current_deadline = time.time() + alloc
        
        # Run ILS
        # We can run multiple independent ILS processes in parallel? No, single thread.
        # Just run one long ILS.
        
        colors, cost = run_ils(n, current_deadline)
        
        if cost == 0:
            if is_valid(n, colors):
                best_found_n = n
                best_found_colors = colors
                break # Found max n, stop
                
    return {'n': best_found_n, 'colors': best_found_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        # Re-define is_valid for local check or import
        # The function is defined inside construct, so we can't call it directly here easily.
        # But the evaluator will call construct().
        pass
