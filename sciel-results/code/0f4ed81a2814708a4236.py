import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Fallback: A known valid small coloring (n=9 is trivially safe, but let's use a simple pattern)
    # The seed program used n=9. We will try to beat it significantly.
    best_n = 9
    triples_9 = list(combinations(range(9), 3))
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]

    def solve_n(n, time_budget):
        """
        Attempts to find a valid coloring for n vertices within time_budget.
        Returns (colors_array, is_valid).
        """
        if time.time() > deadline - 2:
            return None, False
            
        # 1. Precompute structure
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Build K4 and K5 index lists
        k4_indices = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4_indices.append([
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ])
            
        k5_indices = []
        for quint in combinations(range(n), 5):
            idxs = [triple_to_idx[t] for t in combinations(quint, 3)]
            k5_indices.append(idxs)

        num_k4 = len(k4_indices)
        num_k5 = len(k5_indices)

        # Build reverse indices: triple_idx -> list of clique indices
        # This is crucial for fast delta updates
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(k4_indices):
            for t in idxs:
                triple_to_k4s[t].append(i)
                
        for i, idxs in enumerate(k5_indices):
            for t in idxs:
                triple_to_k5s[t].append(i)

        # Convert to numpy arrays for faster access if needed, but lists are fine for reverse indices
        # For the cliques themselves, we keep them as lists of indices into the colors array.

        # 2. Initialization Strategies
        # We want to try a few different "structures" to see if one is close to valid.
        
        def init_random():
            return np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
        def init_mod3():
            return np.array([0 if (a+b+c)%3==0 else 1 for a,b,c in triples], dtype=np.int8)
            
        def init_xor():
            return np.array([1 if (a^b^c)%2==0 else 0 for a,b,c in triples], dtype=np.int8)
            
        def init_mod5():
            return np.array([0 if (a+2*b+3*c)%5 < 2 else 1 for a,b,c in triples], dtype=np.int8)

        strategies = [init_mod3, init_xor, init_mod5, init_random]
        
        best_colors_local = None
        best_cost_local = float('inf')
        
        for strat in strategies:
            if time.time() > deadline - 2:
                break
                
            colors = strat()
            
            # Calculate initial cost
            # Red K4: all 4 are 0. Count zeros in each K4.
            # Blue K5: all 10 are 1. Count ones in each K5.
            
            # Efficient cost calculation using numpy
            # For K4: violation if sum(colors[idxs]) == 0
            # For K5: violation if sum(colors[idxs]) == 10
            
            # We can vectorize the sum calculation for all cliques at once?
            # k4_indices is a list of lists. 
            # Let's convert to a 2D array for batch summing if size is manageable.
            # n=34 -> num_k4 = 46376, num_k5 = 278256. 
            # k4_matrix shape (46376, 4). k5_matrix shape (278256, 10).
            # This is feasible in memory and fast with numpy.
            
            try:
                k4_mat = np.array(k4_indices, dtype=np.int32)
                k5_mat = np.array(k5_indices, dtype=np.int32)
            except Exception:
                # Fallback if memory fails (unlikely for n<=34)
                return None, False

            # Calculate counts
            # zeros_k4[i] = number of 0s in k4 i
            # ones_k5[i] = number of 1s in k5 i
            
            # colors is int8. 
            # zeros = 1 - colors
            # ones = colors
            
            zeros_k4 = np.sum(1 - colors[k4_mat], axis=1)
            ones_k5 = np.sum(colors[k5_mat], axis=1)
            
            cost = np.sum(zeros_k4 == 4) + np.sum(ones_k5 == 10)
            
            if cost < best_cost_local:
                best_cost_local = cost
                best_colors_local = colors.copy()
                
                if cost == 0:
                    return colors, True
            
            # If initial cost is 0, we are done.
            # Otherwise, run local search from this start.
            
            # Local Search Setup
            # We maintain:
            # current_cost
            # zeros_k4 array (counts of 0s in each K4)
            # ones_k5 array (counts of 1s in each K5)
            
            current_cost = cost
            cur_colors = colors.copy()
            
            # Precompute reverse indices are already built.
            # We need to update zeros_k4 and ones_k5 when a flip happens.
            
            # Time slice for this restart
            # Allocate remaining time proportionally or fixed slices?
            # Let's give each restart a slice of the total budget.
            # If we have 4 strategies, maybe 25% each? 
            # But random is usually the best for large n. 
            # Let's do 2 quick heuristic restarts and 1 long random restart.
            
            if strat == init_random:
                restart_deadline = deadline - 1.0
            else:
                restart_deadline = min(time.time() + (deadline - start_time) * 0.1, deadline - 1.0)
            
            # Simulated Annealing / Hill Climbing
            # Temperature strategy: Start high, cool down.
            # Or simple hill climbing with random restarts on stagnation.
            # Given the "count" structure, delta is very cheap.
            
            temp = 2.0
            cool_rate = 0.999
            no_improve = 0
            max_no_improve = 2000
            
            while time.time() < restart_deadline:
                # Pick random triple
                t = np.random.randint(num_triples)
                old_c = cur_colors[t]
                new_c = 1 - old_c
                
                delta = 0
                
                # Update K4s containing t
                # For each k4_idx in triple_to_k4s[t]:
                #   if old_c == 0: zeros_k4[k4_idx] decreases by 1
                #   else: zeros_k4[k4_idx] increases by 1
                #   Check if violation status changes.
                
                for k4_idx in triple_to_k4s[t]:
                    old_zeros = zeros_k4[k4_idx]
                    if old_c == 0:
                        new_zeros = old_zeros - 1
                    else:
                        new_zeros = old_zeros + 1
                    
                    # Violation if zeros == 4
                    was_violation = (old_zeros == 4)
                    is_violation = (new_zeros == 4)
                    
                    if was_violation and not is_violation:
                        delta -= 1
                    elif not was_violation and is_violation:
                        delta += 1
                    
                    zeros_k4[k4_idx] = new_zeros
                
                # Update K5s containing t
                for k5_idx in triple_to_k5s[t]:
                    old_ones = ones_k5[k5_idx]
                    if old_c == 1:
                        new_ones = old_ones - 1
                    else:
                        new_ones = old_ones + 1
                    
                    # Violation if ones == 10
                    was_violation = (old_ones == 10)
                    is_violation = (new_ones == 10)
                    
                    if was_violation and not is_violation:
                        delta -= 1
                    elif not was_violation and is_violation:
                        delta += 1
                    
                    ones_k5[k5_idx] = new_ones
                
                new_cost = current_cost + delta
                
                # Acceptance
                accepted = False
                if new_cost < current_cost:
                    accepted = True
                    no_improve = 0
                elif new_cost == current_cost:
                    if np.random.rand() < 0.5:
                        accepted = True
                    no_improve += 1
                else:
                    # Accept worse moves with probability exp(-delta/temp)
                    # delta is positive here
                    if np.random.rand() < np.exp(-delta / temp):
                        accepted = True
                    no_improve += 1
                
                if accepted:
                    cur_colors[t] = new_c
                    current_cost = new_cost
                    
                    if current_cost < best_cost_local:
                        best_cost_local = current_cost
                        best_colors_local = cur_colors.copy()
                        if current_cost == 0:
                            return cur_colors, True
                
                temp *= cool_rate
                if temp < 0.01:
                    temp = 0.01
                
                # Stagnation handling: If stuck for a while, reheat
                if no_improve > max_no_improve:
                    temp = 2.0
                    no_improve = 0

        return best_colors_local, (best_cost_local == 0)

    # Main Loop: Try to find the largest n
    # Prioritize 34, then 33, 32, etc.
    # Allocate time: 34 gets most time.
    
    targets = [34, 33, 32, 31, 30, 25, 20]
    
    # Time allocation:
    # 34: 60s
    # 33: 30s
    # 32: 20s
    # 31: 15s
    # 30: 10s
    # 25: 5s
    # 20: 5s
    
    time_alloc = {
        34: 60.0,
        33: 30.0,
        32: 20.0,
        31: 15.0,
        30: 10.0,
        25: 5.0,
        20: 5.0
    }
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        budget = time_alloc.get(n, 5.0)
        # Ensure we don't exceed deadline
        actual_budget = min(budget, deadline - time.time())
        
        if actual_budget < 1.0:
            continue
            
        try:
            colors, is_valid = solve_n(n, actual_budget)
            if is_valid and colors is not None:
                best_n = n
                best_colors = colors.tolist()
                # If we found 34, it's the known max. We can stop.
                # If we found 33, maybe try 34 with remaining time? 
                # But 34 is the first target, so if we passed it, it failed.
                # If we are at 33 and found it, 34 already failed.
                # So we can break if we find a valid one? 
                # No, we iterate from high to low. If 34 failed, we try 33.
                # If 33 succeeds, we return 33.
                # We don't need to try 32 if 33 succeeded, because we want MAX n.
                # So yes, break on first success.
                break
        except Exception as e:
            # In case of memory error or other issues, skip
            pass

    return {'n': best_n, 'colors': best_colors}
