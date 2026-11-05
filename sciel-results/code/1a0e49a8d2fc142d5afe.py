import numpy as np
import time
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    def get_k4_indices(n):
        """Return array of shape (C(n,4), 4) with indices of triples in each K4."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))
        return np.array(k4s, dtype=np.int32)

    def get_k5_indices(n):
        """Return array of shape (C(n,5), 10) with indices of triples in each K5."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
        return np.array(k5s, dtype=np.int32)

    def build_reverse_indices(k4_indices, k5_indices, num_triples):
        """Build lists of K4/K5 indices for each triple."""
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i in range(len(k4_indices)):
            for t in k4_indices[i]:
                triple_to_k4s[t].append(i)
                
        for i in range(len(k5_indices)):
            for t in k5_indices[i]:
                triple_to_k5s[t].append(i)
                
        # Convert to numpy arrays for faster access? 
        # Actually, lists are fine if we slice k4_indices with them.
        # k4_indices[np.array(triple_to_k4s[t])]
        return triple_to_k4s, triple_to_k5s

    def count_violations(colors, k4_indices, k5_indices):
        """Calculate total cost."""
        # Red K4: all 4 are 0
        red_k4 = np.sum(np.all(colors[k4_indices] == 0, axis=1))
        # Blue K5: all 10 are 1
        blue_k5 = np.sum(np.all(colors[k5_indices] == 1, axis=1))
        return int(red_k4 + blue_k5)

    def local_search(n, time_limit):
        num_triples = len(list(combinations(range(n), 3)))
        k4_indices = get_k4_indices(n)
        k5_indices = get_k5_indices(n)
        triple_to_k4s, triple_to_k5s = build_reverse_indices(k4_indices, k5_indices, num_triples)
        
        # Pre-convert reverse indices to numpy arrays for faster indexing
        # triple_to_k4s_np[i] is an array of K4 indices containing triple i
        triple_to_k4s_np = [np.array(lst, dtype=np.int32) for lst in triple_to_k4s]
        triple_to_k5s_np = [np.array(lst, dtype=np.int32) for lst in triple_to_k5s]
        
        best_colors = None
        best_cost = float('inf')
        
        max_restarts = 200
        # Estimate time per restart dynamically? 
        # Let's just run until time_limit is reached.
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
                
            # Initialize colors
            # Random initialization
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Calculate initial cost
            current_cost = count_violations(colors, k4_indices, k5_indices)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors, True
            
            # Simulated Annealing
            # Start temperature high, cool down
            temp = 10.0
            cool_rate = 0.999
            min_temp = 0.1
            
            # Run SA for a fixed number of steps or time slice
            # Let's run for ~0.5s per restart or until cost 0
            step_deadline = time.time() + 0.5
            
            while time.time() < step_deadline:
                # Pick random triple
                t = np.random.randint(num_triples)
                
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta_cost = 0
                
                # Check K4s
                k4_idxs = triple_to_k4s_np[t]
                if len(k4_idxs) > 0:
                    # Get colors of all triples in these K4s
                    # Shape: (num_k4s, 4)
                    colors_k4 = colors[k4_indices[k4_idxs]]
                    # Count zeros in each K4
                    sum_zeros = np.sum(colors_k4 == 0, axis=1)
                    
                    if old_color == 0:
                        # Flipping 0 -> 1
                        # If it was a violation (4 zeros), it becomes 3 zeros. Delta = -1
                        # Count how many had 4 zeros
                        delta_cost -= np.sum(sum_zeros == 4)
                    else:
                        # Flipping 1 -> 0
                        # If the other 3 are 0 (so sum_zeros was 3, since t was 1->0 so currently t is 1? No.)
                        # Current state: t is 1. sum_zeros counts zeros in current state.
                        # If other 3 are 0, sum_zeros is 3.
                        # After flip, t becomes 0, sum_zeros becomes 4. Delta = +1
                        delta_cost += np.sum(sum_zeros == 3)
                        
                # Check K5s
                k5_idxs = triple_to_k5s_np[t]
                if len(k5_idxs) > 0:
                    colors_k5 = colors[k5_indices[k5_idxs]]
                    # Count ones in each K5
                    sum_ones = np.sum(colors_k5 == 1, axis=1)
                    
                    if old_color == 1:
                        # Flipping 1 -> 0
                        # If it was a violation (10 ones), it becomes 9 ones. Delta = -1
                        delta_cost -= np.sum(sum_ones == 10)
                    else:
                        # Flipping 0 -> 1
                        # If other 9 are 1 (sum_ones is 9), after flip it becomes 10. Delta = +1
                        delta_cost += np.sum(sum_ones == 9)
                        
                new_cost = current_cost + delta_cost
                
                # Acceptance criterion
                if new_cost < current_cost:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors, True
                elif new_cost == current_cost and np.random.rand() < 0.5:
                    colors[t] = new_color
                elif new_cost > current_cost and np.random.rand() < np.exp(-new_cost / temp):
                    colors[t] = new_color
                    current_cost = new_cost
                
                temp *= cool_rate
                if temp < min_temp:
                    temp = min_temp

        return best_colors, (best_cost == 0)

    best_n = 9
    # Fallback coloring for n=9 (just to be safe, though we should find better)
    # A valid coloring for n=9 is easy to find or we can just use random and check.
    # Actually, let's just try to find the best n we can.
    
    targets = [34, 33, 32, 31, 30, 28, 25, 20]
    
    # Time allocation
    # 34: 100s
    # 33: 30s
    # 32: 10s
    # Rest: share remaining
    
    time_alloc = {
        34: 100.0,
        33: 30.0,
        32: 10.0,
        31: 5.0,
        30: 5.0,
        28: 5.0,
        25: 5.0,
        20: 5.0
    }
    
    for n in targets:
        if time.time() > deadline - 5:
            break
        
        t_limit = time.time() + time_alloc.get(n, 5.0)
        # Ensure we don't exceed global deadline
        if t_limit > deadline:
            t_limit = deadline
            
        try:
            colors, is_valid = local_search(n, t_limit)
            if is_valid:
                best_n = n
                # Convert numpy array to list of ints
                best_colors = colors.tolist()
                # If we found a valid one, we might want to try higher n?
                # But the loop is descending. So if we find 34, we break?
                # The task says maximize n.
                # If we find 34, it's likely the max.
                # If we find 33, we might have missed 34.
                # But we tried 34 first.
                # So if we find valid for n, we can return it if n is high enough?
                # Or just keep the best found so far.
                # Since we iterate descending, the first valid one is the best.
                # So we can break?
                # Yes, if we find a valid coloring for n, and we are going down,
                # any subsequent n will be smaller.
                # So we can return immediately?
                # Wait, what if 34 failed but 35 would have worked?
                # We don't try 35.
                # The known bound is 34.
                # So if we find 34, we return.
                # If we find 33, we return 33 (since 34 failed).
                # So yes, break on first success.
                return {'n': best_n, 'colors': best_colors}
        except Exception as e:
            pass

    # If no valid coloring found for targets, try a small n to return something valid
    # n=9 is always valid?
    # Let's check n=9.
    # C(9,3) = 84.
    # We can just run a quick search for n=9.
    try:
        colors, is_valid = local_search(9, time.time() + 5.0)
        if is_valid:
            best_n = 9
            best_colors = colors.tolist()
        else:
            # Hardcode a valid coloring for n=9?
            # Or just return random? No, must be valid.
            # For n=9, it's very likely to find one quickly.
            # If it fails, we have a problem.
            # Let's try a few more restarts for n=9.
            for _ in range(10):
                colors, is_valid = local_search(9, time.time() + 0.5)
                if is_valid:
                    best_n = 9
                    best_colors = colors.tolist()
                    break
    except:
        pass

    # Fallback: n=1 is always valid (no triples)
    if best_n < 1:
        best_n = 1
        best_colors = []
        
    return {'n': best_n, 'colors': best_colors}
