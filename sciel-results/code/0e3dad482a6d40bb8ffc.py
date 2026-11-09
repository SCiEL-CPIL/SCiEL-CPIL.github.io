"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _precompute_mappings(n):
    """
    Precompute:
    1. List of all triples (a,b,c).
    2. For each triple index, the indices of the 4-sets (K4s) that contain it.
    3. For each triple index, the indices of the 5-sets (K5s) that contain it.
    4. Mapping from 4-set index to its 4 vertices.
    5. Mapping from 5-set index to its 5 vertices.
    """
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    
    n_triples = len(triples)
    
    # Map from sorted tuple of vertices to index in the list of k-sets
    # We need to identify which 4-sets and 5-sets contain a given triple.
    
    # Generate all 4-sets and 5-sets
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    # Create a mapping from tuple to index for fast lookup
    k4_index = {k: i for i, k in enumerate(k4s)}
    k5_index = {k: i for i, k in enumerate(k5s)}
    
    # For each triple, find the 4-sets and 5-sets that contain it
    # triple_k4s[t_idx] = list of indices of k4s containing triples[t_idx]
    # triple_k5s[t_idx] = list of indices of k5s containing triples[t_idx]
    
    triple_k4s = [[] for _ in range(n_triples)]
    triple_k5s = [[] for _ in range(n_triples)]
    
    # Iterate through all 4-sets and add them to the triples they contain
    for k4_idx, k4 in enumerate(k4s):
        # A 4-set {a,b,c,d} contains 4 triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        # We need to find the indices of these triples in the `triples` list.
        # Since n is small (<=40), we can just check or use a map.
        # Let's create a map from triple tuple to index for O(1) lookup.
        pass # We'll do this more efficiently below.

    # More efficient: Iterate triples, find containing k-sets.
    # For a triple (a,b,c), the containing 4-sets are (a,b,c,d) for d not in {a,b,c}.
    # The containing 5-sets are (a,b,c,d,e) for d<e not in {a,b,c}.
    
    # To make this fast, we need to map (a,b,c) to its index in `triples`.
    triple_index_map = {t: i for i, t in enumerate(triples)}
    
    for t_idx, (a, b, c) in enumerate(triples):
        # Find 4-sets containing (a,b,c)
        for d in range(n):
            if d == a or d == b or d == c:
                continue
            # The 4-set is the sorted tuple of {a,b,c,d}
            k4_tuple = tuple(sorted((a, b, c, d)))
            k4_idx = k4_index.get(k4_tuple)
            if k4_idx is not None:
                triple_k4s[t_idx].append(k4_idx)
                
        # Find 5-sets containing (a,b,c)
        # We need to choose 2 more vertices from the remaining n-3 vertices
        remaining = [v for v in range(n) if v not in (a, b, c)]
        for d_idx in range(len(remaining)):
            for e_idx in range(d_idx + 1, len(remaining)):
                d = remaining[d_idx]
                e = remaining[e_idx]
                k5_tuple = tuple(sorted((a, b, c, d, e)))
                k5_idx = k5_index.get(k5_tuple)
                if k5_idx is not None:
                    triple_k5s[t_idx].append(k5_idx)
                    
    return triples, triple_k4s, triple_k5s, k4s, k5s

def construct():
    """
    Construct a valid two-coloring of triples on n vertices with no red K4^(3)
    and no blue K5^(3), maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try to find a valid coloring for n starting from 34 downwards
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
            
        # Precompute mappings for this n
        triples, triple_k4s, triple_k5s, k4s, k5s = _precompute_mappings(n)
        n_triples = len(triples)
        n_k4 = len(k4s)
        n_k5 = len(k5s)
        
        # We will use a local search / simulated annealing approach.
        # State:
        # colors: array of 0/1 for each triple
        # red_counts: array of size n_k4, count of red triples in each k4
        # blue_counts: array of size n_k5, count of blue triples in each k5
        # violations: total number of violations (k4s with 4 reds + k5s with 10 blues)
        
        found = False
        for attempt in range(10): # Try multiple random restarts
            if time.time() > deadline:
                break
                
            # Initialize colors randomly
            colors = np.random.randint(0, 2, size=n_triples, dtype=np.int8)
            
            # Initialize counts
            red_counts = np.zeros(n_k4, dtype=np.int32)
            blue_counts = np.zeros(n_k5, dtype=np.int32)
            
            # Populate initial counts
            for t_idx in range(n_triples):
                if colors[t_idx] == 0: # Red
                    for k4_idx in triple_k4s[t_idx]:
                        red_counts[k4_idx] += 1
                else: # Blue
                    for k5_idx in triple_k5s[t_idx]:
                        blue_counts[k5_idx] += 1
            
            # Calculate initial violations
            violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
            
            if violations == 0:
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
                
            # Simulated Annealing / Local Search
            T = 2.0
            cooling = 0.9995
            best_viol = violations
            best_colors_copy = colors.copy()
            
            # Precompute the number of k4s and k5s affected by each flip
            # We already have triple_k4s and triple_k5s
            
            # Optimization: We can precompute the "impact" of a flip?
            # No, the impact depends on the current state of the counts.
            
            # To speed up, we can try to only consider flips that reduce violations.
            # But we need to calculate the delta.
            
            # Let's use a simple hill climbing with random restarts if stuck.
            # Or Simulated Annealing.
            
            # For each flip, we need to update red_counts and blue_counts.
            # This involves iterating over the list of k4s/k5s for that triple.
            # Length of triple_k4s[t] is n-3.
            # Length of triple_k5s[t] is C(n-3, 2).
            # For n=34, n-3=31, C(31,2)=465.
            # So each flip takes ~500 operations.
            # We can do ~10,000 flips per second easily in Python if optimized.
            # 150 seconds -> 1.5 million flips. Should be enough.
            
            # Let's implement the loop.
            no_improve_count = 0
            max_no_improve = 5000
            
            while time.time() < deadline and no_improve_count < max_no_improve:
                if violations == 0:
                    best_n = n
                    best_colors = colors.tolist()
                    found = True
                    break
                    
                # Pick a random triple to flip
                t_idx = random.randint(0, n_triples - 1)
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                # Calculate delta in violations
                # This is the tricky part. We need to know how many k4s/k5s
                # will change their status (cross the threshold).
                
                # If flipping Red -> Blue:
                # Red counts in associated k4s decrease by 1.
                # Blue counts in associated k5s increase by 1.
                
                # If flipping Blue -> Red:
                # Red counts in associated k4s increase by 1.
                # Blue counts in associated k5s decrease by 1.
                
                delta_viol = 0
                
                if old_color == 0: # Red -> Blue
                    # Red counts decrease
                    # A k4 is a violation if count == 4.
                    # If count was 4, it becomes 3 (violation removed, delta -1).
                    # If count was 5, it becomes 4 (violation added, delta +1).
                    # But counts can't be > 4? No, a k4 has 4 triples. Max count is 4.
                    # So only count == 4 changes to 3.
                    # Wait, if count was 4, it was a violation. Now it's 3. Delta -1.
                    # Can count be > 4? No.
                    # So for Red -> Blue:
                    # For each k4 in triple_k4s[t_idx]:
                    #   if red_counts[k4] == 4: delta_viol -= 1
                    #   # if red_counts[k4] == 5: impossible
                    
                    # Blue counts increase
                    # A k5 is a violation if count == 10.
                    # If count was 9, it becomes 10 (violation added, delta +1).
                    # If count was 10, it becomes 11 (violation removed? No, 11 is still >= 10? 
                    # Wait, the condition is "all 10 triples blue". So count MUST be 10.
                    # If count is 11, it's impossible because there are only 10 triples.
                    # So count can only be 10 if all are blue.
                    # If we flip one to red, count becomes 9. Violation removed. Delta -1.
                    # If we flip one to blue, and count was 9, it becomes 10. Violation added. Delta +1.
                    
                    for k4_idx in triple_k4s[t_idx]:
                        if red_counts[k4_idx] == 4:
                            delta_viol -= 1
                            
                    for k5_idx in triple_k5s[t_idx]:
                        if blue_counts[k5_idx] == 9:
                            delta_viol += 1
                        elif blue_counts[k5_idx] == 10:
                            delta_viol -= 1
                            
                else: # Blue -> Red
                    # Red counts increase
                    # If count was 3, it becomes 4 (violation added, delta +1).
                    # If count was 4, it becomes 5 (impossible).
                    
                    # Blue counts decrease
                    # If count was 10, it becomes 9 (violation removed, delta -1).
                    # If count was 9, it becomes 8 (no change).
                    
                    for k4_idx in triple_k4s[t_idx]:
                        if red_counts[k4_idx] == 3:
                            delta_viol += 1
                            
                    for k5_idx in triple_k5s[t_idx]:
                        if blue_counts[k5_idx] == 10:
                            delta_viol -= 1
                            
                # Acceptance criterion
                if delta_viol < 0 or (delta_viol == 0 and random.random() < 0.5) or (delta_viol > 0 and random.random() < np.exp(-delta_viol / T)):
                    # Apply flip
                    colors[t_idx] = new_color
                    violations += delta_viol
                    
                    if violations < best_viol:
                        best_viol = violations
                        best_colors_copy = colors.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    # Update counts
                    if old_color == 0: # Was Red, now Blue
                        for k4_idx in triple_k4s[t_idx]:
                            red_counts[k4_idx] -= 1
                        for k5_idx in triple_k5s[t_idx]:
                            blue_counts[k5_idx] += 1
                    else: # Was Blue, now Red
                        for k4_idx in triple_k4s[t_idx]:
                            red_counts[k4_idx] += 1
                        for k5_idx in triple_k5s[t_idx]:
                            blue_counts[k5_idx] -= 1
                            
                    T *= cooling
                else:
                    # Reject flip, no state change
                    no_improve_count += 1
                    T *= cooling
                    
            if best_viol == 0:
                best_n = n
                best_colors = best_colors_copy.tolist()
                found = True
                break
                
        if found:
            break
            
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
        
    # Fallback
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 
              for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
