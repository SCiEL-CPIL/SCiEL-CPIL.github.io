import random
import math
import time
from itertools import combinations
import numpy as np

def construct():
    n = 34
    triples = list(combinations(range(n), 3))
    T = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    N_K4 = len(k4s)
    N_K5 = len(k5s)
    
    # Precompute triple indices for each K4 and K5
    k4_triples = []
    for k4 in k4s:
        t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
        k4_triples.append(t_indices)
        
    k5_triples = []
    for k5 in k5s:
        t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
        k5_triples.append(t_indices)
        
    # Reverse maps: triple index -> list of K4 indices, list of K5 indices
    triple_to_k4s = [[] for _ in range(T)]
    triple_to_k5s = [[] for _ in range(T)]
    
    for i, t_inds in enumerate(k4_triples):
        for t in t_inds:
            triple_to_k4s[t].append(i)
            
    for i, t_inds in enumerate(k5_triples):
        for t in t_inds:
            triple_to_k5s[t].append(i)

    # Optimal p for red (0)
    # Minimize E[V] = C(34,4)p^4 + C(34,5)(1-p)^10
    p_red = 0.65
    colors = np.array([0 if random.random() < p_red else 1 for _ in range(T)], dtype=np.int8)
    
    # Initialize counts
    red_counts_k4 = np.zeros(N_K4, dtype=np.int32)
    blue_counts_k5 = np.zeros(N_K5, dtype=np.int32)
    
    # Calculate initial counts
    for i, t_inds in enumerate(k4_triples):
        red_counts_k4[i] = np.sum(colors[t_inds] == 0)
        
    for i, t_inds in enumerate(k5_triples):
        blue_counts_k5[i] = np.sum(colors[t_inds] == 1)
        
    # Initialize violation lists
    violated_k4_list = [i for i in range(N_K4) if red_counts_k4[i] == 4]
    violated_k5_list = [i for i in range(N_K5) if blue_counts_k5[i] == 10]
    
    total_violations = len(violated_k4_list) + len(violated_k5_list)
    
    start_time = time.time()
    timeout = 115
    
    T_curr = 1.0
    cooling = 0.999
    
    # Helper to update violation lists and counts
    def flip_triple(t_idx):
        nonlocal total_violations
        nonlocal violated_k4_list
        nonlocal violated_k5_list
        nonlocal red_counts_k4
        nonlocal blue_counts_k5
        
        old_color = colors[t_idx]
        new_color = 1 - old_color
        colors[t_idx] = new_color
        
        delta_violations = 0
        
        # Update K4s
        for k4_idx in triple_to_k4s[t_idx]:
            old_count = red_counts_k4[k4_idx]
            if old_color == 0: # Red -> Blue, red count decreases
                new_count = old_count - 1
            else: # Blue -> Red, red count increases
                new_count = old_count + 1
            
            # Update violation status
            if old_count == 4 and new_count != 4:
                # Was violated, no longer is
                delta_violations -= 1
                # Remove from violated_k4_list
                # Find index in list
                # To keep O(1), we could use a dict or swap-remove
                # For simplicity and given small lists potentially, or we can just leave it and clean up?
                # No, we need accurate list for picking.
                # Let's use a dictionary mapping k4_idx to position in list? Or just scan?
                # Scanning is O(N). N_K4 ~ 46k. Too slow.
                # We need efficient removal.
                # We can maintain a dict `k4_pos` {k4_idx: pos_in_list}.
                # But modifying lists and dicts is complex.
                # Alternative: Don't maintain a list of violated indices.
                # Instead, pick a random index and check if it's violated.
                # If violations are rare, this is bad.
                # But initially, violations are common.
                # Let's try the "Pick random, check" approach first. It's simpler.
                # If it's too slow, we'll optimize.
                # Actually, let's just use the list but accept O(N) removal? No.
                # Let's use a different strategy for picking:
                # Pick a random triple.
                # This avoids the list maintenance issue.
                # The "biased" picking was an optimization.
                # Let's stick to random triple picking for now, as the count update is fast.
                pass # Logic for list update skipped in this thought, will use random triple in code
                
            elif old_count != 4 and new_count == 4:
                delta_violations += 1
                
            red_counts_k4[k4_idx] = new_count

        # Update K5s
        for k5_idx in triple_to_k5s[t_idx]:
            old_count = blue_counts_k5[k5_idx]
            if old_color == 1: # Blue -> Red, blue count decreases
                new_count = old_count - 1
            else: # Red -> Blue, blue count increases
                new_count = old_count + 1
            
            if old_count == 10 and new_count != 10:
                delta_violations -= 1
            elif old_count != 10 and new_count == 10:
                delta_violations += 1
                
            blue_counts_k5[k5_idx] = new_count
            
        return delta_violations

    # We will use Random Triple Picking.
    # If we want to be smarter, we can pick a triple that is in many violations.
    # But random is robust.
    
    while total_violations > 0 and time.time() - start_time < timeout:
        t_idx = random.randint(0, T - 1)
        
        # We need to calculate delta BEFORE flipping to decide acceptance?
        # Or flip, check delta, and revert if rejected?
        # Reverting is easy: just flip back and revert counts.
        # Let's do: Calculate delta, decide, then apply if accepted.
        
        old_color = colors[t_idx]
        
        delta = 0
        # Simulate changes to calculate delta
        for k4_idx in triple_to_k4s[t_idx]:
            old_count = red_counts_k4[k4_idx]
            if old_color == 0:
                new_count = old_count - 1
            else:
                new_count = old_count + 1
            if old_count == 4 and new_count != 4:
                delta -= 1
            elif old_count != 4 and new_count == 4:
                delta += 1
                
        for k5_idx in triple_to_k5s[t_idx]:
            old_count = blue_counts_k5[k5_idx]
            if old_color == 1:
                new_count = old_count - 1
            else:
                new_count = old_count + 1
            if old_count == 10 and new_count != 10:
                delta -= 1
            elif old_count != 10 and new_count == 10:
                delta += 1
                
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.5
        else:
            prob = math.exp(-delta / T_curr)
            accept = random.random() < prob
            
        if accept:
            # Apply flip
            colors[t_idx] = 1 - old_color
            total_violations += delta
            
            # Update counts
            for k4_idx in triple_to_k4s[t_idx]:
                if old_color == 0:
                    red_counts_k4[k4_idx] -= 1
                else:
                    red_counts_k4[k4_idx] += 1
                    
            for k5_idx in triple_to_k5s[t_idx]:
                if old_color == 1:
                    blue_counts_k5[k5_idx] -= 1
                else:
                    blue_counts_k5[k5_idx] += 1
                    
        T_curr *= cooling
        
    if total_violations == 0:
        return {'n': n, 'colors': colors.tolist()}
    else:
        return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
