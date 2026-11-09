import time
import sys
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 50.0
    
    all_masks = list(range(1 << n))
    n_cands = len(all_masks)
    
    def evaluate(family_masks):
        c = 0
        for i in range(len(family_masks)):
            mi = family_masks[i]
            for j in range(i+1, len(family_masks)):
                mj = family_masks[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    c += 1
        return c

    best_result = []
    best_score = -1
    
    # Run greedy multiple times with different random seeds/tie-breaks
    # Or just one deterministic run might be enough.
    # Let's do 10 runs with random tie-breaking.
    
    for run in range(10):
        if time.time() > deadline - 5:
            break
            
        selected = []
        selected_set = set()
        scores = [0] * n_cands
        
        for _ in range(m):
            if time.time() > deadline - 1:
                break
                
            best_idx = -1
            best_score = -1
            best_tie = -1
            
            # Collect candidates with max score
            max_s = -1
            for k in range(n_cands):
                if k not in selected_set:
                    if scores[k] > max_s:
                        max_s = scores[k]
            
            if max_s == -1:
                break
                
            # Pick random among those with max score
            candidates = []
            for k in range(n_cands):
                if k not in selected_set and scores[k] == max_s:
                    candidates.append(k)
            
            best_idx = random.choice(candidates)
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            # Update scores
            new_mask = all_masks[best_idx]
            for j in range(n_cands):
                if j not in selected_set:
                    mj = all_masks[j]
                    if (mj & new_mask) == mj or (mj & new_mask) == new_mask:
                        scores[j] += 1
        
        if len(selected) == m:
            current_score = evaluate(selected)
            if current_score > best_score:
                best_score = current_score
                best_result = selected[:]
                
    # Convert best_result to lists
    result = []
    for idx in best_result:
        mask = all_masks[idx]
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
