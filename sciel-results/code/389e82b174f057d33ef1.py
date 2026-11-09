import time
import sys

def construct():
    # Use a 7-element base set to provide enough candidates for a dense subposet
    n_base = 7
    all_masks = list(range(1 << n_base))
    n_cands = len(all_masks)
    
    # Precompute comparability matrix
    # comp[i][j] is True if mask_i is a strict subset of mask_j or vice versa
    comp = [[False] * n_cands for _ in range(n_cands)]
    for i in range(n_cands):
        mi = all_masks[i]
        for j in range(i + 1, n_cands):
            mj = all_masks[j]
            # Check strict subset
            if (mi & mj) == mi and mi != mj:
                comp[i][j] = True
                comp[j][i] = True
            elif (mi & mj) == mj and mi != mj:
                comp[i][j] = True
                comp[j][i] = True

    selected = []
    selected_set = set()
    scores = [0] * n_cands
    
    # Greedy selection of 40 sets
    for _ in range(40):
        best_idx = -1
        best_score = -1
        for k in range(n_cands):
            if k not in selected_set:
                if scores[k] > best_score:
                    best_score = scores[k]
                    best_idx = k
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update scores for all unselected candidates
        for j in range(n_cands):
            if j not in selected_set:
                if comp[j][best_idx]:
                    scores[j] += 1

    # Convert selected indices to lists of integers
    result = []
    for idx in selected:
        mask = all_masks[idx]
        subset = [b for b in range(n_base) if (mask >> b) & 1]
        result.append(subset)
        
    return result
