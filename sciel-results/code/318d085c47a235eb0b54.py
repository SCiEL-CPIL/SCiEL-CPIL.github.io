# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    selected = []
    existing_sums = set()
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
        k = len(selected)
        
        if k == 0:
            selected.append(x)
            existing_sums.add(3 * x)
            continue
        
        S = np.array(selected)
        
        # Compute all new triple sums involving x
        parts = []
        
        # 1) x + x + x
        parts.append(np.array([3 * x]))
        
        # 2) x + x + s for each s in S
        parts.append(2 * x + S)
        
        # 3) x + s + s for each s in S
        parts.append(x + 2 * S)
        
        # 4) x + s_i + s_j for all i <= j
        pair_sums = S[:, None] + S[None, :]
        upper_mask = np.triu(np.ones((k, k), dtype=bool))
        parts.append(x + pair_sums[upper_mask])
        
        new_sums = np.concatenate(parts)
        
        # Check for duplicates among new sums
        if len(new_sums) != len(np.unique(new_sums)):
            continue
        
        # Check collision with existing sums
        new_sums_set = set(new_sums.tolist())
        if new_sums_set & existing_sums:
            continue
        
        # Valid: add x
        selected.append(x)
        existing_sums.update(new_sums.tolist())
        
        if len(selected) >= 80:
            break
    
    return selected
# EVOLVE-BLOCK-END
