# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    n = 256
    
    # Precompute all arithmetic progressions
    ap_indices = []
    pos_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for step in range(1, n):
        for start in range(1, n + 1):
            max_len = (n - start) // step + 1
            if max_len < 1:
                continue
            for length in range(1, max_len + 1):
                indices = [(start - 1) + k * step for k in range(length)]
                ap_indices.append(indices)
                for idx in indices:
                    pos_to_aps[idx].append(ap_id)
                ap_id += 1
    
    num_aps = ap_id
    
    np.random.seed(42)
    best_signs = None
    best_max = float('inf')
    
    for restart in range(20):
        if time.time() > DEADLINE:
            break
        
        signs = np.random.choice([-1, 1], size=n).astype(np.int64)
        
        ap_sums = np.zeros(num_aps, dtype=np.int64)
        for ap_i, indices in enumerate(ap_indices):
            s = 0
            for idx in indices:
                s += signs[idx]
            ap_sums[ap_i] = s
        
        current_max = np.max(np.abs(ap_sums))
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
        
        improved = True
        while improved and time.time() < DEADLINE - 3:
            improved = False
            pos_order = np.random.permutation(n)
            for i in pos_order:
                if time.time() > DEADLINE - 3:
                    break
                old_sign = signs[i]
                affected = pos_to_aps[i]
                
                old_vals = [(ap_i, ap_sums[ap_i]) for ap_i in affected]
                
                for ap_i, _ in old_vals:
                    ap_sums[ap_i] -= 2 * old_sign
                
                # Only check max over affected APs and assume others are <= current_max
                # But we need the true max. Let's just compute it.
                new_max = np.max(np.abs(ap_sums))
                
                if new_max < current_max:
                    signs[i] = -old_sign
                    current_max = new_max
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    for ap_i, old_val in old_vals:
                        ap_sums[ap_i] = old_val
        
        if time.time() > DEADLINE:
            break
    
    if best_signs is None:
        best_signs = np.random.choice([-1, 1], size=n).astype(np.int64)
    
    return list(best_signs)

# EVOLVE-BLOCK-END
