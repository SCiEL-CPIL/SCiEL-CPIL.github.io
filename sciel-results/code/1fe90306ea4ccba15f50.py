# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0
    
    # 1. Precompute APs and Matrix M
    # M has shape (num_aps, n). M[k, i] = 1 if i is in AP k.
    ap_indices = []
    for start in range(n):
        for step in range(1, n):
            seq = []
            cur = start
            while cur < n:
                seq.append(cur)
                cur += step
            if len(seq) > 1:
                ap_indices.append(np.array(seq, dtype=np.int32))
    
    num_aps = len(ap_indices)
    
    # Build M as sparse or dense? Dense is 32640 * 256 ~ 8M floats. 64MB. OK.
    # Use float32 to save memory and speed up BLAS? float64 is safer for precision in sums?
    # Sums are integers up to 256. float32 has 24 bits mantissa, can hold integers up to 16M exactly.
    # So float32 is fine and faster.
    M = np.zeros((num_aps, n), dtype=np.float32)
    for k, ap_arr in enumerate(ap_indices):
        M[k, ap_arr] = 1.0

    # 2. Continuous Optimization
    # Objective: minimize ||M x||^2
    # Gradient: 2 * M.T @ (M x)
    
    best_signs = None
    best_score = float('inf')
    
    num_restarts = 5
    steps_per_restart = 500
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        x = np.random.uniform(-1, 1, n).astype(np.float32)
        
        # Adam parameters
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros(n, dtype=np.float32)
        v = np.zeros(n, dtype=np.float32)
        t = 0
        
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit - 5.0:
                break
            t += 1
            
            # Compute sums S = M @ x
            S = M @ x
            
            # Compute gradient G = 2 * M.T @ S
            G = 2.0 * (M.T @ S)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * G
            v = beta2 * v + (1 - beta2) * (G * G)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Keep x bounded to encourage sign separation
            # x = np.clip(x, -1, 1) # Clipping might interfere with gradient flow?
            # Better to just let it go, but normalize if it gets huge?
            # Or use tanh?
            # Let's stick to simple clipping for now, or just rely on the quadratic loss pulling it in.
            # Actually, for sign problems, scaling x to have norm ~ sqrt(n) is good.
            # But let's just run Adam.
            
        # Discretize
        signs = np.sign(x).astype(np.float32)
        signs[signs == 0] = 1.0
        
        # 3. Discrete Local Search
        # Compute initial sums
        sums = M @ signs
        
        max_abs_sum = np.max(np.abs(sums))
        
        # Local search loop
        # We want to flip bits to reduce max_abs_sum.
        # Strategy: Find worst AP, try flipping each of its elements.
        # To speed up, we can precompute columns of M? No, M is already in memory.
        # Accessing M[:, pos] is fast.
        
        improved = True
        while improved and (time.time() - start_time < time_limit - 2.0):
            improved = False
            
            # Find worst AP
            worst_idx = np.argmax(np.abs(sums))
            worst_val = np.abs(sums[worst_idx])
            
            # Get indices of this AP
            # We need to map worst_idx to ap_indices[worst_idx]
            # But M doesn't store indices. We stored ap_indices list.
            # ap_indices[worst_idx] is the array of positions.
            ap_pos = ap_indices[worst_idx]
            
            best_flip_pos = -1
            best_new_max = worst_val
            
            # Try flipping each pos in the worst AP
            for pos in ap_pos:
                # Flip sign at pos
                # delta = -2 * signs[pos]
                # new_sums = sums + delta * M[:, pos]
                
                # M[:, pos] is a column.
                # We can compute new_sums efficiently.
                # But we only care if max(|new_sums|) < worst_val.
                
                # Optimization: If the AP has length L, and current sum is S.
                # Flipping an element in the AP changes the sum of this AP by +/- 2.
                # It changes other APs by +/- 2 as well.
                
                # Let's just compute the full vector update for each candidate.
                # This is O(num_aps) per candidate.
                # Number of candidates = len(ap_pos).
                
                # To avoid allocating a new array for each candidate, we can reuse?
                # Or just do it. 32640 floats * 8 bytes = 256KB. 
                # Allocating this 256 times (max AP length) is fine.
                
                # However, doing this in a Python loop over pos is slow.
                # Let's vectorize the candidate evaluation?
                # We have `len(ap_pos)` candidates.
                # We want to find min over j in ap_pos of max(|sums - 2*signs[j]*M[:, j]|).
                
                # This is hard to vectorize without a huge temp array.
                # Let's stick to the loop but make it tight.
                
                s_j = signs[pos]
                # If s_j is 1, new sum = sums - 2 * M[:, pos]
                # If s_j is -1, new sum = sums + 2 * M[:, pos]
                
                if s_j == 1.0:
                    new_sums = sums - 2.0 * M[:, pos]
                else:
                    new_sums = sums + 2.0 * M[:, pos]
                
                new_max = np.max(np.abs(new_sums))
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_flip_pos = pos
            
            if best_flip_pos != -1:
                # Apply flip
                signs[best_flip_pos] *= -1
                # Update sums
                s_j = signs[best_flip_pos] # This is the NEW sign
                # Wait, the delta calculation above used the OLD sign.
                # Let's re-calculate delta based on OLD sign.
                # We stored best_flip_pos.
                # We need to update `sums` to match the flip.
                # Old sign was `signs[best_flip_pos]` BEFORE flip.
                # Let's just recompute sums from signs? No, too slow.
                # Let's store the old sign before flipping.
                
                # Actually, in the loop above, we computed `new_sums` for the flip.
                # We can just assign `sums = new_sums` if we saved it?
                # We didn't save the full array for the best candidate.
                # Let's re-do the update for the best candidate.
                
                # Get old sign
                old_sign = signs[best_flip_pos] * -1 # Current is new, so old is -new?
                # No, we haven't flipped `signs` yet.
                # So `signs[best_flip_pos]` is still the OLD sign.
                old_s = signs[best_flip_pos]
                
                # Apply flip to signs
                signs[best_flip_pos] *= -1
                
                # Update sums
                # new_sums = sums - 2 * old_s * M[:, pos]
                if old_s == 1.0:
                    sums -= 2.0 * M[:, best_flip_pos]
                else:
                    sums += 2.0 * M[:, best_flip_pos]
                    
                max_abs_sum = best_new_max
                improved = True
            else:
                # No improvement in worst AP.
                # Try a random flip? Or stop?
                # If we stop, we might be stuck in a local minimum.
                # Let's try a few random flips to escape.
                for _ in range(10):
                    pos = np.random.randint(n)
                    old_s = signs[pos]
                    signs[pos] *= -1
                    if old_s == 1.0:
                        new_sums = sums - 2.0 * M[:, pos]
                    else:
                        new_sums = sums + 2.0 * M[:, pos]
                    
                    new_max = np.max(np.abs(new_sums))
                    
                    if new_max < max_abs_sum:
                        # Accept
                        sums = new_sums
                        max_abs_sum = new_max
                        improved = True
                        break
                    else:
                        # Revert
                        signs[pos] *= -1
                        # sums unchanged
                # If no random flip worked, we are stuck.
                # Break out of local search for this restart.
                if not improved:
                    break
        
        score = max_abs_sum
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
            
    if best_signs is None:
        best_signs = np.random.choice([-1, 1], n).astype(np.float32)
        
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
