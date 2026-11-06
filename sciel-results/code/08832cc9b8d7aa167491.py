# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0
    
    # 1. Precompute APs
    ap_indices = []
    pos_to_aps = [[] for _ in range(n)]
    
    for start in range(n):
        for step in range(1, n):
            seq = []
            cur = start
            while cur < n:
                seq.append(cur)
                cur += step
            if len(seq) > 1:
                ap_arr = np.array(seq, dtype=np.int32)
                ap_idx = len(ap_indices)
                ap_indices.append(ap_arr)
                for idx in ap_arr:
                    pos_to_aps[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # 2. Helper functions
    
    def compute_sums(signs):
        """Compute sums for all APs."""
        sums = np.zeros(num_aps, dtype=np.float64)
        for i, ap_arr in enumerate(ap_indices):
            sums[i] = np.sum(signs[ap_arr])
        return sums

    def compute_grad_l_p(signs, sums, p):
        """Compute gradient of sum |S_k|^p w.r.t signs."""
        grad = np.zeros(n, dtype=np.float64)
        # weight for each AP k is p * |S_k|^(p-1) * sign(S_k)
        # Note: sign(0) is 0, so if S_k is 0, weight is 0.
        abs_sums = np.abs(sums)
        # Avoid 0^(p-1) issues if p < 1, but p >= 2 here.
        # For p=2, weight = 2 * S_k.
        # For p=8, weight = 8 * |S_k|^7 * sign(S_k).
        
        # Vectorized weight calculation
        # Handle S_k = 0 carefully. np.sign(0) is 0.
        weights = p * (abs_sums ** (p - 1)) * np.sign(sums)
        
        # Scatter-add weights to gradient
        # For each AP, add its weight to all positions in the AP.
        for i, ap_arr in enumerate(ap_indices):
            if weights[i] != 0:
                grad[ap_arr] += weights[i]
                
        return grad

    def local_search(signs, sums):
        """Greedy local search to reduce max AP sum."""
        # signs is a numpy array of -1, 1
        # sums is a numpy array of current AP sums
        # Returns improved signs and sums
        
        max_abs_sum = np.max(np.abs(sums))
        
        improved = True
        while improved:
            improved = False
            # Find the AP with the largest absolute sum
            # There might be multiple, pick one
            worst_idx = np.argmax(np.abs(sums))
            worst_ap = ap_indices[worst_idx]
            current_worst_val = sums[worst_idx]
            
            # Try flipping each element in the worst AP
            for pos in worst_ap:
                # Flip sign at pos
                signs[pos] *= -1
                
                # Update sums for all APs containing pos
                # We need to revert this if it doesn't improve
                affected_aps = pos_to_aps[pos]
                old_sums = np.array(sums[affected_aps])
                
                # The change in sum for any AP containing pos is -2 * signs[pos] (new sign)
                # Wait, signs[pos] is the NEW sign.
                # Old sum = S. New sum = S - 2 * old_sign.
                # Since we flipped, new_sign = -old_sign.
                # So New sum = S + 2 * new_sign.
                # Let's just recalculate or update.
                # Update: sums[ap] += -2 * signs[pos] ? No.
                # Let old_sign = -signs[pos] (current is new).
                # sum_new = sum_old - 2 * old_sign = sum_old + 2 * signs[pos].
                delta = 2.0 * signs[pos]
                
                for ap_idx in affected_aps:
                    sums[ap_idx] += delta
                
                new_max_abs = np.max(np.abs(sums))
                
                if new_max_abs < max_abs_sum:
                    # Accept flip
                    max_abs_sum = new_max_abs
                    improved = True
                    # Keep the flip and updated sums
                    break
                else:
                    # Revert flip
                    signs[pos] *= -1
                    # Revert sums
                    for ap_idx in affected_aps:
                        sums[ap_idx] -= delta
                        
        return signs, sums

    # 3. Main Optimization Loop
    
    best_signs = None
    best_score = float('inf')
    
    num_restarts = 6
    p_norm = 8.0 # Use L8 to approximate L-infinity better than L2
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Initialize x
        x = np.random.uniform(-1, 1, n)
        
        # Adam parameters
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        steps = 200
        
        m = np.zeros(n)
        v = np.zeros(n)
        t = 0
        
        for step in range(steps):
            if time.time() - start_time > time_limit - 3.0:
                break
                
            t += 1
            
            # Compute sums using current x (continuous)
            # Note: The loss is defined on the continuous x.
            # S_k = sum(x[ap_k])
            sums = np.zeros(num_aps)
            for i, ap_arr in enumerate(ap_indices):
                sums[i] = np.sum(x[ap_arr])
                
            # Compute gradient
            grad = compute_grad_l_p(x, sums, p_norm)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            x -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Project to [-1, 1] to keep it bounded, though sign() will handle it later
            # Clipping helps with stability
            x = np.clip(x, -1.0, 1.0)
            
        # Discretize
        signs = np.sign(x)
        signs[signs == 0] = 1.0
        
        # Compute initial sums for local search
        sums = compute_sums(signs)
        
        # Local Search
        ls_start = time.time()
        ls_limit = 5.0
        
        while (time.time() - ls_start) < ls_limit:
            signs, sums = local_search(signs, sums)
            # Check if we reached 0
            if np.max(np.abs(sums)) == 0:
                break
                
        score = np.max(np.abs(sums))
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
            
    if best_signs is None:
        best_signs = np.random.choice([-1, 1], n)
        
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
