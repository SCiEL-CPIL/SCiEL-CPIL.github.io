# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0
    
    # Precompute APs
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
    
    # Build M for L2 gradient
    M = np.zeros((n, n), dtype=np.float64)
    for idx_arr in ap_indices:
        M[np.ix_(idx_arr, idx_arr)] += 1.0

    def evaluate_discrete(signs):
        """Calculate the maximum absolute AP sum for a discrete sign sequence."""
        max_sum = 0
        # Using a list comprehension or loop. 
        # np.sum is fast.
        for idx_arr in ap_indices:
            s = np.sum(signs[idx_arr])
            abs_s = abs(s)
            if abs_s > max_sum:
                max_sum = abs_s
        return max_sum

    def adam_step(x, grad, m, v, t, lr, beta1, beta2, eps, lam):
        # Update bias-corrected moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Regularization gradient
        grad_reg = 4.0 * lam * x * (x**2 - 1.0)
        total_grad = m_hat + grad_reg # Wait, Adam usually applies to the total gradient.
        
        # Let's compute total grad outside and pass it? 
        # Or compute total grad here?
        # Better to compute total grad in the main loop.
        pass

    best_signs = None
    best_score = float('inf')
    
    num_restarts = 4
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize x
        x = np.random.uniform(-1, 1, n)
        
        # Adam parameters
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lam = 10.0
        steps = 400
        
        m = np.zeros(n)
        v = np.zeros(n)
        t = 0
        
        for step in range(steps):
            if time.time() - start_time > time_limit - 2.0:
                break
            
            t += 1
            # Gradient of L2 loss: 2 * M * x
            grad_L2 = 2.0 * M.dot(x)
            # Gradient of regularization
            grad_reg = 4.0 * lam * x * (x**2 - 1.0)
            
            grad = grad_L2 + grad_reg
            
            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            x -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Project to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            
        # Discretize
        signs = np.sign(x)
        signs[signs == 0] = 1.0
        
        # Local Search: Greedy flip on worst APs
        # Limit local search time
        ls_start = time.time()
        ls_limit = 5.0 # 5 seconds per restart
        
        improved = True
        while improved and (time.time() - ls_start) < ls_limit:
            improved = False
            # Find current max and the AP index
            current_max = 0
            worst_ap_idx = -1
            worst_ap_sum = 0
            for i, idx_arr in enumerate(ap_indices):
                s = np.sum(signs[idx_arr])
                abs_s = abs(s)
                if abs_s > current_max:
                    current_max = abs_s
                    worst_ap_idx = i
                    worst_ap_sum = s
            
            # Try flipping signs in the worst AP
            if worst_ap_idx != -1:
                idx_arr = ap_indices[worst_ap_idx]
                # Try flipping each element in this AP
                for pos in range(len(idx_arr)):
                    idx_to_flip = idx_arr[pos]
                    # Flip
                    signs[idx_to_flip] *= -1
                    
                    # Evaluate new max
                    new_max = 0
                    # Optimization: We can just check if the max decreased.
                    # But we need to ensure no other AP became larger.
                    # Full evaluation is ~0.1s. 
                    # If we do this for every pos (len ~5-10), it's 0.5-1s per iteration.
                    # This is too slow for many iterations.
                    
                    # Let's just do a few random flips in the worst AP instead of exhaustive?
                    # Or just one flip: the one that most reduces the sum of the worst AP?
                    # If worst_ap_sum > 0, we want to flip a +1 to -1.
                    # If worst_ap_sum < 0, we want to flip a -1 to +1.
                    
                    signs[idx_to_flip] *= -1 # Unflip
                    
                    # Let's pick the "best" flip in this AP heuristically:
                    # Flip the sign that has the same sign as the sum (to reduce magnitude).
                    # If sum > 0, flip a +1. If sum < 0, flip a -1.
                    # If multiple, pick the first one.
                    target_sign = 1 if worst_ap_sum > 0 else -1
                    candidates = [p for p in range(len(idx_arr)) if signs[idx_arr[p]] == target_sign]
                    
                    if candidates:
                        pos = candidates[0]
                        idx_to_flip = idx_arr[pos]
                        signs[idx_to_flip] *= -1
                        
                        # Check if this improved the global max
                        new_max = evaluate_discrete(signs)
                        if new_max < current_max:
                            improved = True
                            # Keep the flip
                            break # Restart the while loop to find the new worst AP
                        else:
                            # Revert
                            signs[idx_to_flip] *= -1
                            # Try next candidate? Or give up on this AP?
                            # Let's try the next candidate
                            # Actually, let's just try one flip per worst AP identification to save time.
                            # If it fails, we might be stuck. 
                            # Let's try all candidates in this AP if time permits.
                            # But for now, let's just try the first one.
                            # If it fails, we mark improved=False for this AP and break?
                            # No, let's try the next candidate.
                            # To keep it simple and fast, let's just try the first candidate.
                            # If it doesn't work, we stop local search for this restart?
                            # That's risky.
                            # Let's try all candidates in the worst AP.
                            
                            # Re-implementing the loop to try all candidates
                            found_improvement = False
                            for p in candidates:
                                idx_to_flip = idx_arr[p]
                                signs[idx_to_flip] *= -1
                                new_max = evaluate_discrete(signs)
                                if new_max < current_max:
                                    found_improvement = True
                                    improved = True
                                    break
                                else:
                                    signs[idx_to_flip] *= -1
                            
                            if found_improvement:
                                break # Restart while loop
                            else:
                                # No flip in this AP improved the max
                                # We could try flipping in the 2nd worst AP, etc.
                                # But for simplicity, let's stop if the worst AP can't be improved.
                                # This might be a local minimum.
                                break
                    else:
                        # No candidate to flip in this direction?
                        # This shouldn't happen if sum != 0.
                        break

        score = evaluate_discrete(signs)
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
            
    if best_signs is None:
        best_signs = np.random.choice([-1, 1], n)
        
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
