"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for evaluation overhead
    
    # Precompute M: M[i, j] = count of APs (length > 1) containing both i and j.
    # This allows us to compute the gradient of sum_{AP} (sum_{i in AP} x_i)^2 efficiently as 2*M*x.
    # We also need to handle the "max" aspect better. Sum of squares minimizes variance, but max is the target.
    # A better surrogate for max is a soft-max or a higher power.
    # Let's stick to sum of squares for the main gradient but add a penalty for large sums if possible.
    # Actually, minimizing sum of squares is a good proxy for minimizing max if the distribution is tight.
    # To specifically target the max, we can use a weighted sum where weights are based on current sums?
    # Or use L4 norm: sum(ap_sum^4). Grad = 4 * sum(ap_sum^2 * 1_ap).
    # This requires storing ap_sums.
    
    # Let's generate all APs and store them as a list of index arrays.
    # Also store the start/step to allow fast sum calculation?
    # No, just index arrays.
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
    
    # Number of APs is ~180k.
    # Storing 180k arrays of variable length is okay.
    
    # Build M for the L2 surrogate (sum of squares).
    # M[i, j] = number of APs containing i and j.
    M = np.zeros((n, n), dtype=np.float64)
    for idx_arr in ap_indices:
        M[np.ix_(idx_arr, idx_arr)] += 1.0
    
    # Optimization strategy:
    # 1. Initialize x.
    # 2. Iterate:
    #    a. Compute current AP sums: S = [x[idx_arr].sum() for idx_arr in ap_indices]
    #    b. Compute a "soft max" gradient or use the L4 gradient: grad = 4 * sum(S^2 * 1_AP)
    #       This emphasizes large sums more than L2.
    #    c. Add regularization to push x to +/-1.
    #    d. Update x.
    #    e. Periodically discretize x to +/-1 to escape local minima and refine the discrete solution.
    
    # Let's try a hybrid:
    # - Use L2 gradient (via M) for speed.
    # - Add a term that penalizes large sums more heavily?
    # - Actually, let's just run many iterations of L2 minimization with strong regularization.
    # - Then, perform local search on the discrete signs.
    
    # But the directive says "gradient-based SGD ... smooth surrogate loss".
    # Let's use a loss that approximates max better: sum(exp(S^2 / T)) is hard to differentiate efficiently.
    # sum(S^4) is a good differentiable surrogate for max.
    # Grad of sum(S^4) = 4 * sum_{AP} S_AP^2 * 1_AP.
    # This is equivalent to: For each i, grad_i = 4 * sum_{AP containing i} S_AP^2.
    # We can compute this efficiently if we have S.
    # S has length ~180k.
    # We can compute S in a loop over APs (180k iterations, each doing a sum over ~2-10 elements on avg).
    # Then compute grad:
    # grad = np.zeros(n)
    # for idx_arr, s in zip(ap_indices, S):
    #     grad[idx_arr] += 4 * s * s
    # This is 180k iterations. In Python, this might be slow if done naively.
    # 180k * (average len) operations. Average len is small.
    # Let's vectorize the sum calculation?
    # Hard to vectorize variable length sums.
    
    # Alternative: Use the L2 gradient (M*x) which is O(n^2) per step (very fast).
    # L2 minimization tends to make all AP sums small on average.
    # Combined with strong regularization to +/-1, this often yields good results.
    # Let's enhance the L2 approach with adaptive learning rate and multiple restarts.
    
    def evaluate_discrete(signs):
        """Calculate the maximum absolute AP sum for a discrete sign sequence."""
        max_sum = 0
        # We can vectorize this or just loop.
        # Looping over 180k APs and summing is fast in C if we use numpy arrays.
        for idx_arr in ap_indices:
            s = np.sum(signs[idx_arr])
            abs_s = abs(s)
            if abs_s > max_sum:
                max_sum = abs_s
        return max_sum
    
    best_signs = None
    best_score = float('inf')
    
    # Try multiple random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize x
        x = np.random.uniform(-1, 1, n)
        
        # SGD parameters
        lr = 0.05
        lam = 50.0 # Strong regularization
        steps = 300
        
        for t in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            # Gradient of L2 loss: 2 * M * x
            grad_L2 = 2.0 * M.dot(x)
            
            # Gradient of regularization: 4 * lam * x * (x^2 - 1)
            grad_reg = 4.0 * lam * x * (x**2 - 1.0)
            
            grad = grad_L2 + grad_reg
            
            # Update
            x -= lr * grad
            
            # Project to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            
            # Periodically discretize to escape local minima?
            # Let's try a "stochastic rounding" or just snap every few steps?
            # Snapping too early might trap in local minima.
            # Let's keep it continuous for the first 200 steps, then snap and refine?
            
        # Discretize
        signs = np.sign(x)
        signs[signs == 0] = 1.0
        
        # Evaluate
        score = evaluate_discrete(signs)
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
            
        # If time allows, do a local search on best_signs
        if time.time() - start_time < time_limit - 5.0 and best_signs is not None:
            # Simple local search: flip one sign if it improves the max sum
            # This is expensive if done naively (256 flips * 180k APs).
            # We can optimize by only checking APs affected by the flip.
            # But for simplicity, let's do a few random flips and check.
            
            for _ in range(10): # 10 random perturbation attempts
                if time.time() - start_time > time_limit:
                    break
                
                trial_signs = best_signs.copy()
                # Flip a random sign
                idx = np.random.randint(n)
                trial_signs[idx] *= -1
                
                trial_score = evaluate_discrete(trial_signs)
                
                if trial_score < best_score:
                    best_score = trial_score
                    best_signs = trial_signs
    
    # Final check
    if best_signs is None:
        # Fallback: random signs
        best_signs = np.random.choice([-1, 1], n)
    
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
