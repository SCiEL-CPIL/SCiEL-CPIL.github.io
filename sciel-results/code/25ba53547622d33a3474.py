import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    N_local = N
    target_sum = N_local / 2.0
    
    # Initialize with uniform distribution
    h = np.full(N_local, 0.5)
    
    # Parameters for the iteration
    num_iters = 500
    learning_rate = 0.05
    tau = 0.1  # Temperature for soft-min
    
    def project(h_vec):
        """
        Project h_vec onto the set {h : sum(h) = target_sum, 0 <= h_i <= 1}.
        Uses bisection on the shift lambda to satisfy the sum constraint after clipping.
        """
        # We want to find lambda such that sum(clip(h_vec + lambda, 0, 1)) = target_sum
        # The function f(lambda) = sum(clip(h_vec + lambda, 0, 1)) is non-decreasing in lambda.
        
        # Estimate bounds for lambda
        # Min possible sum is 0, max is N.
        # If sum(h_vec) is far from target, lambda might be large.
        # But h_vec is usually close to [0,1].
        
        def sum_clipped(lambda_val):
            return np.sum(np.clip(h_vec + lambda_val, 0, 1))
        
        # Bisection search for lambda
        lo = -1.0
        hi = 1.0
        
        # Ensure bracket
        if sum_clipped(lo) > target_sum:
            # Target is smaller than min possible sum? Impossible if target_sum=N/2 and h in [0,1]
            # But if h_vec has huge negative values? Unlikely.
            # Just shift lo down.
            while sum_clipped(lo) > target_sum:
                lo -= 1.0
        if sum_clipped(hi) < target_sum:
            while sum_clipped(hi) < target_sum:
                hi += 1.0
                
        for _ in range(50): # 50 bisection steps gives high precision
            mid = (lo + hi) / 2
            s_mid = sum_clipped(mid)
            if s_mid < target_sum:
                lo = mid
            else:
                hi = mid
        
        lambda_opt = (lo + hi) / 2
        return np.clip(h_vec + lambda_opt, 0, 1)

    for _ in range(num_iters):
        # Compute autocorrelations
        # A_s = sum h_i h_{i+s}
        # Using FFT for efficiency? N=100 is small, direct calculation is O(N^2) = 10,000 ops.
        # 500 iters * 10,000 ops = 5,000,000 ops. Very fast.
        
        A = np.zeros(N_local)
        for s in range(N_local):
            A[s] = np.dot(h, np.roll(h, s))
            
        # Compute weights for soft-min
        # We want to maximize min(A).
        # Soft-min approximation: -tau * log(sum(exp(-A/tau)))
        # Gradient of soft-min w.r.t A_s is -exp(-A_s/tau) / sum(exp(-A'/tau))
        # We want to maximize soft-min, so we add gradient of soft-min.
        # Grad_softmin_s = -w_s
        # Wait, we are maximizing.
        # Let f(h) = softmin(A(h)).
        # grad_f = sum_s (d f / d A_s) * (d A_s / d h)
        # d f / d A_s = -exp(-A_s/tau) / Z, where Z = sum exp(-A'/tau).
        # Note: This is negative.
        # So we are moving in the direction of - sum w_s grad A_s ?
        # Let's check signs.
        # If A_s is small, exp(-A_s/tau) is large, so w_s is large.
        # d f / d A_s is large negative.
        # We want to increase A_s.
        # The gradient of the objective (which we want to maximize) is sum (d f / d A_s) grad A_s.
        # Since d f / d A_s is negative, the gradient points in the direction that *decreases* A_s?
        # No, the gradient points in the direction of steepest ascent.
        # If A_s is small, we want to increase it.
        # The term for small A_s has large magnitude.
        # Let's just compute the vector D = sum_s w_s * grad A_s, where w_s = exp(-A_s/tau)/Z.
        # Then we update h = h + eta * D?
        # If we increase h in the direction of grad A_s, A_s increases.
        # But the weight is associated with the "need" to increase.
        # Actually, the gradient of the *negative* soft-min (which we minimize) is:
        # g = - sum_s w_s grad A_s.
        # We do gradient descent on this: h = h - eta * g = h + eta * sum_s w_s grad A_s.
        # So yes, we add the weighted sum of gradients.
        
        exp_term = np.exp(-A / tau)
        Z = np.sum(exp_term)
        weights = exp_term / Z
        
        # Compute weighted gradient of A
        # grad A_s w.r.t h_k is h_{k-s} + h_{k+s}
        # D_k = sum_s weights[s] * (h_{k-s} + h_{k+s})
        D = np.zeros(N_local)
        for s in range(N_local):
            # Contribution of shift s
            # grad A_s is h shifted by -s plus h shifted by s
            # h_{k-s} is np.roll(h, s) ?
            # np.roll(h, s) moves element at i to i+s.
            # So (np.roll(h, s))_k = h_{k-s}.
            # (np.roll(h, -s))_k = h_{k+s}.
            grad_A_s = np.roll(h, s) + np.roll(h, -s)
            D += weights[s] * grad_A_s
            
        # Update
        h = h + learning_rate * D
        
        # Project
        h = project(h)
        
    # Final check and adjustment
    h = np.clip(h, 0, 1)
    # Ensure exact sum
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Simple redistribution
        # Add diff/N, clip, repeat
        for _ in range(100):
            h = h + diff / N_local
            h = np.clip(h, 0, 1)
            new_diff = target_sum - np.sum(h)
            if abs(new_diff) < 1e-12:
                break
            diff = new_diff
            
    return h
# EVOLVE-BLOCK-END
