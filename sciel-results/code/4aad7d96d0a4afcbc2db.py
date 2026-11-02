import numpy as np
import time

N = 100
W = 2.0 / N

def _compute_overlaps(h):
    """
    Compute the overlap for all shifts k.
    overlap[k] = W * (sum_h - autocorr[k])
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    sum_h = np.sum(h)
    overlaps = W * (sum_h - autocorr)
    return overlaps

def _soft_max(overlaps, beta):
    """
    Smooth approximation of the maximum.
    """
    # Numerical stability
    max_ov = np.max(overlaps)
    # If beta is very large, this approaches max
    # Using log-sum-exp trick
    shifted = beta * (overlaps - max_ov)
    # Clip to avoid overflow in exp, though shifted <= 0
    exp_sum = np.sum(np.exp(shifted))
    return max_ov + (1.0 / beta) * np.log(exp_sum)

def _grad_soft_max(overlaps, beta):
    """
    Gradient of the soft max with respect to overlaps.
    d/dx (1/beta log sum exp(beta x_i)) = exp(beta x_i) / sum exp(beta x_j)
    """
    max_ov = np.max(overlaps)
    shifted = beta * (overlaps - max_ov)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    return exp_vals / sum_exp

def _project_to_simplex_box(h, target_sum):
    """
    Project h onto the intersection of the box [0,1] and the hyperplane sum(h) = target_sum.
    This is a standard quadratic programming problem (projection onto a polytope).
    We use a simple iterative water-filling or binary search on the Lagrange multiplier.
    """
    # Binary search for lambda such that sum(clamp(h - lambda, 0, 1)) = target_sum
    # Let f(lambda) = sum(clamp(h_i - lambda, 0, 1))
    # f is monotonic decreasing in lambda.
    
    # Bounds for lambda
    # If lambda is very large, all become 0, sum=0.
    # If lambda is very small (negative), all become 1, sum=N.
    # We need sum = target_sum (which is N/2).
    
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    # Final projection with the best lambda
    lambda_opt = (lo + hi) / 2.0
    proj = np.clip(h - lambda_opt, 0.0, 1.0)
    return proj

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy: Continuous Simulated Annealing with Gradient Descent
    # Objective: Minimize Soft-Max Overlap
    
    best_h = np.full(N, 0.5)
    best_score = np.inf  # We track the raw overlap value, lower is better
    
    # Initial randomization
    # Start with a perturbed uniform distribution
    np.random.seed(42)
    
    # Parameters
    beta = 5.0  # Sharpness of the max approximation
    beta_max = 50.0
    learning_rate = 0.01
    noise_scale = 0.05
    noise_decay = 0.99
    
    max_iters = 2000
    
    # Precompute shift matrices? No, FFT is fast enough.
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        # Generate a random starting point for this "epoch" or just continue?
        # Let's do a single long annealing run with restarts if stuck.
        # For simplicity, one long run with adaptive noise.
        
        if i == 0:
            # Initial h: Uniform + small noise, then project
            h = np.full(N, 0.5) + np.random.normal(0, 0.01, N)
            h = _project_to_simplex_box(h, N / 2.0)
        else:
            # This block is for the case where we restart, but we are doing a single loop.
            # In this structure, h persists.
            pass

        # Compute current overlaps
        overlaps = _compute_overlaps(h)
        current_val = np.max(overlaps)
        
        if current_val < best_score:
            best_score = current_val
            best_h = h.copy()
            
        # Compute gradient of Soft-Max Overlap w.r.t h
        # L = (1/beta) log sum exp(beta * overlap_k)
        # dL/dh = sum_k (dL/doverlap_k) * (doverlap_k/dh)
        # doverlap_k/dh = W * (-d(autocorr_k)/dh)
        # autocorr_k = sum_j h_j h_{j+k}
        # d(autocorr_k)/dh_i = h_{i-k} + h_{i+k} (with periodic boundaries)
        # So doverlap_k/dh = -W * (h_{i-k} + h_{i+k})
        
        # Weights for the soft max
        weights = _grad_soft_max(overlaps, beta)
        
        # Gradient of the objective
        grad_obj = np.zeros(N)
        # Efficiently compute the weighted sum of derivatives
        # dL/dh_i = -W * sum_k weights[k] * (h_{i-k} + h_{i+k})
        # Note: sum_k weights[k] h_{i-k} is a convolution of weights and h.
        # Let W_k = weights[k].
        # Conv1 = convolve(h, W) where W is symmetric?
        # autocorr is symmetric, so overlaps are symmetric?
        # overlap[k] = W(N/2 - autocorr[k]). autocorr[k] = autocorr[-k].
        # So weights[k] = weights[-k].
        # The term sum_k weights[k] h_{i-k} is the circular convolution of h and weights.
        
        H = np.fft.fft(h)
        Wk = np.fft.fft(weights)
        conv = np.fft.ifft(H * Wk).real
        
        # The derivative is -W * 2 * conv?
        # Let's verify:
        # sum_k w_k (h_{i-k} + h_{i+k})
        # Let j = -k. sum_j w_{-j} h_{i-j} + sum_k w_k h_{i+k}
        # If w is symmetric, w_{-j} = w_j.
        # Both terms are the circular convolution of h and w.
        # So sum_k w_k (h_{i-k} + h_{i+k}) = 2 * (h * w)_i
        
        grad_obj = -W * 2.0 * conv
        
        # Gradient Descent Step
        # Add noise for exploration
        noise = np.random.normal(0, noise_scale, N)
        h_new = h - learning_rate * grad_obj + noise
        
        # Project back to feasible region
        h_new = _project_to_simplex_box(h_new, N / 2.0)
        
        # Acceptance criterion (Simple Hill Climbing with Noise)
        # Since we are using a smooth approximation, we can just accept if it improves the soft objective,
        # or use the raw max improvement.
        # Let's use the raw max overlap for acceptance to avoid getting stuck in soft-max artifacts.
        overlaps_new = _compute_overlaps(h_new)
        current_val_new = np.max(overlaps_new)
        
        # Accept if better, or with some probability if worse (Annealing)
        if current_val_new < current_val:
            h = h_new
        else:
            # Annealing: accept worse moves with probability exp(-delta / T)
            # T is effectively controlled by noise_scale and lr
            # Simple heuristic: accept with small probability if not much worse
            if np.random.random() < 0.1 and current_val_new < current_val + 0.01:
                h = h_new
            else:
                # Keep old h
                pass
                
        # Anneal parameters
        # Increase beta to sharpen the max
        beta = min(beta * 1.005, beta_max)
        # Decrease noise
        noise_scale *= noise_decay
        # Adapt learning rate?
        # Keep lr constant for now, projection handles bounds.
        
    # Final polish: A few steps of pure gradient descent on the raw max (subgradient)
    # Or just return the best found.
    
    # Ensure exact sum and bounds
    best_h = _project_to_simplex_box(best_h, N / 2.0)
    
    # Final sanity check
    if not np.isclose(np.sum(best_h), N/2.0, atol=1e-5):
        # Fallback
        best_h = np.full(N, 0.5)
        
    return best_h
