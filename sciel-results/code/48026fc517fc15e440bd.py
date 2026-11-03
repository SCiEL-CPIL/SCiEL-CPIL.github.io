import numpy as np
import time

N = 50

def _compute_ratio(h):
    """
    Computes the ratio: ||h||_2^2 / (||h||_1 * sup_{k != 0} |(h*h)_k|)
    Assumes h is non-negative.
    If the max off-diagonal peak is 0, return a large number (or 1 if h is 0).
    """
    if np.all(h == 0):
        return 0.0
    
    norm1 = np.sum(h)
    if norm1 == 0:
        return 0.0
        
    norm2_sq = np.sum(h**2)
    
    # Compute autoconvolution
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    
    # The center index for linear convolution of two length N vectors is N-1
    center_idx = N - 1
    
    # We need sup over k != 0.
    # In the convolution result, index `j` corresponds to lag `j - (N-1)`.
    # So we exclude the center index.
    
    # To be safe, if the problem implies circular convolution, we should use that.
    # But "equal intervals" and "autoconvolution" often implies linear in signal processing contexts unless specified.
    # However, Sidon sets on a circle (Z_N) are also common.
    # Let's check the seed behavior. The seed uses indices 0..49.
    # If it were circular, a Sidon set in Z_50 would have specific properties.
    # Given the ambiguity, I will try Linear first, but keep Circular in mind.
    # Actually, the "second autocorrelation inequality" often refers to:
    # ||h||_2^2 <= C ||h||_1 ||h*h||_inf
    # If the constant C is what we are bounding, we want to maximize the LHS/RHS.
    
    # Let's assume the peak excludes the main lobe (k=0).
    peaks = h_conv.copy()
    peaks[center_idx] = 0.0
    
    max_peak = np.max(peaks)
    
    if max_peak < 1e-12:
        return norm2_sq / norm1 # If off-diagonal is 0, ratio is infinite? Or just return high value.
        
    return (norm2_sq / norm1) / max_peak

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We will optimize x such that h = x^2.
    # Initialize with a random sparse vector or a Sidon-like set.
    
    # Strategy:
    # 1. Generate a good initial support (e.g. random Sidon-like or just random sparse).
    # 2. Optimize the weights x on that support using stochastic gradient descent.
    # 3. Occasionally restart with a new random support.
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Helper to compute gradient of the objective w.r.t x
    # Objective: Maximize J(x) = (sum x_i^4) / (sum x_i^2 * max_{k!=0} sum x_i^2 x_{i+k}^2)
    # Let A = sum x_i^4, B = sum x_i^2, C = max_{k!=0} sum x_i^2 x_{i+k}^2.
    # J = A / (B * C).
    # log J = log A - log B - log C.
    # d(log J)/dx = d(log A)/dx - d(log B)/dx - d(log C)/dx.
    
    def grad_log_J(x, active_lag):
        h = x**2
        A = np.sum(h**2)
        B = np.sum(h)
        
        # Compute autoconvolution
        h_conv = np.convolve(h, h, mode='full')
        center_idx = N - 1
        
        # Find max peak excluding center
        # We need the gradient of C = max_k Conv_k.
        # If we sample the active lag k*, the gradient is the gradient of Conv_{k*}.
        
        # d A / d x_i = 4 x_i^3
        grad_log_A = 4 * x**3 / A if A > 0 else np.zeros(N)
        
        # d B / d x_i = 2 x_i
        grad_log_B = 2 * x / B if B > 0 else np.zeros(N)
        
        # d C / d x_i for a fixed lag k*
        # C_k = sum_j h_j h_{j+k} = sum_j x_j^2 x_{j+k}^2
        # d C_k / d x_i = 2 x_i (x_{i-k}^2 + x_{i+k}^2) (handling boundaries)
        
        # Let's compute the gradient for the sampled lag
        k_star = active_lag # This is the lag in the original signal, not the conv index
        
        # We need to be careful with indices.
        # If k_star > 0:
        # Term 1: x_i is part of x_j where j = i. Pair (i, i+k).
        # Term 2: x_i is part of x_{j+k} where j+k = i => j = i-k. Pair (i-k, i).
        
        grad_C = np.zeros(N)
        
        if k_star > 0:
            # Pairs (j, j+k)
            # For i in range(N-k_star):
            #   d/dx_i (x_i^2 x_{i+k}^2) = 2 x_i x_{i+k}^2
            #   d/dx_{i+k} (x_i^2 x_{i+k}^2) = 2 x_{i+k} x_i^2
            for i in range(N - k_star):
                grad_C[i] += 2 * x[i] * x[i+k_star]**2
                grad_C[i+k_star] += 2 * x[i+k_star] * x[i]**2
        elif k_star < 0:
            # Symmetric
            k_abs = -k_star
            for i in range(N - k_abs):
                grad_C[i+k_abs] += 2 * x[i+k_abs] * x[i]**2
                grad_C[i] += 2 * x[i] * x[i+k_abs]**2
        else:
            # k=0 is excluded from max, so this shouldn't happen if we sample correctly.
            pass
            
        # If C is 0, gradient is undefined, but we can return 0 or handle it.
        C_val = np.sum(h * np.roll(h, k_star)) # Note: np.roll is circular. 
        # Wait, my manual loop above was linear.
        # I should stick to one definition.
        # Let's use the linear convolution peak value for the scalar C.
        
        # Recompute C_val for the specific lag k_star using linear logic
        C_val = 0.0
        if k_star > 0:
            for i in range(N - k_star):
                C_val += x[i]**2 * x[i+k_star]**2
        elif k_star < 0:
            for i in range(N - abs(k_star)):
                C_val += x[i+abs(k_star)]**2 * x[i]**2
        else:
            C_val = np.sum(x**4) # Should not be used
            
        if C_val < 1e-12:
            grad_log_C = np.zeros(N)
        else:
            grad_log_C = grad_C / C_val
            
        return grad_log_A - grad_log_B - grad_log_C

    # Optimization loop
    max_restarts = 50
    steps_per_restart = 500
    
    for r in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize x
        # Pick a random sparse support
        num_spikes = np.random.randint(5, 15)
        support = np.random.choice(N, num_spikes, replace=False)
        
        x = np.zeros(N)
        x[support] = 1.0 / np.sqrt(num_spikes) # Normalize roughly
        
        current_score = _compute_ratio(x**2)
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            h = x**2
            current_score = _compute_ratio(h)
            
            if current_score > best_score:
                best_score = current_score
                best_h = h.copy()
                
            # Sample active lag
            # Compute all peaks to find the max one
            h_conv = np.convolve(h, h, mode='full')
            center_idx = N - 1
            
            # Find indices of peaks
            # We need to map conv index to lag
            # Lag k corresponds to index N-1+k
            # We want k != 0
            
            max_val = -1
            max_lag = 0
            for k in range(1, N):
                # Check k and -k
                idx_pos = N - 1 + k
                if idx_pos < len(h_conv):
                    val = h_conv[idx_pos]
                    if val > max_val:
                        max_val = val
                        max_lag = k
                idx_neg = N - 1 - k
                if idx_neg >= 0:
                    val = h_conv[idx_neg]
                    if val > max_val:
                        max_val = val
                        max_lag = -k
            
            if max_val < 1e-12:
                break
                
            # Compute gradient
            grad = grad_log_J(x, max_lag)
            
            # Learning rate
            lr = 0.01 / (1 + step * 0.01)
            
            # Update x
            x = x + lr * grad
            
            # Ensure x is non-negative? h = x^2 is always non-negative.
            # But x can be negative. That's fine.
            # However, if x becomes 0, it stays 0?
            # We can add a small penalty or just let it be.
            
            # Periodic re-normalization?
            # The objective is scale invariant for h?
            # J(c h) = (c^2 ||h||_2^2) / (c ||h||_1 * c ||h*h||_inf) = J(h).
            # So scale doesn't matter.
            
            # To prevent x from drifting to infinity or 0, we can normalize x
            # such that sum x^2 = 1?
            norm_x = np.linalg.norm(x)
            if norm_x > 1e-9:
                x = x / norm_x
                
    return list(best_h)
