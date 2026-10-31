import numpy as np
from scipy.fft import fft, ifft
from scipy import optimize
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    # Use FFT for efficient convolution
    size = 2 * N
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*N-1])
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Initial: uniform distribution
    h_uniform = np.ones(N) / N
    best_h = h_uniform.copy()
    best_score = compute_autoconv_max(best_h)
    
    # Soft-max objective with gradient
    def soft_max_objective_and_grad(x, T):
        """
        Compute soft-max of autoconvolution and its gradient.
        x: array of length N (non-negative, sum=1)
        T: temperature parameter
        Returns: (objective_value, gradient)
        """
        h = np.maximum(x, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10, np.zeros(N)
        h = h / s  # Project to simplex
        
        # Compute autoconvolution via FFT
        size = 2 * N
        H = fft(h, size)
        conv = np.real(ifft(H * H, size)[:2*N-1])
        
        conv_max = np.max(conv)
        
        # Soft-max: conv_max + T * log(sum(exp((conv - conv_max)/T)))
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -50, 50)  # Prevent overflow
        log_sum_exp = conv_max / T + np.log(np.sum(np.exp(shifted)))
        obj = log_sum_exp * T  # This is the soft-max value
        
        # Gradient: d(obj)/d(h_i)
        # soft_max = max(conv) + T * log(sum_j exp((conv_j - max)/T))
        # d(soft_max)/d(conv_k) = exp((conv_k - max)/T) / sum_j exp((conv_j - max)/T)
        weights = np.exp(shifted)
        weights = weights / np.sum(weights)
        
        # d(conv_k)/d(h_i) = h_{k-i} + h_{i-k} (for linear convolution)
        # conv_k = sum_j h_j * h_{k-j}
        # d(conv_k)/d(h_i) = h_{k-i} (if 0<=k-i<N) + h_{i-k} (if 0<=i-k<N)
        
        # Compute gradient efficiently:
        # grad_i = sum_k weights_k * (h_{k-i} + h_{i-k})
        # This is a cross-correlation of weights with h
        
        # For the first term: sum_k weights_k * h_{k-i}
        # This is correlation of weights (length 2N-1) with h (length N)
        # For the second term: sum_k weights_k * h_{i-k}
        # This is correlation of weights with reversed h
        
        # Use FFT for efficient computation
        # Pad weights to size 2N
        W = fft(weights, size)
        H_pad = fft(h, size)
        
        # First correlation: sum_k weights_k * h_{k-i}
        # This is like (weights * h) but we need to be careful with indices
        # Let's compute it directly using the structure
        
        # Actually, let's compute the gradient more carefully:
        # conv_k = sum_{j=max(0,k-N+1)}^{min(k,N-1)} h_j * h_{k-j}
        # d(conv_k)/d(h_i) = h_{k-i} (if 0<=k-i<N) + h_{i-k} (if 0<=i-k<N, and i!=k-i to avoid double counting, but since we sum both terms, it's fine)
        
        # grad_i = sum_k weights_k * [h_{k-i} * I(0<=k-i<N) + h_{i-k} * I(0<=i-k<N)]
        
        # First part: sum_k weights_k * h_{k-i} for k-i in [0,N-1], i.e., k in [i, i+N-1]
        # Second part: sum_k weights_k * h_{i-k} for i-k in [0,N-1], i.e., k in [i-N+1, i]
        
        # Let's compute these using FFT-based correlation
        # Part 1: For each i, sum_{k=i}^{i+N-1} weights_k * h_{k-i}
        # Let m = k-i, then sum_{m=0}^{N-1} weights_{i+m} * h_m
        # This is a sliding dot product
        
        # Part 2: For each i, sum_{k=max(0,i-N+1)}^{i} weights_k * h_{i-k}
        # Let m = i-k, then sum_{m=0}^{min(i,N-1)} weights_{i-m} * h_m
        
        # Combined: grad_i = sum_{m=0}^{N-1} h_m * (weights_{i+m} + weights_{i-m})
        # where weights indices are taken modulo (2N-1) with appropriate bounds
        
        # This is a convolution-like operation. Let's use FFT:
        # Define an array W_full of length 2N-1 (weights)
        # We want: grad_i = sum_m h_m * (W_full[i+m] + W_full[i-m]) for valid indices
        
        # Actually, let's just compute it directly for clarity and correctness,
        # then optimize if needed. For N=600, this is O(N^2) = 360K ops per gradient,
        # which is fast enough for L-BFGS-B (typically ~100-200 iterations).
        
        # Direct computation:
        grad = np.zeros(N)
        for i in range(N):
            # Part 1: sum_{m=0}^{N-1} weights[i+m] * h[m] where i+m < 2N-1
            m1 = np.arange(N)
            idx1 = i + m1
            valid1 = idx1 < (2*N - 1)
            if np.any(valid1):
                grad[i] += np.sum(weights[idx1[valid1]] * h[m1[valid1]])
            
            # Part 2: sum_{m=0}^{N-1} weights[i-m] * h[m] where i-m >= 0
            m2 = np.arange(N)
            idx2 = i - m2
            valid2 = idx2 >= 0
            if np.any(valid2):
                grad[i] += np.sum(weights[idx2[valid2]] * h[m2[valid2]])
        
        # The objective is soft_max / (sum(h)^2), but since we project to sum=1,
        # the denominator is 1, so gradient is just grad
        # But we need to account for the projection h = x / sum(x)
        # For simplicity, since we use bounds and the projection is nearly identity
        # when x is already in simplex, we can use grad directly.
        
        return obj, grad
    
    def soft_max_objective(x, T):
        """Objective function for L-BFGS-B."""
        h = np.maximum(x, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h = h / s
        
        size = 2 * N
        H = fft(h, size)
        conv = np.real(ifft(H * H, size)[:2*N-1])
        
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -50, 50)
        log_sum_exp = conv_max / T + np.log(np.sum(np.exp(shifted)))
        return log_sum_exp * T
    
    # Frequency-domain flattening pre-conditioning
    def fd_flatten(h, gamma, iters):
        """Apply frequency-domain flattening with simplex projection."""
        current_h = h.copy()
        for _ in range(iters):
            if time.time() > deadline - 5:
                break
            
            size = 2 * N
            H = fft(current_h, size)
            P = np.abs(H) ** 2
            P_mean = np.mean(P)
            
            delta_P = P - P_mean
            H_new = H * np.exp(-gamma * delta_P)
            h_new = np.real(ifft(H_new, size)[:N])
            
            # Project to simplex: non-negative and sum=1
            h_new = np.maximum(h_new, 0)
            s = np.sum(h_new)
            if s > 1e-10:
                h_new = h_new / s
            else:
                h_new = current_h.copy()
                break
            
            current_h = h_new
        
        return current_h
    
    # Multiple restarts with different initializations
    np.random.seed(42)
    
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Cosine perturbations
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3]:
        for amp in [0.1, 0.2, 0.3]:
            h0 = 1.0 + amp * np.cos(2 * np.pi * freq * t)
            h0 = np.maximum(h0, 0.01)
            h0 = h0 / np.sum(h0)
            initializations.append(h0.copy())
    
    # Random perturbations
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        h0 = np.ones(N) + 0.1 * np.random.randn(N)
        h0 = np.maximum(h0, 0.01)
        h0 = h0 / np.sum(h0)
        initializations.append(h0.copy())
    
    # Edge/center enhanced
    t_lin = np.linspace(-1, 1, N)
    h_edge = 1.0 + 0.3 * np.abs(t_lin)
    h_edge = np.maximum(h_edge, 0.01)
    h_edge = h_edge / np.sum(h_edge)
    initializations.append(h_edge.copy())
    
    h_center = 1.0 + 0.3 * np.exp(-t_lin**2 * 5)
    h_center = np.maximum(h_center, 0.01)
    h_center = h_center / np.sum(h_center)
    initializations.append(h_center.copy())
    
    # Process each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Frequency-domain flattening (fast pre-conditioning)
        h_fd = fd_flatten(h0, gamma=0.05, iters=20)
        
        # Evaluate
        score_fd = compute_autoconv_max(h_fd)
        if score_fd < best_score:
            best_score = score_fd
            best_h = h_fd.copy()
        
        # Phase 2: L-BFGS-B optimization with soft-max objective
        if time.time() > deadline - 15:
            break
        
        # Stage 1: Coarse optimization (higher T)
        T1 = 0.02
        try:
            result1 = optimize.minimize(
                lambda x: soft_max_objective(x, T1),
                h_fd,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            result1 = None
        
        if result1 is not None and result1.fun < best_score * 1.05:
            h_new = np.maximum(result1.x, 0)
            s = np.sum(h_new)
            if s > 1e-10:
                h_new = h_new / s
            score_new = compute_autoconv_max(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
        
        # Stage 2: Fine optimization (lower T)
        if time.time() > deadline - 10:
            break
        
        T2 = 0.005
        try:
            h_start = best_h.copy()
            result2 = optimize.minimize(
                lambda x: soft_max_objective(x, T2),
                h_start,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            result2 = None
        
        if result2 is not None:
            h_new = np.maximum(result2.x, 0)
            s = np.sum(h_new)
            if s > 1e-10:
                h_new = h_new / s
            score_new = compute_autoconv_max(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
    
    # Final polish with very low temperature
    if time.time() < deadline - 5:
        T_final = 0.002
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            try:
                h_perturbed = best_h + 0.002 * np.random.randn(N)
                h_perturbed = np.maximum(h_perturbed, 0)
                s = np.sum(h_perturbed)
                if s > 1e-10:
                    h_perturbed = h_perturbed / s
                
                result = optimize.minimize(
                    lambda x: soft_max_objective(x, T_final),
                    h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_new = np.maximum(result.x, 0)
                s = np.sum(h_new)
                if s > 1e-10:
                    h_new = h_new / s
                score_new = compute_autoconv_max(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
                    
            except Exception:
                pass
    
    return list(best_h)
