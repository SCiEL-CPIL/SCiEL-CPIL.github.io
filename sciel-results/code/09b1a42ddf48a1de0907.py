"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time

N = 400
DEADLINE = 145.0  # safety margin before 150s kill

def _value(h):
    """Compute the task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _smooth_max_value(h, temperature=0.01):
    """Smooth approximation of the max using log-sum-exp."""
    conv = np.convolve(h, h)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    # log-sum-exp for stable computation
    m = np.max(conv)
    if m < 0:
        # All negative, max is the least negative
        return abs(2.0 * N * m / (s * s))
    # Scale for numerical stability
    scaled = conv / max(temperature, 1e-10)
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = temperature * log_sum_exp
    return abs(2.0 * N * smooth_max / (s * s))

def _gradient_smooth(h, temperature=0.01):
    """Gradient of smooth max value w.r.t. h."""
    conv = np.convolve(h, h)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.zeros_like(h), np.inf
    
    # For smooth max: d/dh_i [max_j conv_j] approximated
    # conv_k = sum_i h_i * h_{k-i} (with appropriate indexing)
    # Use soft-max weights
    scaled = conv / max(temperature, 1e-10)
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient of smooth_max w.r.t. h
    # smooth_max = temperature * (max_scaled + log(sum(exp(scaled - max_scaled))))
    # d(smooth_max)/d(conv_k) = temperature * weights[k]
    # d(conv_k)/d(h_i) = h_{k-i} (for the appropriate range)
    
    # conv has length 2N-1, index from 0 to 2N-2
    # conv[k] = sum_{i} h[i] * h[k-i] where 0<=i<N and 0<=k-i<N
    # So d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, else 0
    
    grad_conv = np.zeros_like(h)
    for k in range(len(conv)):
        w = weights[k] * temperature
        if w < 1e-15:
            continue
        # For this k, valid i range: max(0, k-(N-1)) <= i <= min(N-1, k)
        i_min = max(0, k - (N - 1))
        i_max = min(N - 1, k)
        if i_min > i_max:
            continue
        # d(conv[k])/d(h[i]) = h[k-i] for i in [i_min, i_max]
        idx = np.arange(i_min, i_max + 1)
        h_vals = h[k - idx]
        grad_conv[idx] += w * h_vals
    
    # Now value = abs(2*N*smooth_max / s^2)
    # Let M = smooth_max, S = s
    # f = 2*N*M / S^2 (assuming positive for gradient purposes)
    # df/dh_i = 2*N * (dM/dh_i * S^2 - M * 2*S * dS/dh_i) / S^4
    #         = 2*N * (grad_conv[i] * S - 2*M * 1) / S^3
    
    M = temperature * (max_scaled + np.log(np.sum(exp_scaled)))
    S = s
    grad_value = 2.0 * N * (grad_conv * S - 2.0 * M) / (S ** 3)
    
    return grad_value, abs(2.0 * N * M / (S * S))

def _project_sum_one(h):
    """Project h to have sum = 1 (simplest projection)."""
    return h - np.sum(h) / N + 1.0 / N

def _coordinate_descent(h, max_iters=500, tol=1e-10):
    """Coordinate descent to minimize value."""
    best_h = h.copy()
    best_val = _value(best_h)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for delta in [-1.0, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1.0]:
                trial = best_h.copy()
                trial[i] += delta
                # Renormalize to sum=1
                trial = _project_sum_one(trial)
                val = _value(trial)
                if val < best_val - tol:
                    best_h = trial
                    best_val = val
                    improved = True
                    break
        if not improved:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # Should be 2.0
    
    # Try several random initializations and gradient-based optimization
    rng = np.random.default_rng(42)
    
    num_restarts = 8
    max_grad_iters = 3000
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Random initialization
        h = rng.standard_normal(N)
        h = _project_sum_one(h)
        
        # Gradient descent with adaptive step size
        temperature = 0.05
        step_size = 0.01
        patience = 500
        no_improve_count = 0
        current_val = _value(h)
        
        for it in range(max_grad_iters):
            if time.time() - start_time > DEADLINE - 25:
                break
            
            # Adjust temperature (annealing)
            temperature = max(0.001, 0.05 * (0.995 ** it))
            
            grad, val = _gradient_smooth(h, temperature)
            
            if not np.isfinite(val):
                break
            
            if val < current_val:
                current_val = val
                no_improve_count = 0
                step_size = min(0.1, step_size * 1.05)
            else:
                no_improve_count += 1
                step_size *= 0.9
                if no_improve_count > patience:
                    break
            
            # Gradient step with projection
            h_new = h - step_size * grad
            h_new = _project_sum_one(h_new)
            
            # Check if it improved
            val_new = _value(h_new)
            if np.isfinite(val_new) and val_new < current_val * 1.01:
                h = h_new
                current_val = val_new
            
            # Periodic coordinate descent polish
            if it % 500 == 499 and time.time() - start_time < DEADLINE - 20:
                h, val_cd = _coordinate_descent(h, max_iters=100)
                if val_cd < current_val:
                    current_val = val_cd
            
            # Update best
            if current_val < best_val:
                best_h = h.copy()
                best_val = current_val
        
        # Final coordinate descent on best from this restart
        if time.time() - start_time < DEADLINE - 15:
            h_final, val_final = _coordinate_descent(h, max_iters=200)
            if val_final < best_val:
                best_h = h_final.copy()
                best_val = val_final
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        h_polished, val_polished = _coordinate_descent(best_h, max_iters=500, tol=1e-12)
        if val_polished < best_val:
            best_h = h_polished
            best_val = val_polished
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = 2.0
    
    s = np.sum(best_h)
    if abs(s) < 1e-15:
        best_h = np.ones(N)
    
    return best_h
