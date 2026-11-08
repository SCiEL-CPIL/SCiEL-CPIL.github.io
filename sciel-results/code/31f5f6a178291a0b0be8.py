"""Combined approach: FFT-based overlap + multi-start L-BFGS-B optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _compute_max_overlap(h):
    """Compute max over all circular shifts k of sum_j h[j]*(1-h[(j+k)%N]) * (2/N)."""
    n = len(h)
    dx = 2.0 / n
    # overlap(k) = sum_j h[j] * (1 - h[(j+k)%n]) * dx
    #            = (sum_j h[j] - sum_j h[j]*h[(j+k)%n]) * dx
    # sum_j h[j] = n/2 (since integral = 1, dx = 2/n, sum h * dx = 1 => sum h = n/2)
    # So overlap(k) = (n/2 - autocorr[k]) * dx
    # where autocorr[k] = sum_j h[j] * h[(j+k)%n]
    h_ext = np.concatenate([h, h])
    # autocorr[k] = sum_{j=0}^{n-1} h[j] * h[(j+k)%n]
    # Use FFT: autocorr = ifft(fft(h)^2) for circular autocorrelation
    # But we need cross-correlation style: sum_j h[j] * h[j+k]
    # This is the circular cross-correlation of h with h
    H = np.fft.rfft(h, n=2*n)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    # autocorr[k] = sum_j h[j] * h[(j-k)%n] for k=0..n-1
    # We want sum_j h[j] * h[(j+k)%n] which is autocorr[-k % n] = autocorr[(n-k)%n]
    # Actually for circular, autocorr[k] as computed = sum_j h[j]*h[j-k]
    # sum_j h[j]*h[j+k] = autocorr[-k mod n] = autocorr[(n-k)%n]
    # But since autocorr is symmetric for real signals in circular sense, let's just use it
    # overlap(k) = (n/2 - autocorr_shifted[k]) * dx
    # Let's verify: for k=0, autocorr[0] = sum h[j]^2
    # overlap(0) = sum h[j]*(1-h[j]) * dx = (sum h[j] - sum h[j]^2)*dx = (n/2 - sum h[j]^2)*dx
    # That's correct.
    # For general k, we need sum_j h[j]*h[(j+k)%n]
    # autocorr from ifft(|FFT|^2) gives sum_j h[j]*h[(j-k)%n]
    # So sum_j h[j]*h[(j+k)%n] = autocorr[(-k)%n] = autocorr[(n-k)%n] for k>0, autocorr[0] for k=0
    sum_h = np.sum(h)
    # autocorr[k] computed = sum_j h[j] * h[(j-k)%n]
    # We want C[k] = sum_j h[j] * h[(j+k)%n] = autocorr[(-k)%n]
    # For k=0: C[0] = autocorr[0]
    # For k=1: C[1] = autocorr[n-1]
    # etc.
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    overlaps = (sum_h - C) * dx
    return np.max(overlaps)

def _objective(coeffs):
    """Objective function for optimization: max overlap from Fourier coefficients."""
    x = np.linspace(0, 2, N, endpoint=False)
    h = np.full(N, 0.5)
    for k in range(len(coeffs) // 2):
        a_k = coeffs[2*k]
        b_k = coeffs[2*k+1]
        h += a_k * np.cos(np.pi * k * x) + b_k * np.sin(np.pi * k * x)
    # Check bounds
    if np.any(h < -1e-8) or np.any(h > 1.0 + 1e-8):
        return 10.0 + (np.max(np.abs(np.clip(h, None, None) - np.clip(h, 0, 1))))
    h = np.clip(h, 0.0, 1.0)
    # Check integral: sum h * (2/N) should be 1, i.e., sum h = N/2
    current_sum = np.sum(h)
    if abs(current_sum - N/2) > 1e-6:
        return 10.0 + abs(current_sum - N/2)
    return _compute_max_overlap(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Start with baseline
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Try some analytic candidates first
    candidates = []
    
    # h = 0.5 + 0.5*sin(pi*x)
    h1 = 0.5 + 0.5 * np.sin(np.pi * x)
    h1 = np.clip(h1, 0, 1)
    candidates.append(h1)
    
    # h = 0.5 + 0.5*cos(pi*x)  
    h2 = 0.5 + 0.5 * np.cos(np.pi * x)
    h2 = np.clip(h2, 0, 1)
    candidates.append(h2)
    
    # Square wave: 1 on [0,1), 0 on [1,2)
    h3 = (x < 1.0).astype(float)
    candidates.append(h3)
    
    # h = 0.5 + 0.3*sin(2*pi*x)
    h4 = 0.5 + 0.3 * np.sin(2 * np.pi * x)
    h4 = np.clip(h4, 0, 1)
    candidates.append(h4)
    
    for h_cand in candidates:
        # Ensure valid
        h_cand = np.clip(h_cand, 0, 1)
        s = _compute_max_overlap(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Multi-start local optimization with Fourier coefficients
    n_modes = 4  # Use 4 modes for more flexibility
    n_coeffs = 2 * n_modes
    bounds = [(-0.5, 0.5)] * n_coeffs
    
    np.random.seed(42)
    
    n_starts = 15
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initial point
        if start_idx == 0:
            x0 = np.zeros(n_coeffs)
        else:
            x0 = np.random.uniform(-0.3, 0.3, n_coeffs)
        
        try:
            result = minimize(
                _objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                # Reconstruct h from coefficients
                h_new = np.full(N, 0.5)
                for k in range(n_modes):
                    h_new += result.x[2*k] * np.cos(np.pi * k * x)
                    h_new += result.x[2*k+1] * np.sin(np.pi * k * x)
                h_new = np.clip(h_new, 0, 1)
                
                # Fix integral
                current_sum = np.sum(h_new)
                target_sum = N / 2
                if abs(current_sum - target_sum) > 1e-8:
                    # Adjust values
                    diff = target_sum - current_sum
                    if diff > 0:
                        mask = h_new < 1.0 - 1e-10
                        if np.sum(mask) > 0:
                            h_new[mask] += diff / np.sum(mask)
                    else:
                        mask = h_new > 1e-10
                        if np.sum(mask) > 0:
                            h_new[mask] += diff / np.sum(mask)
                    h_new = np.clip(h_new, 0, 1)
                
                # Verify
                if abs(np.sum(h_new) - N/2) < 1e-6:
                    actual_score = _compute_max_overlap(h_new)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_new.copy()
        except Exception:
            pass
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
