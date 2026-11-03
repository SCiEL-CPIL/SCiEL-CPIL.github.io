"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq


def construct():
    """
    Construct step heights h on [0,1] with sum = N/2.
    Minimize the maximum overlap of h with shifted 1-h.
    Equivalent to maximizing the minimum periodic autocorrelation.
    
    Uses a fixed-point iteration with soft-min regularization and projection.
    """
    N_local = N
    target_sum = N_local / 2.0
    
    # Initialize with uniform distribution (known global optimum for max min-autocorrelation)
    h = np.full(N_local, 0.5)
    
    # Parameters for the iterative scheme
    lambda_reg = 50.0  # Sharpness of the soft-min (higher focuses more on worst lags)
    alpha = 0.1        # Step size for gradient ascent
    max_iter = 100
    tol = 1e-9
    
    for _ in range(max_iter):
        # Compute circular autocorrelation using FFT
        H = np.fft.fft(h)
        A = np.real(np.fft.ifft(H * np.conj(H)))
        
        # Compute weights for the soft-min approximation
        # w_s proportional to exp(-lambda * A_s)
        # Shift A to ensure numerical stability
        A_min = np.min(A)
        w = np.exp(-lambda_reg * (A - A_min))
        
        # Normalize weights to sum to 1
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        else:
            # Fallback to uniform weights if numerical issues
            w = np.full(N_local, 1.0 / N_local)
            
        # Compute gradient direction
        # d_k = sum_s w_s (h_{k-s} + h_{k+s}) = 2 * (w * h)_k
        W = np.fft.fft(w)
        d = 2.0 * np.real(np.fft.ifft(W * H))
        
        # Update h
        h_new = h + alpha * d
        
        # Project h_new to the feasible set: sum(h) = target_sum, 0 <= h_i <= 1
        # Find mu such that sum(clip(h_new - mu, 0, 1)) = target_sum
        def proj_f(mu):
            return np.sum(np.clip(h_new - mu, 0, 1)) - target_sum
        
        # Determine bounds for mu
        # f(mu) is decreasing.
        # f(min(h_new) - 1) = N - target_sum > 0
        # f(max(h_new) + 1) = -target_sum < 0
        lo = np.min(h_new) - 1.0
        hi = np.max(h_new) + 1.0
        
        try:
            mu = brentq(proj_f, lo, hi)
            h_proj = np.clip(h_new - mu, 0, 1)
        except (ValueError, RuntimeError):
            # Fallback: simple redistribution
            h_proj = h_new.copy()
            diff = target_sum - np.sum(h_proj)
            h_proj += diff / N_local
            h_proj = np.clip(h_proj, 0, 1)
            # Iterate to fix sum
            for _ in range(10):
                diff = target_sum - np.sum(h_proj)
                if abs(diff) < 1e-12:
                    break
                mask = (h_proj > 0) & (h_proj < 1)
                if np.sum(mask) > 0:
                    h_proj[mask] += diff / np.sum(mask)
                else:
                    break
                h_proj = np.clip(h_proj, 0, 1)
        
        # Check convergence
        if np.max(np.abs(h_proj - h)) < tol:
            h = h_proj
            break
        
        h = h_proj
        
    # Final adjustment to ensure exact sum
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        mask = (h > 0) & (h < 1)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
            # Re-adjust if clipping changed sum
            diff = target_sum - np.sum(h)
            if abs(diff) > 1e-12:
                # Distribute to nearest to 0.5
                idx = np.argsort(np.abs(h - 0.5))
                # Adjust the first few elements
                for i in idx:
                    if abs(diff) < 1e-12:
                        break
                    if diff > 0:
                        space = 1.0 - h[i]
                        add = min(space, diff)
                        h[i] += add
                        diff -= add
                    else:
                        space = h[i]
                        sub = min(space, -diff)
                        h[i] -= sub
                        diff += sub
                        
    return h
# EVOLVE-BLOCK-END
