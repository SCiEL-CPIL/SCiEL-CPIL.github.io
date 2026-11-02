"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _chebyshev_to_heights(coeffs, N):
    """Convert Chebyshev coefficients to step heights on [0,2]."""
    # Step centers at x_i = 2*(i+0.5)/N for i=0..N-1
    # Map to [-1,1]: t_i = 2*x_i/2 - 1 = x_i - 1 = 2*(i+0.5)/N - 1
    t = np.linspace(-1 + 1.0/N, 1 - 1.0/N, N)
    # Evaluate Chebyshev polynomial
    h = np.zeros(N)
    for k, c in enumerate(coeffs):
        h += c * np.cos(k * np.arccos(t))
    return h

def _heights_to_chebyshev(h, N, max_k):
    """Convert step heights to Chebyshev coefficients (least squares)."""
    t = np.linspace(-1 + 1.0/N, 1 - 1.0/N, N)
    theta = np.arccos(t)
    # Build Chebyshev matrix
    V = np.zeros((N, max_k + 1))
    for k in range(max_k + 1):
        V[:, k] = np.cos(k * theta)
    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(V, h, rcond=None)
    return coeffs

def _max_overlap(h):
    """Compute maximum overlap of h with shifted 1-h."""
    N = len(h)
    max_ov = 0.0
    for s in range(N):
        shift = s * 2.0 / N
        if shift <= 1.0:
            # overlap = integral_0^{1-s} h + integral_{1+s}^2 (1-h)
            # Discretized: sum h[i] for steps in [0, 1-s] + (measure of [1+s,2] - sum h[i] for steps in [1+s,2])
            # Approximate using step centers
            centers = np.linspace(1.0/N, 2.0 - 1.0/N, N)
            mask1 = centers < (1.0 - shift)
            mask2 = centers >= (1.0 + shift)
            ov = np.sum(h[mask1]) + (np.sum(mask2) * 2.0/N - np.sum(h[mask2]))
        else:
            # overlap = integral_0^{s-1} (1-h) + integral_{2-s}^2 h
            centers = np.linspace(1.0/N, 2.0 - 1.0/N, N)
            mask1 = centers < (shift - 1.0)
            mask2 = centers >= (2.0 - shift)
            ov = (np.sum(mask1) * 2.0/N - np.sum(h[mask1])) + np.sum(h[mask2])
        max_ov = max(max_ov, ov)
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Try different N values and Chebyshev orders
    for N in [50, 100, 200]:
        if time.time() > deadline:
            break
        
        for max_k in [3, 5, 7, 9, 11]:
            if time.time() > deadline:
                break
            
            # Initialize with constant + small perturbation
            h_init = np.full(N, 0.5)
            # Add small random perturbation to break symmetry
            np.random.seed(N * 100 + max_k)
            h_init += np.random.randn(N) * 0.01
            h_init = np.clip(h_init, 0, 1)
            # Adjust sum to N/2
            diff = N/2.0 - np.sum(h_init)
            h_init += diff / N
            h_init = np.clip(h_init, 0, 1)
            
            coeffs_init = _heights_to_chebyshev(h_init, N, max_k)
            
            def objective(coeffs):
                h = _chebyshev_to_heights(coeffs, N)
                h = np.clip(h, 0, 1)
                # Penalize sum deviation
                sum_dev = np.sum(h) - N/2.0
                # Compute max overlap
                ov = _max_overlap(h)
                # Quadratic penalty for sum deviation
                return ov + 10.0 * sum_dev**2
            
            # Use SLSQP for constrained optimization
            try:
                result = minimize(objective, coeffs_init, method='SLSQP', 
                                  options={'maxiter': 100, 'ftol': 1e-6})
                if result.success or result.fun < best_score:
                    h_opt = _chebyshev_to_heights(result.x, N)
                    h_opt = np.clip(h_opt, 0, 1)
                    # Ensure exact sum
                    diff = N/2.0 - np.sum(h_opt)
                    h_opt += diff / N
                    h_opt = np.clip(h_opt, 0, 1)
                    
                    score = _max_overlap(h_opt)
                    if score < best_score:
                        best_score = score
                        best_h = h_opt
            except Exception:
                pass
    
    if best_h is not None:
        return best_h
    else:
        return np.full(100, 0.5)

# EVOLVE-BLOCK-END
