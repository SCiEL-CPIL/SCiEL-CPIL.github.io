import numpy as np
from scipy.optimize import minimize
import time

def construct():
    N = 100
    deadline = time.time() + 145  # 145s deadline with 5s safety margin
    
    # Divisors of 100 for spectral basis
    divisors = [d for d in range(1, 101) if 100 % d == 0]
    # Exclude d=1 (DC component) since we handle mean separately
    freq_indices = [d for d in divisors if d > 1]
    # Also include N/2 = 50 for anti-periodic structure
    if 50 not in freq_indices:
        freq_indices.append(50)
    
    i = np.arange(N)
    
    # Build cosine basis matrix (excluding DC)
    # Each column is cos(2*pi*k*i/N) for k in freq_indices
    basis = np.array([np.cos(2 * np.pi * k * i / N) for k in freq_indices]).T
    n_basis = basis.shape[1]
    
    # Initial coefficients: small values around zero
    # The anti-periodic structure suggests emphasizing k=50 (Nyquist)
    c0 = np.zeros(n_basis)
    # Find index of k=50 and give it initial weight
    for idx, k in enumerate(freq_indices):
        if k == 50:
            c0[idx] = 0.4
        elif k == 1:
            c0[idx] = 0.1
        else:
            c0[idx] = 0.01
    
    def objective(c):
        """Compute max correlation value to minimize"""
        h = 0.5 + basis @ c
        # Project to [0,1]
        h = np.clip(h, 0, 1)
        # Compute correlation
        corr = np.correlate(h, 1 - h, mode="full")
        return np.max(corr) / N * 2
    
    def constraints_sum(c):
        """Enforce sum(h) = N/2"""
        h = 0.5 + basis @ c
        h = np.clip(h, 0, 1)
        return np.sum(h) - N / 2
    
    # Constraints: sum = N/2
    constraints = [{'type': 'eq', 'fun': constraints_sum}]
    
    # Bounds for coefficients (keep h in [0,1] approximately)
    bounds = [(-0.5, 0.5)] * n_basis
    
    best_c = c0.copy()
    best_val = objective(c0)
    
    # Multi-start optimization
    n_starts = 8
    rng = np.random.RandomState(42)
    
    for start in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        if start == 0:
            c_init = c0.copy()
        else:
            # Perturb initial coefficients
            c_init = c0 + 0.1 * rng.randn(n_basis)
            c_init = np.clip(c_init, -0.5, 0.5)
        
        try:
            result = minimize(
                objective,
                c_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or result.fun < best_val:
                c_candidate = result.x
                val = objective(c_candidate)
                if val < best_val:
                    best_val = val
                    best_c = c_candidate
        except Exception:
            continue
    
    # Construct final h
    h = 0.5 + basis @ best_c
    h = np.clip(h, 0, 1)
    
    # Ensure sum is exactly N/2 within tolerance
    current_sum = np.sum(h)
    if abs(current_sum - N/2) > 1e-9 * N:
        # Adjust by adding small constant and re-clipping
        diff = N/2 - current_sum
        h = h + diff / N
        h = np.clip(h, 0, 1)
        # Final projection to ensure exact sum
        current_sum = np.sum(h)
        if abs(current_sum - N/2) > 1e-9 * N:
            # Iterative adjustment
            for _ in range(10):
                diff = N/2 - np.sum(h)
                h = h + diff / N
                h = np.clip(h, 0, 1)
                if abs(np.sum(h) - N/2) <= 1e-9 * N:
                    break
    
    # Verify validity
    assert np.all(np.isfinite(h)), "Non-finite values in h"
    assert np.all(h >= 0) and np.all(h <= 1), "h out of [0,1] range"
    assert abs(np.sum(h) - N/2) <= 1e-9 * N, "Sum constraint violated"
    
    return h
