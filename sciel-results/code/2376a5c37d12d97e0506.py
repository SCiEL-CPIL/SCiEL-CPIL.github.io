import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: uniform
    h0 = np.ones(N) / N
    
    # Objective: Minimize the peak of the autoconvolution.
    # We use a smooth surrogate: sum of squares of autoconvolution.
    # This encourages a flatter autoconvolution, reducing the max.
    def objective(h):
        # h is expected to sum to 1 and be non-negative due to constraints/bounds
        # Compute autoconvolution using FFT
        # n_fft should be at least 2*N-1
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        C = np.fft.ifft(H * H).real[:2*N-1]
        
        # Return sum of squares
        return np.sum(C**2)
        
    # We also want to check the actual max occasionally to guide the search?
    # No, L-BFGS-B works on the surrogate.
    # However, minimizing L2 norm might not minimize L-inf norm optimally.
    # Let's try to combine them or use a soft-max.
    # Soft-max is better for L-inf.
    
    def objective_softmax(h):
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        C = np.fft.ifft(H * H).real[:2*N-1]
        
        # Soft max parameter
        # C values are ~ 1/N = 1/600 ~ 0.0016
        # We want lambda * C to be O(1) or larger to capture max.
        # lambda = 1000 gives lambda * C ~ 1.6.
        # Let's use lambda = 100.
        lam = 100.0
        c_max = np.max(C)
        if c_max == 0:
            return 0.0
        # Log-sum-exp trick for stability
        # J = c_max + (1/lam) * log(sum(exp(lam * (C - c_max))))
        # This is numerically stable.
        shifted = lam * (C - c_max)
        # Clip to avoid underflow issues if range is large, though here it's small
        log_sum = np.log(np.sum(np.exp(shifted)))
        return c_max + log_sum / lam

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)}
    ]
    bounds = [(0, None) for _ in range(N)]
    
    best_h = h0.copy()
    best_score = np.inf
    
    # Evaluate initial
    init_val = objective_softmax(best_h)
    # Also compute the true max for comparison
    c_init = np.fft.ifft(np.fft.fft(best_h, n=2*N)**2).real[:2*N-1]
    true_max_init = np.max(c_init)
    best_score = true_max_init
    
    try:
        # Run optimization
        # We can try multiple runs or just one long one.
        # L-BFGS-B is robust.
        res = minimize(
            objective_softmax, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.x is not None:
            # Check time
            if time.time() < deadline:
                h_candidate = res.x
                # Normalize and clip
                h_candidate = np.maximum(h_candidate, 0)
                h_candidate = h_candidate / np.sum(h_candidate)
                
                # Evaluate true max
                c_cand = np.fft.ifft(np.fft.fft(h_candidate, n=2*N)**2).real[:2*N-1]
                true_max_cand = np.max(c_cand)
                
                if true_max_cand < best_score:
                    best_score = true_max_cand
                    best_h = h_candidate
                    # If we improved, we might try to polish further?
                    # For now, one run is enough.
                    
    except Exception:
        pass
        
    # Final check
    if time.time() > deadline:
        return [float(x) for x in best_h]
        
    # Ensure valid output
    best_h = np.maximum(best_h, 0)
    best_h = best_h / np.sum(best_h)
    
    return [float(x) for x in best_h]
