# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute projection indices for x, y, x-y, x+y, x+2y.
    2. Directly optimize the 100 log-probabilities (unconstrained real numbers) 
       using L-BFGS-B with multiple random restarts.
    3. This avoids restricting the solution to a parametric family (like quadratic),
       allowing for arbitrary distributions that might better maximize the ratio.
    """
    n = 10
    deadline = time.time() + 580.0  # Safety margin
    
    # Grid coordinates
    # P[y, x] -> y is row, x is col
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    # Note: The problem defines P[y, x]. 
    # d_val = x - y
    # m0_val = x
    # m1_val = x + y
    # m2_val = x + 2y
    
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)
    
    def compute_entropies_from_p(p_flat):
        """
        p_flat: shape (100,) normalized probabilities
        Returns: D, M
        """
        # Use bincount with weights for efficiency
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def objective(log_p_params):
        """
        log_p_params: shape (100,) log-probabilities (unnormalized)
        Returns: negative ratio -D/M (to be minimized)
        """
        if time.time() > deadline:
            return 1e9
            
        # Softmax to get probabilities
        # Subtract max for numerical stability
        log_p_flat = log_p_params - np.max(log_p_params)
        p_flat = np.exp(log_p_flat)
        sum_p = np.sum(p_flat)
        
        if sum_p < 1e-15:
            return 1e9
            
        p_flat /= sum_p
        
        H_d, M = compute_entropies_from_p(p_flat)
        
        if M < 1e-9:
            return 1e9
            
        return -H_d / M

    # Initial best: Uniform distribution
    # Uniform corresponds to all log_p = 0
    init_params = np.zeros(100)
    best_score = -objective(init_params)
    best_params = init_params.copy()
    
    # Multi-start optimization
    # We will run several independent L-BFGS-B optimizations from different starting points.
    # L-BFGS-B is fast and effective for smooth functions.
    
    rng = np.random.default_rng(42)
    
    # Define a helper to run one optimization
    def run_optimization(start_params, maxiter=2000):
        try:
            res = minimize(
                objective, 
                start_params, 
                method='L-BFGS-B', 
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res
        except Exception:
            return None

    # Start 1: Uniform
    res1 = run_optimization(init_params)
    if res1 is not None and res1.fun < -best_score:
        best_score = -res1.fun
        best_params = res1.x
        
    # If we have time, try random restarts
    # Each evaluation is very fast (microseconds), so we can afford many restarts.
    # L-BFGS-B converges quickly if the landscape is smooth.
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 50:
            break
            
        # Random initialization: small noise around uniform
        # Or potentially more structured noise to explore corners
        # Let's try a mix: some pure noise, some corner-biased noise
        
        if i % 2 == 0:
            # Gaussian noise
            start = rng.normal(0, 1.0, 100)
        else:
            # Corner-biased: high weight on (0,9) and (9,0) in log space
            # This corresponds to indices:
            # P[y, x] -> index = y * 10 + x
            # (0, 9) -> y=0, x=9 -> idx 9
            # (9, 0) -> y=9, x=0 -> idx 90
            start = rng.normal(0, 0.5, 100)
            start[9] += 3.0
            start[90] += 3.0
            # Maybe also (0,0) and (9,9) to explore diagonal?
            # start[0] += 1.0
            # start[99] += 1.0
            
        res = run_optimization(start, maxiter=1000)
        if res is not None:
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_params = res.x

    # Convert best parameters back to P
    log_p_flat = best_params - np.max(best_params)
    p_flat = np.exp(log_p_flat)
    p_flat /= np.sum(p_flat)
    
    P = p_flat.reshape((n, n))
    
    # Final validation
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
