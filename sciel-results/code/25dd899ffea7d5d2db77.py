import numpy as np
import time
import random

def construct():
    """
    Returns the 24 coefficients of the polynomial P(z) = sum c_k z^k 
    that minimizes the sup norm on the unit circle.
    """
    # Precompute theta grid
    N = 20000
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Precompute z^k for k = 0, 1, ..., 23
    # Z[k, n] = z[n]^k
    Z = np.zeros((24, N), dtype=np.complex128)
    Z[0, :] = 1.0
    for k in range(1, 24):
        Z[k, :] = Z[k-1, :] * z
    
    def eval_fast(coeffs):
        """Fast evaluation using precomputed Z matrix."""
        c = np.array(coeffs, dtype=np.float64)
        P = Z.T @ c  # Shape: (N,)
        return np.max(np.abs(P)) / np.sqrt(24)
    
    def eval_refine(coeffs):
        """Evaluation with golden-section refinement."""
        c = np.array(coeffs, dtype=np.float64)
        P = Z.T @ c
        mags = np.abs(P)
        max_mag = np.max(mags)
        
        idx = np.argmax(mags)
        step = 2 * np.pi / N
        t_low = thetas[(idx - 1) % N]
        t_high = thetas[(idx + 1) % N]
        
        def neg_sq_mag(t):
            zt = np.exp(1j * t)
            pt = 0.0 + 0.0j
            ztk = 1.0 + 0.0j
            for ck in coeffs:
                pt += ck * ztk
                ztk *= zt
            return -np.abs(pt)**2
            
        gr = (np.sqrt(5) - 1) / 2
        a, b = t_low, t_high
        c_ = b - gr * (b - a)
        d_ = a + gr * (b - a)
        fc = neg_sq_mag(c_)
        fd = neg_sq_mag(d_)
        
        for _ in range(50):
            if fc > fd:
                b = d_
                d_ = c_
                fd = fc
                c_ = b - gr * (b - a)
                fc = neg_sq_mag(c_)
            else:
                a = c_
                c_ = d_
                fc = fd
                d_ = a + gr * (b - a)
                fd = neg_sq_mag(d_)
                
        t_opt = (a + b) / 2
        z_opt = np.exp(1j * t_opt)
        p_opt = 0.0 + 0.0j
        z_opt_k = 1.0 + 0.0j
        for ck in coeffs:
            p_opt += ck * z_opt_k
            z_opt_k *= z_opt
            
        return np.abs(p_opt) / np.sqrt(24)
    
    # Known good starting points
    best_coeffs = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1]
    best_val = eval_fast(best_coeffs)
    
    OPT_VAL = 1.2583758
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Local search
    def local_search(coeffs, time_limit=5.0):
        current_coeffs = list(coeffs)
        current_val = eval_fast(current_coeffs)
        improved = True
        start_time_ls = time.time()
        
        while improved and (time.time() - start_time_ls) < time_limit:
            improved = False
            indices = list(range(24))
            random.shuffle(indices)
            for k in indices:
                new_coeffs = current_coeffs[:]
                new_coeffs[k] *= -1
                new_val = eval_fast(new_coeffs)
                if new_val < current_val - 1e-12:
                    current_coeffs = new_coeffs
                    current_val = new_val
                    improved = True
                    
        return current_coeffs, current_val
    
    # First, try to improve the starting point
    best_coeffs, best_val = local_search(best_coeffs, time_limit=10.0)
    
    # If we are already at the optimum (within tolerance), return
    if best_val <= OPT_VAL + 1e-6:
        final_val = eval_refine(best_coeffs)
        return best_coeffs
    
    # Random restarts with local search
    while time.time() < deadline:
        # Generate random coefficients
        rand_coeffs = [random.choice([1, -1]) for _ in range(24)]
        
        # Local search
        c, v = local_search(rand_coeffs, time_limit=3.0)
        
        if v < best_val - 1e-12:
            best_coeffs = c
            best_val = v
            
            # Check if we hit the global optimum
            if best_val <= OPT_VAL + 1e-6:
                break
                
    return best_coeffs
