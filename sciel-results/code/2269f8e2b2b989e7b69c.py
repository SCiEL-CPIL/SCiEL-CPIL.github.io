"""
Combined program: Chebyshev parameterization + Differential Evolution global search 
+ L-BFGS-B local polish with log-sum-exp smooth max surrogate.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_chebyt
from scipy.special import logsumexp

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Precompute Chebyshev basis
    # Map indices 0..N-1 to [-1, 1]
    x = np.linspace(-1, 1, N)
    K = 20  # Number of Chebyshev modes (0 to K)
    basis = np.zeros((N, K + 1))
    for k in range(K + 1):
        basis[:, k] = eval_chebyt(k, x)
    
    # Precompute sum of each basis function to help with normalization
    basis_sums = np.sum(basis, axis=0)
    
    def compute_value_exact(h):
        """Compute the exact evaluation value."""
        s = np.sum(h)
        if abs(s) < 1e-15:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        conv = np.convolve(h, h)
        return abs(2 * N * np.max(conv) / (s ** 2))

    def objective_coeffs(c, beta=50.0):
        """
        Objective for optimization over Chebyshev coefficients.
        Normalizes h to sum to 1 to remove scale invariance.
        Uses log-sum-exp for smooth max.
        """
        if not np.all(np.isfinite(c)):
            return 1e10
        
        h = c @ basis.T
        s = np.sum(h)
        
        # If sum is too small, the normalization is unstable.
        # Penalize heavily.
        if abs(s) < 1e-6:
            return 1e10 + abs(s)
            
        # Normalize h so that sum(h) = 1
        # This is valid because the objective is scale-invariant:
        # val(h) = 2N * max(conv(h,h)) / sum(h)^2
        # val(h/s) = 2N * max(conv(h/s, h/s)) / sum(h/s)^2
        #          = 2N * (1/s^2) max(conv(h,h)) / (1/s^2)
        #          = val(h)
        h_norm = h / s
        
        conv = np.convolve(h_norm, h_norm)
        
        # Smooth max approximation
        # logsumexp(conv / beta) * beta approximates max(conv)
        # Note: logsumexp expects the log of values. 
        # Let y = conv. max(y) = beta * logsumexp(y/beta).
        smax = beta * logsumexp(conv / beta)
        
        # The objective to minimize is 2 * N * smax (since sum(h_norm)^2 = 1)
        return 2.0 * N * smax

    def objective_exact_coeffs(c):
        """Exact objective for final verification or DE (DE can handle non-smooth)."""
        h = c @ basis.T
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        h_norm = h / s
        conv = np.convolve(h_norm, h_norm)
        return 2.0 * N * np.max(conv)

    best_h = np.ones(N)
    best_val = compute_value_exact(best_h) # 2.0
    
    # 2. Global Search: Differential Evolution
    # Bounds for coefficients. 
    # T_0 is 1. If c_0 is large, sum is large.
    # We want to explore reasonable ranges.
    # Since we normalize, the scale of c doesn't matter, only the direction.
    # However, DE works better with bounded variables.
    # Let's set bounds to [-10, 10] for all coefficients.
    bounds = [(-10.0, 10.0)] * (K + 1)
    
    # Use a seed for reproducibility
    np.random.seed(42)
    
    de_result = None
    try:
        de_result = differential_evolution(
            objective_exact_coeffs, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-4,
            polish=False # We will do our own polish
        )
    except Exception:
        pass
    
    if de_result and de_result.fun < best_val:
        # de_result.fun is the value of objective_exact_coeffs, which is 2*N*max(conv) for normalized h
        # The actual task value is the same, because we normalized to sum=1.
        # So de_result.fun is comparable to best_val?
        # Wait. best_val was computed on unnormalized h. 
        # compute_value_exact(h) = 2N * max(conv(h,h)) / sum(h)^2.
        # If h is normalized to sum=1, value = 2N * max(conv(h,h)).
        # So yes, they are comparable.
        
        best_c = de_result.x
        best_h = best_c @ basis.T
        best_val = de_result.fun
        
    # 3. Local Polish: L-BFGS-B with smooth surrogate
    # We need to polish the best solution found so far.
    # Start from best_c if found, else random.
    if de_result is not None and de_result.fun < 2.0:
        c_start = de_result.x
    else:
        # Fallback: random initialization
        c_start = np.random.randn(K + 1) * 0.5
        c_start[0] = 1.0 # Ensure some DC component
    
    # Run L-BFGS-B polish
    # Use a moderate beta for the smooth max to allow gradients to flow
    beta_polish = 30.0
    
    try:
        res_polish = minimize(
            objective_coeffs, 
            c_start, 
            args=(beta_polish,),
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res_polish.fun < best_val:
            # Verify with exact value
            h_candidate = res_polish.x @ basis.T
            val_candidate = compute_value_exact(h_candidate)
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                best_c = res_polish.x
    except Exception:
        pass

    # 4. Second Polish with higher beta (sharper max approximation)
    if time.time() < deadline - 20:
        if best_c is not None:
            try:
                beta_high = 100.0
                res_polish2 = minimize(
                    objective_coeffs, 
                    best_c, 
                    args=(beta_high,),
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if res_polish2.fun < best_val:
                    h_candidate = res_polish2.x @ basis.T
                    val_candidate = compute_value_exact(h_candidate)
                    if val_candidate < best_val:
                        best_h = h_candidate
                        best_val = val_candidate
                        best_c = res_polish2.x
            except Exception:
                pass

    # 5. Multi-start local polish from random Chebyshev inits
    # If time permits, try a few more random starts with L-BFGS-B
    if time.time() < deadline - 10:
        for i in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random init
            c_rand = np.random.randn(K + 1) * 0.2
            c_rand[0] = np.random.uniform(0.5, 1.5)
            
            try:
                res_rand = minimize(
                    objective_coeffs, 
                    c_rand, 
                    args=(50.0,),
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-9}
                )
                
                if res_rand.fun < best_val:
                    h_candidate = res_rand.x @ basis.T
                    val_candidate = compute_value_exact(h_candidate)
                    if val_candidate < best_val:
                        best_h = h_candidate
                        best_val = val_candidate
            except Exception:
                continue

    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
        
    return best_h

# Entry point
if __name__ == "__main__":
    construct()
