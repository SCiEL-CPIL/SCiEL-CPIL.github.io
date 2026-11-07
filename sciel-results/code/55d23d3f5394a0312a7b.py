"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s kill
    
    # Pre-compute sine/cosine basis for parametric model
    # h[i] = 0.5 + sum_k a_k sin(2*pi*k*i/N + phi_k)
    # Let's use K=6 harmonics.
    K = 6
    i = np.arange(N)
    
    # Pre-compute arguments
    # args[k] = 2*pi*k*i/N
    args = 2 * np.pi * np.arange(1, K+1)[:, None] * i[None, :] / N
    
    def project_to_feasible(h_raw):
        """
        Project h_raw to the feasible set:
        sum(h) = 50, 0 <= h <= 1.
        Uses binary search on lambda for h = clip(h_raw - lambda, 0, 1).
        """
        target_sum = 50.0
        # Bounds for lambda
        # If lambda is very large, h is 0. Sum=0.
        # If lambda is very small, h is 1. Sum=N=100.
        lo = -1.0
        hi = 2.0
        
        # Check if h_raw is already feasible (within tolerance)
        # This is an optimization, not strictly necessary but good.
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            h_proj = np.clip(h_raw - mid, 0.0, 1.0)
            s = np.sum(h_proj)
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        
        h_final = np.clip(h_raw - (lo + hi) / 2.0, 0.0, 1.0)
        
        # Final correction for sum
        diff = np.sum(h_final) - target_sum
        if abs(diff) > 1e-12:
            # Distribute residual
            # To avoid violating bounds, we can just subtract diff/N
            # Since diff is tiny, this is safe.
            h_final -= diff / N
            h_final = np.clip(h_final, 0.0, 1.0)
            
        return h_final

    def compute_value(h):
        """Compute the objective value."""
        corr = np.correlate(h, 1 - h, mode="full")
        return np.max(corr) / N * 2

    def param_to_h(params):
        """
        Convert parameters to heights.
        params: [a1..aK, phi1..phiK]
        """
        a = params[:K]
        phi = params[K:]
        
        h_raw = 0.5 * np.ones(N)
        # h_raw[i] = 0.5 + sum_k a_k sin(args[k, i] + phi_k)
        # args is (K, N), phi is (K,)
        # We need to add phi to each row of args.
        # sin(args + phi[:, None])
        sines = np.sin(args + phi[:, None])
        h_raw += sines.T @ a
        
        return project_to_feasible(h_raw)

    def objective(params):
        if time.time() > deadline:
            return 1e6
        h = param_to_h(params)
        return compute_value(h)

    # Initialization
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)
    max_restarts = 15
    num_params = 2 * K
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Random initial parameters
        # Amplitudes small to start
        a_init = rng.uniform(-0.2, 0.2, K)
        phi_init = rng.uniform(0, 2 * np.pi, K)
        x0 = np.concatenate([a_init, phi_init])
        
        try:
            # Use Powell method for robustness to non-smooth projection
            result = minimize(objective, x0, method='Powell', 
                             options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10})
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = param_to_h(result.x)
                
        except Exception:
            continue

    # Polish phase: SLSQP on full vector
    if time.time() < deadline - 10.0:
        try:
            def obj_h(h):
                return compute_value(h)
            
            cons = [
                {'type': 'eq', 'fun': lambda h: np.sum(h) - 50.0},
                {'type': 'ineq', 'fun': lambda h: h},
                {'type': 'ineq', 'fun': lambda h: 1.0 - h}
            ]
            bounds = [(0.0, 1.0)] * N
            
            # SLSQP might be slow, so limit iterations
            res = minimize(obj_h, best_h, method='SLSQP', bounds=bounds, constraints=cons, 
                          options={'maxiter': 50, 'ftol': 1e-12})
            
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_h = res.x
                
        except Exception:
            pass

    # Final verification
    h_final = np.clip(best_h, 0.0, 1.0)
    # Ensure sum is exactly 50 (within tolerance)
    diff = np.sum(h_final) - 50.0
    if abs(diff) > 1e-9 * N:
        # Fallback if projection failed badly
        # Try to fix sum
        h_final -= diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
        # If still bad, return seed
        if abs(np.sum(h_final) - 50.0) > 1e-9 * N:
            return np.full(N, 0.5)
            
    return h_final
# EVOLVE-BLOCK-END
