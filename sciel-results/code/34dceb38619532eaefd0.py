import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_genlaguerre
import time

def construct():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0
    
    def time_left():
        return deadline - safety_margin - (time.time() - start_time)
    
    # Get initial roots from Laguerre polynomial L_12^{(2)}
    try:
        roots_lag = roots_genlaguerre(12, 2.0)
        lag_roots = np.sort(roots_lag.real)
    except Exception:
        # Fallback: approximate roots
        lag_roots = np.array([0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 13.0, 16.5, 20.0, 24.0, 28.0, 32.0])
    
    # Strategy: 6 variables s_i, roots are s_i^2 (each used twice)
    # Initialize with square roots of every other Laguerre root
    s0 = np.sqrt(lag_roots[::2])  # 6 values
    
    def compute_c4(s):
        """Compute C4 upper bound."""
        s = np.abs(s)
        # 12 double roots
        roots = np.array([s[i]**2 for i in range(6)] * 2)
        roots = np.sort(roots)
        
        # Use Gauss-Laguerre quadrature with enough points
        n_quad = 200
        x_nodes, x_weights = np.polynomial.laguerre.laggauss(n_quad)
        
        # Compute Q(x) = prod (x - r_i)^2
        # Do this carefully to avoid overflow
        log_Q = np.zeros_like(x_nodes)
        for r in roots:
            diff = x_nodes - r
            # Handle sign
            log_Q += 2 * np.log(np.abs(diff) + 1e-30)
        
        # This can have issues with signs, let's compute directly but in chunks
        Q = np.ones_like(x_nodes)
        for r in roots:
            Q *= (x_nodes - r)**2
        
        Q2 = Q**2
        
        I0 = np.sum(x_weights * Q2)
        I1 = np.sum(x_weights * x_nodes * Q2)
        I2 = np.sum(x_weights * x_nodes**2 * Q2)
        
        if I0 <= 1e-30 or I2 <= 1e-30:
            return 1e30
        
        # Variance-type bound
        c4 = (I1**2) / (I0 * I2)
        
        return c4
    
    def objective(s):
        if time_left() < 5.0:
            return 1e30
        return compute_c4(s)
    
    best_val = compute_c4(s0)
    best_s = s0.copy()
    
    # Local refinement from initial guess
    try:
        res = minimize(objective, best_s, method='L-BFGS-B', 
                      bounds=[(0.1, 10.0)]*6,
                      options={'maxiter': 2000, 'ftol': 1e-15})
        if res.fun < best_val:
            best_val = res.fun
            best_s = res.x
    except Exception:
        pass
    
    # Try differential evolution if time permits
    if time_left() > 60.0:
        try:
            de_res = differential_evolution(
                objective, [(0.1, 10.0)]*6, seed=42,
                maxiter=200, tol=1e-10, polish=True, workers=1
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_s = de_res.x
        except Exception:
            pass
    
    # Final local refinement
    if time_left() > 10.0:
        try:
            res = minimize(objective, best_s, method='L-BFGS-B',
                          bounds=[(0.1, 10.0)]*6,
                          options={'maxiter': 5000, 'ftol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_s = res.x
        except Exception:
            pass
    
    # Convert to 12 double roots
    s_final = np.abs(best_s)
    roots_12 = np.array([s_final[i]**2 for i in range(6)] * 2)
    roots_12 = np.sort(roots_12)
    roots_12 = np.maximum(roots_12, 1e-10)
    
    return roots_12.tolist()
