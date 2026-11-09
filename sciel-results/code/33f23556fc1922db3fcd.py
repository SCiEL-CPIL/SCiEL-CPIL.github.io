import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    m = 12
    n = 1
    
    # Initial guess: uniform spacing
    best_roots = np.linspace(1.0, 12.0, m)
    best_score = np.inf
    
    # Compute C4 bound for given roots
    def compute_c4(roots):
        """Compute the C4 upper bound for given double-root positions."""
        roots = np.asarray(roots, dtype=np.float64)
        if np.any(roots <= 0) or np.any(np.diff(roots) < 0):
            return np.inf
        
        # The polynomial is P(x) = prod (x - r_i)^2
        # For the Laguerre-based bound, we need to evaluate the relevant supremum
        # C4 relates to max of |P(x)| * exp(-x/2) / (x^(n/2) * normalization)
        # 
        # For the refined construction, the bound involves the Laguerre polynomial
        # with the given root structure. Let's compute the supremum numerically.
        
        # Evaluate on a grid
        x_max = roots[-1] + 20.0
        x = np.linspace(0.01, x_max, 20000)
        
        # Compute P(x) = prod (x - r_i)^2
        # Use log to avoid overflow
        log_P = np.zeros_like(x)
        for r in roots:
            log_P += 2.0 * np.log(np.abs(x - r) + 1e-30)
        
        # The bound: C4 ~ max |P(x)| * exp(-x/2) / x^(n/2)
        # Normalize appropriately
        log_bound = log_P - 0.5 * x - 0.5 * n * np.log(x + 1e-30)
        
        # Take max
        max_log_bound = np.max(log_bound)
        
        # The actual C4 value (we need the right normalization)
        # For the standard Laguerre bound, C4 = max |L_m(x)| e^{-x/2} x^{n/2}
        # With our constructed polynomial, we need the proper scaling
        c4 = np.exp(max_log_bound)
        
        return c4
    
    # Try to improve using scipy optimization
    try:
        from scipy.optimize import minimize, differential_evolution
        import warnings
        warnings.filterwarnings('ignore')
        
        def objective(roots_flat):
            roots = np.asarray(roots_flat, dtype=np.float64)
            if np.any(roots <= 0.01) or np.any(np.diff(roots) < 0.001):
                return 1e10
            return compute_c4(roots)
        
        # Differential evolution for global search
        bounds = [(0.01, 50.0)] * m
        
        # Check time
        if time.time() < deadline - 100:
            result_de = differential_evolution(
                objective, bounds, 
                maxiter=50, popsize=15, tol=1e-8,
                seed=42, polish=False,
                init='latinhypercube'
            )
            if result_de.success and result_de.fun < best_score:
                best_score = result_de.fun
                best_roots = result_de.x
        
        # Local refinement with L-BFGS-B
        if time.time() < deadline - 30:
            def objective_local(roots_flat):
                roots = np.asarray(roots_flat, dtype=np.float64)
                if np.any(roots <= 0.01) or np.any(np.diff(roots) < 0.001):
                    return 1e10
                return compute_c4(roots)
            
            # Multiple local starts
            starts = [
                best_roots.copy(),
                np.linspace(0.5, 15.0, m),
                np.linspace(1.0, 20.0, m),
                np.sort(np.random.uniform(0.5, 25.0, m)),
            ]
            
            for s in starts:
                if time.time() > deadline - 20:
                    break
                try:
                    res = minimize(
                        objective_local, s,
                        method='L-BFGS-B',
                        bounds=[(0.01, 60.0)] * m,
                        options={'maxiter': 200, 'ftol': 1e-10}
                    )
                    if res.fun < best_score:
                        best_score = res.fun
                        best_roots = res.x
                except Exception:
                    pass
                    
    except Exception:
        pass
    
    # Ensure roots are positive and sorted
    best_roots = np.sort(np.asarray(best_roots, dtype=np.float64))
    best_roots = np.maximum(best_roots, 0.01)
    
    # Convert to list
    return [float(r) for r in best_roots]
