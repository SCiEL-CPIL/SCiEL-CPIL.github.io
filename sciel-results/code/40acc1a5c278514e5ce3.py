import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    m = 12
    n = 1
    
    # Initial guess: use known good starting points
    best_roots = np.linspace(0.5, 12.0, m)
    best_score = np.inf
    
    # Precompute Gauss-Laguerre nodes and weights for accurate integration
    # For the bound, we need to evaluate max over x of |P(x)| * x^(-n/2) * exp(-x/2)
    # where P(x) = prod (x - r_i)^2
    # 
    # Better approach: use the fact that for Laguerre polynomials, the relevant
    # quantity can be computed via quadrature or by finding the maximum directly.
    
    def compute_c4_fast(roots):
        """Compute the C4 upper bound efficiently using log-space and careful max-finding."""
        roots = np.asarray(roots, dtype=np.float64)
        if len(roots) != m:
            return np.inf
        if np.any(roots <= 0) or np.any(np.diff(roots) < 0):
            return np.inf
        
        # The bound involves maximizing f(x) = |P(x)| * exp(-x/2) / x^(n/2)
        # where P(x) = prod (x - r_i)^2
        # 
        # In log space: log f(x) = 2*sum(log|x-r_i|) - x/2 - (n/2)*log(x)
        # 
        # Find the maximum by evaluating on a fine grid and then refining
        
        # Coarse grid to find approximate max location
        x_max = max(roots[-1] + 10.0, 30.0)
        x_coarse = np.linspace(0.01, x_max, 5000)
        
        log_f = np.zeros_like(x_coarse)
        for r in roots:
            log_f += 2.0 * np.log(np.abs(x_coarse - r) + 1e-30)
        log_f -= 0.5 * x_coarse
        log_f -= 0.5 * n * np.log(x_coarse + 1e-30)
        
        # Find approximate max
        idx = np.argmax(log_f)
        
        # Refine around the max with a finer grid
        if idx > 0 and idx < len(x_coarse) - 1:
            x_left = max(x_coarse[idx-1], 0.001)
            x_right = min(x_coarse[idx+1], x_max)
        else:
            x_left = max(0.001, x_coarse[idx] * 0.5)
            x_right = min(x_coarse[idx] * 2.0, x_max)
        
        x_fine = np.linspace(x_left, x_right, 2000)
        log_f_fine = np.zeros_like(x_fine)
        for r in roots:
            log_f_fine += 2.0 * np.log(np.abs(x_fine - r) + 1e-30)
        log_f_fine -= 0.5 * x_fine
        log_f_fine -= 0.5 * n * np.log(x_fine + 1e-30)
        
        max_log_f = np.max(log_f_fine)
        
        # Also check endpoints and other potential maxima
        # The function can have multiple local maxima due to the |x-r_i| terms
        # Check intervals between roots
        for i in range(len(roots) - 1):
            if time.time() > deadline - 30:
                break
            a = max(roots[i] + 0.001, 0.001)
            b = max(roots[i+1] - 0.001, a + 0.001)
            if b <= a:
                continue
            x_check = np.linspace(a, b, 500)
            log_f_check = np.zeros_like(x_check)
            for r in roots:
                log_f_check += 2.0 * np.log(np.abs(x_check - r) + 1e-30)
            log_f_check -= 0.5 * x_check
            log_f_check -= 0.5 * n * np.log(x_check + 1e-30)
            max_log_f = max(max_log_f, np.max(log_f_check))
        
        # Check before first root and after last root
        if time.time() < deadline - 30:
            # Before first root
            x_before = np.linspace(0.001, max(roots[0] - 0.001, 0.002), 500)
            if len(x_before) > 1:
                log_f_before = np.zeros_like(x_before)
                for r in roots:
                    log_f_before += 2.0 * np.log(np.abs(x_before - r) + 1e-30)
                log_f_before -= 0.5 * x_before
                log_f_before -= 0.5 * n * np.log(x_before + 1e-30)
                max_log_f = max(max_log_f, np.max(log_f_before))
            
            # After last root
            x_after = np.linspace(roots[-1] + 0.001, x_max, 500)
            if len(x_after) > 1:
                log_f_after = np.zeros_like(x_after)
                for r in roots:
                    log_f_after += 2.0 * np.log(np.abs(x_after - r) + 1e-30)
                log_f_after -= 0.5 * x_after
                log_f_after -= 0.5 * n * np.log(x_after + 1e-30)
                max_log_f = max(max_log_f, np.max(log_f_after))
        
        c4 = np.exp(max_log_f)
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
            return compute_c4_fast(roots)
        
        # Use SLSQP with ordering constraints as suggested
        def objective_slsqp(roots_flat):
            roots = np.asarray(roots_flat, dtype=np.float64)
            if np.any(roots <= 0.01) or np.any(np.diff(roots) < 0.001):
                return 1e10
            return compute_c4_fast(roots)
        
        # Differential evolution for global search
        bounds = [(0.01, 50.0)] * m
        
        # Check time
        if time.time() < deadline - 150:
            result_de = differential_evolution(
                objective, bounds, 
                maxiter=30, popsize=12, tol=1e-8,
                seed=42, polish=False,
                init='latinhypercube'
            )
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_roots = result_de.x
        
        # Local refinement with SLSQP using inequality constraints for ordering
        if time.time() < deadline - 60:
            # Define constraints for SLSQP: r_i > 0 and r_{i+1} > r_i
            constraints = []
            for i in range(m - 1):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i+1] - x[i] - 0.001
                })
            
            # Multiple local starts
            starts = [
                best_roots.copy(),
                np.linspace(0.5, 15.0, m),
                np.linspace(1.0, 20.0, m),
                np.sort(np.random.uniform(0.5, 25.0, m)),
                np.sort(np.random.uniform(0.1, 10.0, m)),
            ]
            
            for s in starts:
                if time.time() > deadline - 30:
                    break
                try:
                    res = minimize(
                        objective_slsqp, s,
                        method='SLSQP',
                        bounds=[(0.01, 60.0)] * m,
                        constraints=constraints,
                        options={'maxiter': 100, 'ftol': 1e-10}
                    )
                    if res.fun < best_score:
                        best_score = res.fun
                        best_roots = res.x
                except Exception:
                    pass
        
        # Final polish with L-BFGS-B for speed
        if time.time() < deadline - 20:
            def objective_local(roots_flat):
                roots = np.asarray(roots_flat, dtype=np.float64)
                if np.any(roots <= 0.01) or np.any(np.diff(roots) < 0.001):
                    return 1e10
                return compute_c4_fast(roots)
            
            try:
                res = minimize(
                    objective_local, best_roots,
                    method='L-BFGS-B',
                    bounds=[(0.01, 60.0)] * m,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
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
    
    # Final validation and correction
    for i in range(len(best_roots) - 1):
        if best_roots[i+1] <= best_roots[i] + 0.001:
            best_roots[i+1] = best_roots[i] + 0.001
    
    # Convert to list
    return [float(r) for r in best_roots]
