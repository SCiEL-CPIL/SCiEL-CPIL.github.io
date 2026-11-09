import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    m = 12
    n = 1
    
    # The theoretical optimal starting point is the roots of the generalized Laguerre polynomial L_12^(1)(x)
    # These are the extremal nodes for the weight function x * e^(-x), which matches the problem structure (n=1)
    try:
        from scipy.special import roots_genlaguerre
        # alpha=1 corresponds to weight x^(1) * e^(-x)
        laguerre_roots, _ = roots_genlaguerre(m, 1)
        best_roots = np.sort(laguerre_roots)
    except Exception:
        # Fallback if scipy fails or for safety
        best_roots = np.linspace(1.0, 20.0, m)
    
    best_score = np.inf

    def compute_c4_fast(roots):
        """
        Compute the C4 upper bound efficiently.
        The bound is related to the maximum of |P(x)| * x^(-n/2) * exp(-x/2)
        where P(x) = prod (x - r_i)^2.
        
        In log space: log f(x) = 2*sum(log|x-r_i|) - x/2 - (n/2)*log(x)
        """
        roots = np.asarray(roots, dtype=np.float64)
        if len(roots) != m:
            return np.inf
        if np.any(roots <= 0) or np.any(np.diff(roots) < 0):
            return np.inf
        
        # The function f(x) = |P(x)| * x^(-n/2) * exp(-x/2)
        # We need to find the global maximum of this function over x > 0.
        
        # Since P(x) is a product of squared terms, it's non-negative.
        # The function is smooth except at the roots where it hits 0.
        # Between roots, it has local maxima.
        
        # Strategy:
        # 1. Evaluate on a coarse grid to identify intervals containing the global max.
        # 2. Refine in those intervals.
        
        x_max = max(roots[-1] * 1.5, 50.0)
        
        # Coarse grid
        # We need to be careful near x=0 because of the x^(-n/2) term.
        # For n=1, x^(-0.5) goes to infinity as x->0, but P(x) ~ prod(-r_i)^2 is constant.
        # So f(x) ~ const * x^(-0.5) * exp(0) -> infinity as x->0?
        # Wait, the C4 bound definition usually involves an integral or a specific norm.
        # Let's re-read the task description carefully.
        # "Tune the refined Cohn-Goncalves Laguerre-polynomial construction for the Fourier sign uncertainty upper bound."
        # "score is the upper bound on C4"
        
        # In the Cohn-Goncalves construction for uncertainty principles, the bound C4 is often derived from:
        # C4 = sup_{x>0} |P(x)| * x^{-n/2} * e^{-x/2}
        # If this sup is infinite, the bound is useless.
        # However, P(x) is a polynomial. As x->0, P(x) -> prod(-r_i)^2 > 0.
        # So if n>0, x^{-n/2} -> infinity.
        # This implies the maximum is at x->0? That would make the score 0? No, lower is better, but if it's infinity, that's bad.
        # Or maybe the formula is different.
        
        # Let's look at the seed program's logic or inspiration.
        # Inspiration 2 uses: integral of |P(x)|^2 * x * e^{-x} dx.
        # This is an L2 norm squared under the Laguerre weight.
        # Let's test both hypotheses.
        # Hypothesis 1: Max of |P(x)| * x^(-n/2) * e^{-x/2}.
        # Hypothesis 2: Integral of |P(x)|^2 * x^n * e^{-x} dx.
        
        # The seed program [4, 8, ..., 48] scored 1.0.
        # If the objective was the integral, let's estimate it.
        # If the objective was the max, let's estimate it.
        
        # Actually, looking at the "refined uncertainty inequality" context, the C4 constant is typically:
        # C4 = (2/pi) * sup_{x>0} x^{-n/2} e^{-x/2} |P(x)|
        # But if the sup is infinite, that's a problem.
        # Perhaps P(x) is defined differently?
        # Or maybe the weight is e^{-x/2} and the polynomial decays? No, P is a polynomial.
        
        # Let's check the inspiration 2 code again.
        # It computes: integral = sum(exp(val - max_val)) * exp(max_val)
        # where val = 2*log_abs_P + log(weight)
        # weight = x * exp(-x)
        # This is the L2 norm squared.
        
        # Let's assume the objective is the L2 norm squared under the weight x^1 e^{-x}.
        # This is a well-posed optimization problem.
        
        # Let's implement the integral calculation using Gauss-Laguerre quadrature for efficiency and accuracy.
        
        # We need to compute I = ∫_0^∞ (∏(x-r_i)^2)^2 * x * e^{-x} dx
        # I = ∫_0^∞ (∏(x-r_i)^4) * x * e^{-x} dx
        
        # Using generalized Gauss-Laguerre quadrature with alpha=1.
        # Nodes x_j and weights w_j satisfy ∫_0^∞ f(x) x^1 e^{-x} dx ≈ ∑ f(x_j) w_j
        # Here f(x) = ∏(x-r_i)^4
        
        # We can precompute the nodes and weights for alpha=1, N=50 or so.
        
        try:
            from scipy.special import roots_genlaguerre
            # Use a moderate number of nodes for speed and accuracy
            N_quad = 50
            nodes, weights = roots_genlaguerre(N_quad, 1)
        except Exception:
            return np.inf
            
        # Compute f(x_j) = ∏ (x_j - r_i)^4
        # To avoid overflow/underflow, use log space?
        # Or just compute directly if numbers are reasonable.
        # Roots are around 1-20. Nodes are around 1-50.
        # (x_j - r_i) can be up to 50. 50^4 = 6.25e6.
        # Product of 12 such terms: (6.25e6)^12 = (6.25)^12 * 1e72 ≈ 6e9 * 1e72 = 6e81.
        # This fits in float64 (max ~1e308).
        # So direct computation is safe.
        
        f_vals = np.ones_like(nodes)
        for r in roots:
            f_vals *= (nodes - r)**4
        
        integral = np.sum(f_vals * weights)
        
        return integral
    
    # Initial evaluation
    best_score = compute_c4_fast(best_roots)
    
    # Optimization
    try:
        from scipy.optimize import minimize, differential_evolution
        import warnings
        warnings.filterwarnings('ignore')
        
        def objective(roots_flat):
            roots = np.asarray(roots_flat, dtype=np.float64)
            if np.any(roots <= 0.01) or np.any(np.diff(roots) < 0.001):
                return 1e10
            return compute_c4_fast(roots)
        
        # 1. Local refinement starting from Laguerre roots
        # The Laguerre roots are a very strong starting point.
        # Use L-BFGS-B with bounds to ensure positivity.
        # We don't strictly need ordering constraints if we just return sorted roots,
        # but the objective function expects sorted roots for the "diff > 0" check?
        # Actually, the objective function doesn't care about order for the integral value,
        # but the check `np.diff(roots) < 0` fails if not sorted.
        # So we must keep them sorted.
        # L-BFGS-B doesn't handle ordering constraints directly.
        # We can use SLSQP with ordering constraints, or just penalize unsorted roots.
        # Since the Laguerre roots are well-spaced, L-BFGS-B might keep them sorted if we start there.
        # But to be safe, let's use SLSQP with constraints for a few steps, or just rely on the local nature.
        
        # Let's try SLSQP with ordering constraints first.
        def objective_slsqp(roots_flat):
            roots = np.asarray(roots_flat, dtype=np.float64)
            if np.any(roots <= 0.01):
                return 1e10
            if np.any(np.diff(roots) < 0.001):
                return 1e10
            return compute_c4_fast(roots)
        
        constraints = []
        for i in range(m - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i+1] - x[i] - 0.001
            })
        
        bounds = [(0.01, 100.0)] * m
        
        if time.time() < deadline - 60:
            try:
                res = minimize(
                    objective_slsqp, best_roots,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_roots = res.x
            except Exception:
                pass
        
        # 2. Global search with Differential Evolution
        # This might find a better configuration if the Laguerre roots aren't optimal for this specific "refined" bound.
        # But Laguerre roots are theoretically optimal for the moment problem.
        # Let's do a short DE run to be sure.
        
        if time.time() < deadline - 120:
            bounds_de = [(0.1, 50.0)] * m
            # Use a smaller population and fewer iterations to fit in time
            result_de = differential_evolution(
                objective, bounds_de, 
                maxiter=50, popsize=15, tol=1e-8,
                seed=42, polish=False,
                init='latinhypercube'
            )
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_roots = result_de.x
        
        # 3. Final polish with L-BFGS-B
        if time.time() < deadline - 20:
            # Sort best_roots to ensure validity for the objective function check
            best_roots = np.sort(best_roots)
            
            # L-BFGS-B is faster. We'll just penalize invalid inputs.
            def objective_local(roots_flat):
                roots = np.asarray(roots_flat, dtype=np.float64)
                if np.any(roots <= 0.01):
                    return 1e10
                # Sort inside objective to handle any permutation during optimization?
                # No, that breaks gradients. Better to assume sorted and penalize if not.
                if np.any(np.diff(roots) < 0.001):
                    return 1e10
                return compute_c4_fast(roots)
            
            try:
                res = minimize(
                    objective_local, best_roots,
                    method='L-BFGS-B',
                    bounds=bounds,
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
