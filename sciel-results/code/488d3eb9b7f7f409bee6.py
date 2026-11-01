import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
from scipy.integrate import quad
import time

START_TIME = time.time()
TIME_LIMIT = 200.0

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def objective(coeffs):
        # Deadline guard
        if time.time() > TIME_LIMIT - 10:
            return np.inf
            
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Polynomial coefficients for P(x)
        # H0 = 1
        # H4 = x^4 - 6x^2 + 3
        # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        # H12 = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 221760x^2 + 945
        
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # Coeffs [x^12, x^11, ..., x^0]
        c12_term = c12
        c8_term = c8
        c4_term = c4
        c0_term = c0
        
        # x^12: c12
        # x^10: -84*c12
        # x^8:  2520*c12 + c8
        # x^6:  -30240*c12 - 28*c8
        # x^4:  151200*c12 + 210*c8 + c4
        # x^2:  -221760*c12 - 420*c8 - 6*c4
        # x^0:  945*c12 + 105*c8 + 3*c4 + c0
        
        coeffs_poly = np.array([
            c12_term,
            0.0,
            -84.0 * c12_term,
            0.0,
            2520.0 * c12_term + c8_term,
            0.0,
            -30240.0 * c12_term - 28.0 * c8_term,
            0.0,
            151200.0 * c12_term + 210.0 * c8_term + c4_term,
            0.0,
            -221760.0 * c12_term - 420.0 * c8_term - 6.0 * c4_term,
            0.0,
            945.0 * c12_term + 105.0 * c8_term + 3.0 * c4_term + c0_term
        ])
        
        # Find roots in [0, inf)
        try:
            roots = np.roots(coeffs_poly)
        except Exception:
            return np.inf
            
        real_roots = roots[np.isreal(roots)].real
        # Filter positive roots, slightly above 0 to avoid boundary issues
        pos_roots = real_roots[real_roots > 1e-9]
        pos_roots.sort()
        
        total_integral = 0.0
        
        # Define integration intervals
        intervals = []
        if len(pos_roots) == 0:
            intervals.append((0.0, np.inf))
        else:
            prev = 0.0
            for r in pos_roots:
                intervals.append((prev, r))
                prev = r
            intervals.append((prev, np.inf))
        
        for a, b in intervals:
            if a == b:
                continue
                
            # Determine sign of P(x) in this interval
            # Use a test point
            if b == np.inf:
                test_pt = a + 1.0
            else:
                test_pt = (a + b) / 2.0
            
            # Evaluate P at test_pt
            try:
                p_val_test = np.polyval(coeffs_poly, test_pt)
            except Exception:
                return np.inf
                
            if p_val_test == 0:
                # Degenerate, perturb
                p_val_test = 1e-10
                
            sign = 1.0 if p_val_test > 0 else -1.0
            
            # Define integrand for |P(x)| * x^4 * exp(-x^2/2)
            # = sign * P(x) * x^4 * exp(-x^2/2)
            def integrand(x):
                p_val = np.polyval(coeffs_poly, x)
                return sign * x**4 * p_val * np.exp(-0.5 * x**2)
            
            try:
                val, err = quad(integrand, a, b, limit=100, epsabs=1e-10, epsrel=1e-10)
            except Exception:
                return np.inf
                
            total_integral += val
            
        return total_integral

    # Initial bounds
    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    # Seed solution from the prompt
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed to have a baseline
    seed_score = objective(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    best_score = seed_score
    
    # Global Search using Differential Evolution
    # We use a moderate population size to balance exploration and time
    try:
        res_de = differential_evolution(
            objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6,
            polish=False  # We will do our own polish
        )
        
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_coeffs = res_de.x.copy()
    except Exception:
        pass
    
    # Local Refinement using L-BFGS-B
    # Use numerical gradients as analytic gradients for the absolute value integral are complex
    try:
        res_lbfgs = minimize(
            objective, 
            best_coeffs, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res_lbfgs.fun < best_score:
            best_score = res_lbfgs.fun
            best_coeffs = res_lbfgs.x.copy()
    except Exception:
        pass
        
    # Final Polish with SLSQP
    if time.time() < TIME_LIMIT - 15:
        try:
            res_slsqp = minimize(
                objective, 
                best_coeffs, 
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res_slsqp.fun < best_score:
                best_score = res_slsqp.fun
                best_coeffs = res_slsqp.x.copy()
        except Exception:
            pass

    return best_coeffs
