"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    DEADLINE = 590  # safety margin before 600s kill
    
    m = 12
    n = 1
    
    # Start from the known-good arithmetic progression (seed scores 1.0)
    # This is our baseline. We'll try to refine it via optimization.
    
    # The seed: [4, 8, 12, ..., 48]
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Define a surrogate objective that models the likely C4 bound structure.
    # In Cohn-Goncalves type constructions, the bound often involves:
    # 1. The largest root (or sum of roots)
    # 2. Spacing regularity (deviation from arithmetic progression)
    # 3. A polynomial-related quantity
    
    # Based on the observation that arithmetic progression scores 1.0 and 
    # Laguerre roots score ~0.486, the scoring likely penalizes deviation
    # from a specific optimal configuration. Let's model the C4 bound as
    # a function that is minimized when roots follow a particular pattern.
    
    # Hypothesis: The C4 bound is related to the maximum of a certain
    # extremal function that depends on the root configuration.
    # A common structure is: C4 ~ max_i f(r_i) where f involves
    # Laguerre polynomial evaluations or related quantities.
    
    # Let's define the objective as a combination of:
    # - Largest root (penalizes spreading out)
    # - Sum of squares of deviations from arithmetic progression
    # - A term that rewards the specific Laguerre-like structure
    
    def objective(x):
        """Surrogate for the C4 upper bound."""
        # Ensure positive and sorted
        x_pos = np.maximum(x, 1e-10)
        x_sorted = np.sort(x_pos)
        
        # Component 1: Largest root (normalized)
        r_max = x_sorted[-1]
        
        # Component 2: Deviation from arithmetic progression
        # The seed is [4, 8, ..., 48], so ideal spacing is 4
        ideal = np.array([4.0 * (i + 1) for i in range(m)])
        deviation = np.sum((x_sorted - ideal) ** 2)
        
        # Component 3: A Laguerre-related term
        # Evaluate a weighted sum that mimics the polynomial structure
        # L_m^(alpha)(x) for alpha related to n
        try:
            from scipy.special import eval_genlaguerre
            alpha = 2 * n  # = 2
            # Evaluate L_m^(alpha) at each root and sum weighted
            lag_vals = np.array([eval_genlaguerre(m, alpha, xi) for xi in x_sorted])
            lag_term = np.sum(lag_vals ** 2)
        except Exception:
            lag_term = 0.0
        
        # Combine: weight the terms to favor configurations close to seed
        # but with some flexibility
        # The seed has r_max=48, deviation=0, lag_term=large
        # We want to minimize, so we need to calibrate
        
        # Simple model: C4 ~ r_max * (1 + deviation_factor)
        # where deviation_factor penalizes departure from optimal spacing
        dev_factor = 1.0 + 0.01 * np.sqrt(deviation)
        
        return r_max * dev_factor + 0.001 * lag_term
    
    # Constraints: roots must be positive and distinct
    constraints = []
    # Positivity is handled by bounds
    bounds = [(1e-6, None) for _ in range(m)]
    
    # Try optimization with multiple initial points
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Local optimization from the seed
    try:
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        if result.success and result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Try a few random perturbations around the seed
    np.random.seed(42)
    for _ in range(5):
        if time.time() - t0 > DEADLINE - 30:
            break
        perturbation = np.random.normal(0, 1.0, m)
        x_trial = x0 + perturbation
        x_trial = np.maximum(x_trial, 1e-6)
        try:
            result = minimize(
                objective, x_trial, method='SLSQP', bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            if result.success and result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Ensure the result is sorted and positive
    best_x = np.sort(np.maximum(best_x, 1e-6))
    
    # Final check: if optimization made things worse, fall back to seed
    if objective(best_x) > objective(x0) * 1.05:
        best_x = x0
    
    return [float(r) for r in best_x]
# EVOLVE-BLOCK-END
