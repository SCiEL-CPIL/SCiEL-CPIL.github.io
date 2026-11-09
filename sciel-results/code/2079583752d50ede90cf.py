"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import roots_laguerre, eval_laguerre
from scipy.optimize import differential_evolution
import time

def _compute_bound(roots):
    """
    Compute the upper bound C4 for a given set of 12 positive roots.
    The polynomial is P(x) = prod (x - r_i)^2.
    The bound is estimated as sqrt( I1 / I0 ) where I0 = int P(x) e^-x dx, I1 = int x P(x) e^-x dx.
    """
    if len(roots) != 12:
        return 1e10
    if any(r <= 0 for r in roots):
        return 1e10
    
    # Use Gauss-Laguerre quadrature for integrals of form int f(x) e^-x dx
    # We need high precision for degree 24 polynomial.
    # n=100 points should be sufficient for degree 24.
    x_q, w_q = np.polynomial.laguerre.laggauss(100)
    
    # Compute P(x_q) = prod (x_q - r_i)^2
    # To avoid overflow/underflow issues, we can compute log P or just P directly
    # Degree 24, roots up to ~50. x_q max is ~400.
    # (400)^24 is huge. Let's use logs or scaling.
    # Actually, the ratio I1/I0 is scale invariant if we scale x? No, x is fixed.
    # Let's compute P(x) directly with float64.
    # Max root ~ 50. x_q max ~ 400.
    # (400-50)^24 ~ 350^24 ~ 10^59. This is within float64 range (10^308).
    
    P_vals = 1.0
    for r in roots:
        diff = x_q - r
        P_vals *= diff * diff
        
    I0 = np.sum(P_vals * w_q)
    I1 = np.sum(x_q * P_vals * w_q)
    
    if I0 <= 0:
        return 1e10
        
    bound = np.sqrt(I1 / I0)
    return bound

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580.0  # Safety margin
    
    # Initial guess: Roots of L_12(x)
    try:
        r0, _ = roots_laguerre(12)
        initial_roots = sorted([float(x) for x in r0])
    except Exception:
        initial_roots = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    # Check the bound for the initial guess
    current_best_roots = initial_roots
    current_best_score = _compute_bound(current_best_roots)
    
    # Define bounds for differential evolution
    # Roots should be positive.
    # Based on L_12 roots, they range from ~0.6 to ~50.
    # Let's set bounds [0.1, 100.0] for each root.
    bounds = [(0.1, 100.0)] * 12
    
    # We want to minimize the bound.
    # Differential evolution is stochastic and robust.
    # Set maxiter and popsize to fit within time.
    # Each evaluation takes ~1ms (100 quadrature points).
    # 1000 evaluations = 1s. We have 580s.
    # We can afford many iterations.
    
    try:
        # Use differential evolution
        # tol=1e-6 to stop when improvement is small
        # maxiter=10000
        result = differential_evolution(
            _compute_bound,
            bounds,
            maxiter=20000,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,  # Use L-BFGS-B at the end for refinement
            callback=None,
            update_interval=0,
            workers=-1  # Use all cores if possible
        )
        
        if result.success and result.fun < current_best_score:
            # Sort the roots to maintain consistent ordering (though order doesn't matter for the product)
            optimized_roots = sorted([float(x) for x in result.x])
            # Verify positivity
            if all(r > 0 for r in optimized_roots):
                current_best_roots = optimized_roots
                current_best_score = result.fun
                
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Final check: ensure we return a valid list of 12 positive floats
    if len(current_best_roots) != 12 or any(r <= 0 for r in current_best_roots):
        return initial_roots
        
    return current_best_roots

if __name__ == "__main__":
    roots = construct()
    print(roots)
