"""Refined optimization for Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time
import math

def construct():
    """
    Find optimal coefficients (c0, c4, c8) for the Hermite-Gaussian test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0)=0.
    
    The objective is to minimize the upper bound on A(f)*A(Fourier(f)), which for
    this Fourier-invariant combination is sqrt(2*pi) * (sup |x*g(x)|)^2.
    
    We use a robust numerical optimization strategy:
    1. Fast evaluation of the bound using precomputed basis functions.
    2. Differential Evolution for global exploration.
    3. Basin Hopping for local refinement.
    4. Final polish with Nelder-Mead and L-BFGS-B.
    """
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit

    # Pre-compute constants
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 47900160.0

    # Grid for fast evaluation
    # Using a finer grid helps capture the peak of |x*g(x)| accurately
    N = 32768
    x = np.linspace(-12, 12, N)
    dx = x[1] - x[0]
    
    # Pre-compute Hermite polynomials on the grid
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16.0*x2*x2 - 48.0*x2 + 12.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    x10 = x8 * x2
    x12 = x6 * x6
    # Physicists' Hermite H12(x)
    H12 = (4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 
           + 37158400.0*x4 - 54432000.0*x2 + 47900160.0)
    
    # Pre-compute the basis functions for g(x)
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    # So g(x) = exp(-x^2/2) * [ c0*(H0 - H0_0/H12_0*H12) + c4*(H4 - H4_0/H12_0*H12) + c8*(H8 - H8_0/H12_0*H12) ]
    
    gauss = np.exp(-x**2 / 2.0)
    
    basis0 = (H0 - (H0_0/H12_0)*H12) * gauss
    basis4 = (H4 - (H4_0/H12_0)*H12) * gauss
    basis8 = (H8 - (H8_0/H12_0)*H12) * gauss
    
    # x * basis for computing sup |x*g(x)|
    xb0 = x * basis0
    xb4 = x * basis4
    xb8 = x * basis8

    # Smooth approximation of the sup norm using L_p norm with large p
    # This allows gradient-based optimizers to work effectively
    P = 50.0
    
    def objective_smooth(coeffs):
        """
        Compute a smooth approximation of the squared supremum of |x * g(x)|.
        coeffs: [c0, c4, c8]
        """
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        abs_gx = np.abs(gx)
        # L_p norm approximation
        val = np.sum(abs_gx ** P) ** (1.0/P)
        return val * val

    def objective_true(coeffs):
        """
        Compute the exact squared supremum of |x * g(x)| times sqrt(2*pi).
        coeffs: [c0, c4, c8]
        """
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        max_val = np.max(np.abs(gx))
        return math.sqrt(2 * math.pi) * max_val * max_val

    # Initial guess
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = objective_true(x0)
    best_x = x0.copy()
    
    # Define bounds for Differential Evolution
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]

    # 1. Differential Evolution for global search
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                objective_smooth, bounds, 
                maxiter=150, 
                popsize=20, 
                seed=42,
                tol=1e-8,
                init='sobol',
                polish=False
            )
            # Evaluate the DE result with the TRUE objective
            de_true_val = objective_true(de_result.x)
            if de_true_val < best_val:
                best_val = de_true_val
                best_x = de_result.x.copy()
    except Exception:
        pass

    # 2. Basin Hopping for local refinement
    try:
        if time.time() < deadline - 30:
            def local_minimizer(x):
                # L-BFGS-B is efficient for smooth problems
                res = minimize(objective_smooth, x, method='L-BFGS-B', 
                              options={'maxiter': 300, 'ftol': 1e-12})
                return res.x, res.fun, {}
            
            bh_result = basinhopping(
                objective_smooth, best_x, 
                minimizer=local_minimizer,
                niter=80, 
                T=0.5, 
                stepsize=0.05,
                seed=42
            )
            # Evaluate the BH result with the TRUE objective
            bh_true_val = objective_true(bh_result.x)
            if bh_true_val < best_val:
                best_val = bh_true_val
                best_x = bh_result.x.copy()
    except Exception:
        pass

    # 3. Final Polish
    # Use Nelder-Mead on the true objective for high precision
    try:
        if time.time() < deadline - 10:
            res = minimize(objective_true, best_x, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
            # Try Powell on smooth objective as a backup
            res_powell = minimize(objective_smooth, best_x, method='Powell', 
                                 options={'maxiter': 1000, 'ftol': 1e-14})
            res_powell_true_val = objective_true(res_powell.x)
            if res_powell_true_val < best_val:
                best_val = res_powell_true_val
                best_x = res_powell.x.copy()
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
