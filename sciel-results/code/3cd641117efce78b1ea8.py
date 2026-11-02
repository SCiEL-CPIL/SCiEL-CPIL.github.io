import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit

    # Use exact Gauss-Hermite quadrature.
    # g is degree 12. g^2 is degree 24.
    # x^2 g^2 is degree 26.
    # (g' - xg)^2 involves derivatives, so degree increases.
    # g' is degree 11. xg is degree 13.
    # g' - xg is degree 13.
    # (g' - xg)^2 is degree 26.
    # So all integrands are polynomials of degree <= 26 times e^{-x^2}.
    # Gauss-Hermite with N points integrates polynomials up to degree 2N-1 exactly.
    # We need 2N-1 >= 26 => 2N >= 27 => N >= 13.5.
    # So N=14 is sufficient for exact integration.
    N_points = 14
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Pre-compute powers and Hermite polynomials at nodes
    x = x_nodes
    x2 = x**2
    x3 = x2 * x
    x4 = x2**2
    x5 = x4 * x
    x6 = x4 * x2
    x7 = x6 * x
    x8 = x4**2
    x9 = x8 * x
    x10 = x8 * x2
    x11 = x10 * x
    x12 = x6**2
    
    # Hermite polynomials (physicists' convention)
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    # Derivatives
    dH0 = np.zeros_like(x)
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x

    def compute_score_fast(coeffs):
        """
        Computes the score using exact Gauss-Hermite quadrature.
        coeffs: (c0, c4, c8)
        """
        if time.time() > deadline - 2:
            return np.inf
            
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        # Integrals
        I0 = np.sum(w_nodes * g**2)
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf
                
        Ix2 = np.sum(w_nodes * x2 * g**2)
        if not np.isfinite(Ix2) or Ix2 < 0:
            return np.inf
                
        deriv_term = gp - x * g
        Ideriv = np.sum(w_nodes * deriv_term**2)
        if not np.isfinite(Ideriv) or Ideriv < 0:
            return np.inf
                
        A_f_sq = Ix2 / I0
        A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
        
        return np.sqrt(A_f_sq * A_gf_sq)

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_score_fast(x0)
    best_coeffs = x0.copy()
    
    # Strategy:
    # 1. Differential Evolution for global search (robust to local minima)
    # 2. L-BFGS-B for high-precision polish
    
    # Define bounds for Differential Evolution
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    try:
        # Differential Evolution is slow but global.
        res_de = opt.differential_evolution(
            compute_score_fast, 
            bounds, 
            seed=42, 
            maxiter=150, 
            popsize=20,
            tol=1e-10,
            init='sobol',
            mutation=(0.5, 1.0),
            recombination=0.9
        )
        
        if res_de.fun < best_score and np.isfinite(res_de.fun):
            best_score = res_de.fun
            best_coeffs = res_de.x.copy()
            
    except Exception:
        pass

    # Polish with L-BFGS-B - tighter tolerances
    if time.time() < deadline - 15:
        try:
            res_lbfgs = opt.minimize(
                compute_score_fast, 
                best_coeffs, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if res_lbfgs.fun < best_score and np.isfinite(res_lbfgs.fun):
                best_score = res_lbfgs.fun
                best_coeffs = res_lbfgs.x.copy()
        except Exception:
            pass

    # If time permits, try more random restarts with local polish
    # to ensure we didn't miss a better basin.
    np.random.seed(42)
    for i in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random start within reasonable bounds
        x_try = np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.05, 0.05)
        ])
        
        try:
            res_restart = opt.minimize(
                compute_score_fast, 
                x_try, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if res_restart.fun < best_score and np.isfinite(res_restart.fun):
                best_score = res_restart.fun
                best_coeffs = res_restart.x.copy()
        except Exception:
            pass

    # Final high-precision polish pass with trust-constr for exact local solve
    if time.time() < deadline - 5:
        try:
            res_final = opt.minimize(
                compute_score_fast, 
                best_coeffs, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if res_final.fun < best_score and np.isfinite(res_final.fun):
                best_score = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
