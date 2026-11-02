import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit
    
    # Pre-compute Gauss-Hermite nodes and weights
    # g is degree 12, g^2 is degree 24.
    # N=64 points integrate up to degree 127 exactly.
    N_points = 64
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
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
    
    # Hermite Polynomials (Physicists' convention)
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    # Derivatives
    dH0 = np.zeros_like(x)
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x

    # Precompute weighted vectors for fast dot products
    # We need integrals of form <x^k g^2> and <(g'-xg)^2>
    # Let's precompute x^k * w_nodes for various k
    # But g depends on coeffs, so we need to be careful.
    # I0 = sum(w * g^2)
    # Ix2 = sum(w * x2 * g^2)
    # Ideriv = sum(w * (gp - xg)^2)
    
    # To speed up, we can precompute:
    # V_k = x^k * sqrt(w)  (but g is a linear combo, so g^2 is quadratic in coeffs)
    # This is a quadratic form in coeffs.
    # Let's just compute g and gp directly, it's fast enough with N=64.
    
    # Vectorized coefficient evaluation
    # g = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # where c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    # Let's precompute the matrix M such that g = M @ [c0, c4, c8, c12]
    # Actually, let's just compute directly.
    
    def compute_bound(coeffs):
        """
        Computes the uncertainty bound for given coefficients (c0, c4, c8).
        Returns infinity if invalid.
        """
        if time.time() > deadline - 2:
            return np.inf
            
        c0, c4, c8 = coeffs
        
        # c12 is fixed to ensure g(0) = 0
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        # Evaluate g and g'
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        # Integrals
        I0 = np.dot(w_nodes, g**2)
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf
            
        Ix2 = np.dot(w_nodes, x2 * g**2)
        if not np.isfinite(Ix2) or Ix2 < 0:
            return np.inf
            
        deriv_term = gp - x * g
        Ideriv = np.dot(w_nodes, deriv_term**2)
        if not np.isfinite(Ideriv) or Ideriv < 0:
            return np.inf
            
        bound_sq = (Ix2 * Ideriv) / (4.0 * np.pi**2 * I0**2)
        
        if bound_sq < 0:
            return np.inf
            
        return np.sqrt(bound_sq)

    # Compute the seed value to know what we're beating
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_value = compute_bound(x0)
    
    best_score = seed_value
    best_coeffs = x0.copy()
    
    # Strategy: Multi-start with diverse initializations + global optimization
    # Use differential_evolution for global search, then polish with local methods
    
    # Define bounds for the coefficients
    # The coefficients can be quite large or small, so use wide bounds
    bounds = [(-100, 100), (-100, 100), (-100, 100)]
    
    # Run differential evolution for global search
    if time.time() < deadline - 60:
        try:
            de_res = opt.differential_evolution(compute_bound, bounds, 
                                               maxiter=200, tol=1e-10, 
                                               popsize=30, seed=42,
                                               polish=False,  # We'll polish manually
                                               updateinterval=0)
            if np.isfinite(de_res.fun) and de_res.fun < best_score:
                best_score = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # Multi-start local optimization from diverse points
    # Generate diverse starting points
    np.random.seed(123)
    starts = [best_coeffs]
    
    # Add the seed
    starts.append(x0.copy())
    
    # Add random points in log-space
    for _ in range(20):
        # Generate points with varying magnitudes
        scale = np.exp(np.random.normal(-3, 2, 3))
        signs = np.random.choice([-1, 1], 3)
        p = scale * signs
        starts.append(p)
    
    # Add structured points
    for i in range(3):
        for scale in [0.01, 0.1, 1.0, 10.0, 50.0]:
            for sign in [-1, 1]:
                p = np.zeros(3)
                p[i] = scale * sign
                starts.append(p)
    
    # Add points along diagonals
    for scale in [0.1, 0.5, 1.0, 5.0, 20.0]:
        for d in [[1,1,1], [1,-1,1], [-1,1,1], [1,1,-1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]:
            p = np.array(d) * scale
            starts.append(p)
    
    # Local optimization from each start
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        # Try Nelder-Mead
        try:
            res = opt.minimize(compute_bound, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 500, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
        
        # Try BFGS
        if time.time() < deadline - 10:
            try:
                res_bfgs = opt.minimize(compute_bound, x_start, method='BFGS',
                                       options={'maxiter': 300, 'gtol': 1e-12})
                if np.isfinite(res_bfgs.fun) and res_bfgs.fun < best_score:
                    best_score = res_bfgs.fun
                    best_coeffs = res_bfgs.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 8:
        try:
            res2 = opt.minimize(compute_bound, best_coeffs, method='L-BFGS-B',
                               options={'maxiter': 3000, 'ftol': 1e-20, 'gtol': 1e-16})
            if np.isfinite(res2.fun) and res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
        except Exception:
            pass
    
    # Additional polish with Powell
    if time.time() < deadline - 5:
        try:
            res3 = opt.minimize(compute_bound, best_coeffs, method='Powell',
                               options={'maxiter': 1000, 'xtol': 1e-16, 'ftol': 1e-18})
            if np.isfinite(res3.fun) and res3.fun < best_score:
                best_score = res3.fun
                best_coeffs = res3.x.copy()
        except Exception:
            pass
    
    # Final COBYLA polish
    if time.time() < deadline - 3:
        try:
            res4 = opt.minimize(compute_bound, best_coeffs, method='COBYLA',
                               options={'maxiter': 2000, 'tol': 1e-14})
            if np.isfinite(res4.fun) and res4.fun < best_score:
                best_score = res4.fun
                best_coeffs = res4.x.copy()
        except Exception:
            pass
    
    # Verify the result is valid
    final_val = compute_bound(best_coeffs)
    if not np.isfinite(final_val) or final_val > seed_value:
        return x0.copy()
    
    return np.array(best_coeffs)
