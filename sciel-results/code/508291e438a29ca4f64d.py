import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit
    
    # Pre-compute Gauss-Hermite nodes and weights
    # Degree of g is 12, so g^2 is degree 24.
    # Gauss-Hermite with N points integrates polynomials up to degree 2N-1 exactly.
    # We need 2N-1 >= 24 => N >= 12.5. 
    # Let's use N=64 for high precision and speed.
    N_points = 64
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Pre-compute powers of x_nodes for efficiency
    x = x_nodes
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    x3 = x2 * x
    x5 = x4 * x
    x7 = x6 * x
    x9 = x8 * x
    x11 = x10 * x
    
    # Pre-compute Hermite polynomials at nodes
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    # Pre-compute derivatives of Hermite polynomials at nodes
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x
    
    def eval_g_and_deriv(coeffs):
        """
        Evaluates g(x) and g'(x) at the Gauss-Hermite nodes.
        coeffs: (c0, c4, c8)
        Returns: g_vals, gp_vals
        """
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c4*dH4 + c8*dH8 + c12*dH12
        
        return g, gp

    def compute_score(coeffs):
        if time.time() > deadline - 2:
            return np.inf
            
        c0, c4, c8 = coeffs
        try:
            g, gp = eval_g_and_deriv(coeffs)
            
            # Integrals: sum(w_i * f(x_i))
            # I0 = sum(w * g^2)
            I0 = np.sum(w_nodes * g**2)
            if I0 <= 1e-30 or not np.isfinite(I0):
                return np.inf
                
            # Ix2 = sum(w * x^2 * g^2)
            Ix2 = np.sum(w_nodes * x_nodes**2 * g**2)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf
                
            # Ideriv = sum(w * (gp - x*g)^2)
            deriv_term = gp - x_nodes * g
            Ideriv = np.sum(w_nodes * deriv_term**2)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf
                
            # Product
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf

    # Multi-start Nelder-Mead with diverse initial guesses
    np.random.seed(42)
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # Seed guess
    initial_guesses.append(np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0]))
    
    # Random guesses in reasonable ranges
    for _ in range(20):
        c0 = np.random.uniform(-2.0, 2.0)
        c4 = np.random.uniform(-0.5, 0.5)
        c8 = np.random.uniform(-0.01, 0.01)
        initial_guesses.append(np.array([c0, c4, c8]))
    
    # Also try some structured guesses
    for c0 in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        for c4 in [-0.3, 0.0, 0.3]:
            for c8 in [-0.005, 0.0, 0.005]:
                initial_guesses.append(np.array([c0, c4, c8]))
    
    best_score = np.inf
    best_coeffs = initial_guesses[0].copy()
    
    # Run Nelder-Mead from each initial guess
    for x0 in initial_guesses:
        if time.time() > deadline - 30:
            break
        
        try:
            res = opt.minimize(compute_score, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'adaptive': True})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    try:
        res2 = opt.minimize(compute_score, best_coeffs, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res2.fun < best_score and np.isfinite(res2.fun):
            best_score = res2.fun
            best_coeffs = res2.x.copy()
    except Exception:
        pass
    
    # One more Nelder-Mead polish if time permits
    if time.time() < deadline - 20:
        try:
            res3 = opt.minimize(compute_score, best_coeffs, method='Nelder-Mead', 
                               options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 200, 'adaptive': True})
            if res3.fun < best_score and np.isfinite(res3.fun):
                best_score = res3.fun
                best_coeffs = res3.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
