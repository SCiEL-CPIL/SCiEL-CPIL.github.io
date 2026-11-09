import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import eval_hermite

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s kill

    # Known state-of-the-art coefficients (Goncalves et al. 2017)
    c0_known = -113.0 / 100.0
    c4_known = 1.0 / 25.0
    c8_known = 1.0 / 3240.0

    best_coeffs = np.array([c0_known, c4_known, c8_known])
    best_score = float('inf')

    # Helper to calculate c12 from c0, c4, c8 such that g(0) = 0
    # H_n(0) values for even n (physicist's Hermite polynomials):
    # H_0(0) = 1
    # H_4(0) = -3
    # H_8(0) = 105
    # H_12(0) = -10395
    def get_c12(c0, c4, c8):
        return (c0 - 3.0 * c4 + 105.0 * c8) / 10395.0

    def compute_uncertainty_product(c0, c4, c8):
        """Compute Δx · Δk for g(x) = P(x) exp(-x²/2) with g(0)=0."""
        c12 = get_c12(c0, c4, c8)
        
        def g(x):
            H0 = eval_hermite(0, x)
            H4 = eval_hermite(4, x)
            H8 = eval_hermite(8, x)
            H12 = eval_hermite(12, x)
            P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
            return P * np.exp(-x**2 / 2.0)
        
        def gp(x):
            # d/dx [H_n(x) exp(-x²/2)] = 2n H_{n-1}(x) exp(-x²/2)
            H3 = eval_hermite(3, x)
            H7 = eval_hermite(7, x)
            H11 = eval_hermite(11, x)
            return (8.0 * c4 * H3 + 16.0 * c8 * H7 + 24.0 * c12 * H11) * np.exp(-x**2 / 2.0)
        
        try:
            I_g2, _ = quad(lambda x: g(x)**2, -np.inf, np.inf, limit=200)
            I_x2g2, _ = quad(lambda x: x**2 * g(x)**2, -np.inf, np.inf, limit=200)
            I_gp2, _ = quad(lambda x: gp(x)**2, -np.inf, np.inf, limit=200)
        except Exception:
            return 1e100
        
        if I_g2 < 1e-15:
            return 1e100
        
        # By symmetry, <x> = 0
        delta_x2 = I_x2g2 / I_g2
        delta_k2 = I_gp2 / I_g2
        
        if delta_x2 < 0 or delta_k2 < 0:
            return 1e100
        
        return np.sqrt(delta_x2 * delta_k2)

    # Evaluate baseline
    baseline_val = compute_uncertainty_product(c0_known, c4_known, c8_known)
    best_score = baseline_val

    if time.time() - start_time > deadline - 30:
        return best_coeffs

    # Try local optimization with a more careful approach
    try:
        def objective(coeffs):
            if time.time() - start_time > deadline - 10:
                return best_score
            return compute_uncertainty_product(coeffs[0], coeffs[1], coeffs[2])
        
        # Use Nelder-Mead with tight tolerances, starting from known-good baseline
        result = minimize(
            objective,
            best_coeffs,
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x.copy()
            
    except Exception:
        pass
    
    if time.time() - start_time > deadline - 5:
        return best_coeffs
    
    return best_coeffs
