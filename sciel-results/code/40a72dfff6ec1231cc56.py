import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    """
    Optimize the three free Hermite coefficients (H0, H4, H8) to minimize
    the uncertainty bound, with H12 determined by g(0)=0.
    """
    # Hermite polynomial values at 0 for n = 0, 4, 8, 12
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    def get_all_coeffs(a0, a4, a8):
        """Return all four coefficients with a12 determined by g(0)=0."""
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        return np.array([a0, a4, a8, a12])
    
    def g_func(x, coeffs):
        """Evaluate g(x) = sum a_n H_n(x) exp(-x^2/2)"""
        n_vals = [0, 4, 8, 12]
        result = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        for n, a in zip(n_vals, coeffs):
            result += a * eval_hermite(n, x) * np.exp(-x**2 / 2)
        return result
    
    def g_hat_func(xi, coeffs):
        """
        Fourier transform of g. Using convention where
        F[H_n(x) e^{-x^2/2}](xi) = (i)^n H_n(xi) e^{-xi^2/2}
        (up to normalization constants that cancel in the ratio)
        """
        n_vals = [0, 4, 8, 12]
        result = np.zeros_like(xi) if isinstance(xi, np.ndarray) else 0.0
        for n, a in zip(n_vals, coeffs):
            # The FT of H_n(x)e^{-x^2/2} under unitary convention
            # involves (i)^n factor. For L1 norm we need |g_hat|
            result += a * (1j)**n * eval_hermite(n, xi) * np.exp(-xi**2 / 2)
        return result
    
    def compute_l1_norms(coeffs):
        """Compute ||g||_1 and ||g_hat||_1 via numerical quadrature."""
        # Compute ||g||_1
        def abs_g(x):
            return np.abs(g_func(x, coeffs))
        
        # Compute ||g_hat||_1  
        def abs_g_hat(xi):
            return np.abs(g_hat_func(xi, coeffs))
        
        # Integrate over a large enough domain
        # g and g_hat are Gaussian-like, so [-10, 10] should suffice
        l1_g, _ = quad(lambda x: float(abs_g(x)), -10, 10, limit=200)
        l1_g_hat, _ = quad(lambda xi: float(abs_g_hat(xi)), -10, 10, limit=200)
        
        return l1_g, l1_g_hat
    
    def objective(params):
        """Objective: minimize the product of L1 norms (proxy for uncertainty bound)."""
        a0, a4, a8 = params
        coeffs = get_all_coeffs(a0, a4, a8)
        
        # Handle potential numerical issues
        l1_g, l1_g_hat = compute_l1_norms(coeffs)
        
        # Avoid division by zero or negative values
        if l1_g < 1e-12 or l1_g_hat < 1e-12:
            return 1e6
        
        # The uncertainty bound is proportional to l1_g * l1_g_hat
        return l1_g * l1_g_hat
    
    # Initial guess: use the seed values as starting point for optimization
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Run optimization
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < 1e6:
            optimal_params = result.x
        else:
            optimal_params = x0
            
    except Exception:
        # Fallback to seed values if optimization fails
        optimal_params = x0
    
    return np.array(optimal_params)
