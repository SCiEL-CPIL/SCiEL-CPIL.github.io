import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    """
    Minimize the upper bound C4 on A(f)*A(F(f)) for a Hermite-Gaussian test function.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by g(0) = 0.
    
    Returns array of three coefficients [a0, a4, a8].
    """
    
    def hermite_gaussian(x, coeffs):
        """Compute g(x) given coefficients [a0, a4, a8] with a12 derived from g(0)=0."""
        a0, a4, a8 = coeffs
        # H_n(0) values
        H0_0 = eval_hermite(0, 0)   # 1
        H4_0 = eval_hermite(4, 0)   # -12
        H8_0 = eval_hermite(8, 0)   # 1344
        H12_0 = eval_hermite(12, 0) # -103216
        
        # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
        # So a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        # Compute g(x)
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        
        poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        return poly * np.exp(-x**2 / 2)
    
    def fourier_hermite_gaussian(xi, coeffs):
        """
        Compute Fourier transform of g(x).
        F(Hn(x)*e^{-x^2/2}) = (i)^n * Hn(xi) * e^{-xi^2/2}
        """
        a0, a4, a8 = coeffs
        H0_0 = eval_hermite(0, 0)
        H4_0 = eval_hermite(4, 0)
        H8_0 = eval_hermite(8, 0)
        H12_0 = eval_hermite(12, 0)
        
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        # Fourier transform: F(Hn * e^{-x^2/2}) = (i)^n * Hn(xi) * e^{-xi^2/2}
        H0 = eval_hermite(0, xi)
        H4 = eval_hermite(4, xi)
        H8 = eval_hermite(8, xi)
        H12 = eval_hermite(12, xi)
        
        # (i)^0 = 1, (i)^4 = 1, (i)^8 = 1, (i)^12 = -1
        poly_f = a0 * 1 * H0 + a4 * 1 * H4 + a8 * 1 * H8 + a12 * (-1) * H12
        return poly_f * np.exp(-xi**2 / 2)
    
    def objective(coeffs):
        """Compute A(f) * A(F(f)) where A is the integral of |g| (L1 norm)."""
        # Use a fine grid for numerical integration
        x = np.linspace(-15, 15, 10001)
        dx = x[1] - x[0]
        
        g_x = hermite_gaussian(x, coeffs)
        A_f = np.trapz(np.abs(g_x), x)
        
        xi = np.linspace(-15, 15, 10001)
        g_f = fourier_hermite_gaussian(xi, coeffs)
        A_Ff = np.trapz(np.abs(g_f), xi)
        
        return A_f * A_Ff

    # Initial guess: the prior state of the art
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple starting points to avoid local minima
    best_result = None
    best_val = np.inf
    
    # Generate a wider range of starting points
    # The coefficients are likely small.
    starts = [
        x0,
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-0.8, 0.05, 0.0002]),
        np.array([-1.5, 0.02, 0.0001]),
        np.array([-0.5, 0.06, 0.0004]),
        np.array([-1.0, 0.01, 0.0006]),
        np.array([-1.3, 0.045, 0.00025]),
        np.array([-0.9, 0.035, 0.00035]),
        np.array([-1.1, 0.025, 0.00045]),
    ]
    
    for start in starts:
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is not None:
        return best_result.x
    else:
        return x0
