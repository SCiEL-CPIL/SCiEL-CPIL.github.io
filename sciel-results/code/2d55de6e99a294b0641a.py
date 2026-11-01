import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # 15s safety margin before 210s kill

    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395
    # H_n(0) for even n: H0=1, H4=3, H8=105, H12=10395

    def _hermite_coeffs():
        # Return coefficients for H0, H4, H8, H12 as polynomials in x
        # H0 = 1
        # H4 = x^4 - 6x^2 + 3
        # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        # H12 = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395
        return {
            0: np.array([1.0]),
            4: np.array([1.0, 0.0, -6.0, 0.0, 3.0]),  # x^4 - 6x^2 + 3
            8: np.array([1.0, 0.0, -28.0, 0.0, 210.0, 0.0, -420.0, 0.0, 105.0]),
            12: np.array([1.0, 0.0, -84.0, 0.0, 2520.0, 0.0, -30240.0, 0.0, 166320.0, 0.0, -302400.0, 0.0, 10395.0])
        }

    def _compute_g(c0, c4, c8, x):
        """Compute g(x) = P(x) * exp(-x^2/2) where P is the Hermite combination."""
        c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
        H0 = 1.0
        H4 = x**4 - 6.0 * x**2 + 3.0
        H8 = x**8 - 28.0 * x**6 + 210.0 * x**4 - 420.0 * x**2 + 105.0
        H12 = (x**12 - 84.0 * x**10 + 2520.0 * x**8 - 30240.0 * x**6 + 
               166320.0 * x**4 - 302400.0 * x**2 + 10395.0)
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        return P * np.exp(-x**2 / 2.0)

    def _compute_bound(c0, c4, c8, x_samples):
        """
        Compute the upper bound C4 on A(f)*A(Fourier(f)).
        
        The uncertainty bound involves the ratio of integrals.
        For a function g(x) = P(x) * exp(-x^2/2), we need:
        - Position variance: <x^2> = integral(x^2 * g(x)^2) / integral(g(x)^2)
        - Frequency variance: <k^2> = integral(k^2 * |F(k)|^2) / integral(|F(k)|^2)
        
        The bound C4 = 4 * <x^2> * <k^2> (or similar form)
        
        Actually, for the uncertainty principle, the bound is:
        C = 4 * sigma_x^2 * sigma_k^2 where sigma_x^2 = <x^2> - <x>^2
        
        But since g is even (Hermite polynomials of even degree times Gaussian),
        <x> = 0, so sigma_x^2 = <x^2>.
        
        For the Fourier transform, we can use the fact that for Hermite-Gaussian functions,
        the Fourier transform has a known form.
        
        Let's compute this numerically.
        """
        g_vals = _compute_g(c0, c4, c8, x_samples)
        
        # Compute <x^2> = integral(x^2 * g^2) / integral(g^2)
        g2 = g_vals**2
        int_g2 = np.trapz(g2, x_samples)
        if abs(int_g2) < 1e-15:
            return np.inf
        int_x2g2 = np.trapz(x_samples**2 * g2, x_samples)
        sigma_x2 = int_x2g2 / int_g2
        
        # For frequency variance, we need the Fourier transform.
        # For a Gaussian times polynomial, the Fourier transform can be computed analytically
        # or we can use the property that for f(x) = g(x), the frequency variance is related
        # to the derivative.
        
        # Actually, let's use the fact that for the uncertainty principle bound,
        # we can compute it via the integrals directly.
        
        # The Fourier transform of g(x) = P(x) * exp(-x^2/2) can be found by
        # using the fact that the Fourier transform of exp(-x^2/2) is sqrt(2*pi)*exp(-k^2/2)
        # and multiplication by x^n in position space corresponds to (i d/dk)^n in frequency space.
        
        # This is complex. Let's use a different approach.
        # We can compute the frequency variance using the relation:
        # <k^2> = integral(x'^2 * g(x)^2) dx / integral(g(x)^2) dx
        # where x' is the derivative operator.
        
        # Actually, for the standard uncertainty principle:
        # sigma_x^2 * sigma_k^2 >= 1/4
        # The bound C4 is the minimum over all such functions of 4*sigma_x^2*sigma_k^2
        
        # Let's compute sigma_k^2 numerically using the Fourier transform.
        # We'll use a large enough range for the FFT.
        
        # Alternatively, we can use the analytic formula for Hermite-Gaussian functions.
        # For g(x) = exp(-x^2/2), sigma_x^2 = 1/2, and the Fourier transform is
        # sqrt(2*pi)*exp(-k^2/2), so sigma_k^2 = 1/2, giving 4*(1/2)*(1/2) = 1.
        
        # For a general Hermite combination, we need to compute this more carefully.
        
        # Let's use the property that:
        # <k^2> = integral(|g'(x)|^2) dx / integral(|g(x)|^2) dx
        
        # This is a known result from the uncertainty principle.
        
        # Compute g'(x) numerically
        dg = np.gradient(g_vals, x_samples)
        int_dg2 = np.trapz(dg**2, x_samples)
        sigma_k2 = int_dg2 / int_g2
        
        # The bound is 4 * sigma_x2 * sigma_k2
        bound = 4.0 * sigma_x2 * sigma_k2
        return bound

    # Sample points for integration
    x_samples = np.linspace(-15, 15, 2000)
    
    def objective(vars):
        c0, c4, c8 = vars
        return _compute_bound(c0, c4, c8, x_samples)
    
    # We want to minimize the bound, subject to g(x) <= 1 for all x
    # and g(0) = 0 (which is already enforced by the c12 constraint)
    
    def constraint_g_leq_1(vars):
        c0, c4, c8 = vars
        vals = _compute_g(c0, c4, c8, x_samples)
        return 1.0 - np.max(np.abs(vals))
    
    # Initial guesses
    # The seed has c0 = -1.13, c4 = 0.04, c8 = 1/3240
    # Let's try several starting points
    
    best_bound = np.inf
    best_vars = None
    
    starts = [
        [1.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [1.0, -0.1, 0.0],
        [0.8, 0.0, 0.1],
        [1.2, 0.0, -0.1],
        [1.5, 0.05, 0.0],
        [0.5, -0.05, 0.02],
        [1.0, 0.05, -0.02],
        [-1.13, 0.04, 1.0/3240.0],  # Seed
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    
    for x0 in starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, x0, method='SLSQP',
                           constraints=[{'type': 'ineq', 'fun': constraint_g_leq_1}],
                           options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success:
                bound = objective(res.x)
                if bound < best_bound and constraint_g_leq_1(res.x) >= -1e-6:
                    best_bound = bound
                    best_vars = res.x
        except Exception:
            pass
    
    # If we found a good result, try to polish it with tighter tolerances
    if best_vars is not None and time.time() < deadline - 30:
        try:
            res2 = minimize(objective, best_vars, method='SLSQP',
                            constraints=[{'type': 'ineq', 'fun': constraint_g_leq_1}],
                            options={'maxiter': 500, 'ftol': 1e-12})
            if res2.success:
                bound2 = objective(res2.x)
                if bound2 < best_bound and constraint_g_leq_1(res2.x) >= -1e-6:
                    best_bound = bound2
                    best_vars = res2.x
        except Exception:
            pass
    
    if best_vars is None:
        # Fallback to seed
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    return np.array(best_vars)
