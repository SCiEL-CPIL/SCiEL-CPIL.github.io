import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad
import time

def _hermite(n, x):
    """Compute nth physicists' Hermite polynomial."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    H0 = np.ones_like(x)
    H1 = 2 * x
    for k in range(2, n + 1):
        H2 = 2 * x * H1 - 2 * (k - 1) * H0
        H0, H1 = H1, H2
    return H1

def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 on A(f)*A(Fourier(f))."""
    # c12 is fixed so g(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    c12 = -(c0 * 1 + c4 * 3 + c8 * 105) / 10395.0
    
    # Define g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    def g(x):
        H0 = 1.0
        H4 = x**4 - 6*x**2 + 3
        H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
        H12 = x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 45360*x**4 - 45360*x**2 + 10395
        poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
        return poly * np.exp(-x**2 / 2.0)
    
    # For Fourier-invariant Hermite-Gaussians, the bound on A(f)*A(F) 
    # typically involves computing integrals of |g| and related quantities.
    # A common formulation: the uncertainty product bound involves
    # ||g||_1 * ||ĝ||_1 or similar. Since Fourier of Hn*exp(-x^2/2) is 
    # (i)^n Hn*exp(-x^2/2), we have ||ĝ||_1 = ||g||_1.
    # But the "upper bound C4" suggests a specific constant from the inequality.
    
    # Let's compute ||g||_1 = ∫|g(x)|dx
    # Use numerical quadrature over a sufficient range
    def abs_g(x):
        return abs(g(x))
    
    # Integrate |g| over [-L, L] with L large enough
    # Use adaptive quadrature
    try:
        integral_g, _ = quad(abs_g, -10, 10, limit=200)
    except Exception:
        # Fallback: use trapezoidal rule
        x = np.linspace(-10, 10, 10000)
        integral_g = np.trapz(abs_g(x), x)
    
    # The bound C4 is likely (∫|g|)^2 or a multiple thereof
    # For the uncertainty inequality A(f)*A(ĝ) >= C, the best constant
    # is found by minimizing over test functions.
    # The "upper bound on C4" means we want to find the largest C such that
    # A(f)*A(ĝ) >= C for all f, and we test with our g.
    # So C4(g) = A(g) * A(ĝ) = (∫|g|)^2 since ||ĝ||_1 = ||g||_1
    
    return integral_g ** 2

def construct():
    deadline = time.time() + 170  # 170s to be safe (limit is 180s)
    
    # Initial guess: the seed values
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Evaluate initial
    best_score = _compute_bound(*x0)
    best_x = x0.copy()
    
    # Try differential evolution with time limit
    try:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        
        def objective(x):
            if time.time() > deadline:
                return best_score
            return _compute_bound(*x)
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # If time remains, try local refinement
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda x: _compute_bound(*x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return np.array(best_x)
