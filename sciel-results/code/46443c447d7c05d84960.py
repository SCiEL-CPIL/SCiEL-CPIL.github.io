import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss

def construct():
    t_start = time.time()
    deadline = 205.0 # 5s margin from 210s
    
    # Degrees
    degrees = [0, 4, 8, 12]
    
    # Precompute Hermite values on a grid for the sup-norm
    # The function decays as e^{-pi x^2}, so P(x) doesn't need to be huge.
    # But P(x) is a polynomial. The product P(x) e^{-pi x^2} is what matters?
    # No, A(g) = sup |P(x)|.
    # Wait, if A(g) = sup |P(x)|, then for large x, P(x) is large.
    # This would make A(g) infinite unless P(x) is bounded?
    # Polynomials are unbounded.
    # So A(f) cannot be sup |P(x)|.
    
    # Let's reconsider A(f).
    # In the context of the uncertainty principle, A(f) is often the "amplitude" or "concentration".
    # If g(x) = e^{-pi x^2} P(x), then |g(x)| = |P(x)| e^{-pi x^2}.
    # The sup of |g(x)| is finite.
    # Let's assume A(f) = sup_x |f(x)|.
    # Then A(g) = sup_x |P(x)| e^{-pi x^2}.
    # And A(\hat{g}) = sup_x |Q(x)| e^{-pi x^2} = sup_x |P(x)| e^{-pi x^2}.
    # So J = (sup_x |P(x)| e^{-pi x^2})^2.
    # Again, scale invariance. We need normalization.
    # Assume ||g||_2 = 1.
    # Then J = (sup_x |P(x)| e^{-pi x^2})^2 / ||P||_2^2.
    
    # Let's implement this.
    
    # Grid for sup
    x_grid = np.linspace(-10, 10, 2000)
    H_grid = np.array([eval_hermite(d, x_grid) for d in degrees])
    
    # Gauss-Hermite for norm
    # Int P(x)^2 e^{-2 pi x^2} dx
    # Change variable y = sqrt(2 pi) x
    # dx = dy / sqrt(2 pi)
    # Integral = 1/sqrt(2 pi) * Int P(y/sqrt(2 pi))^2 e^{-y^2} dy
    n_points = 200
    y_points, y_weights = hermgauss(n_points)
    x_points = y_points / np.sqrt(2 * np.pi)
    H_points = np.array([eval_hermite(d, x_points) for d in degrees])
    
    def get_c12(c0, c4, c8):
        # c0*1 + c4*(-3) + c8*(105) + c12*(10395) = 0
        return (-c0 + 3*c4 - 105*c8) / 10395.0

    def objective(params):
        c0, c4, c8 = params
        c12 = get_c12(c0, c4, c8)
        coeffs = [c0, c4, c8, c12]
        
        # Calculate P on grid
        P_grid = np.zeros_like(x_grid)
        for i, d in enumerate(degrees):
            P_grid += coeffs[i] * H_grid[i]
        
        # Calculate Sup of |P| * exp(-pi x^2)
        gauss_factor = np.exp(-np.pi * x_grid**2)
        sup_val = np.max(np.abs(P_grid * gauss_factor))
        
        # Calculate Norm
        P_points = np.zeros_like(x_points)
        for i, d in enumerate(degrees):
            P_points += coeffs[i] * H_points[i]
            
        norm_sq = np.sum((P_points**2) * y_weights) / np.sqrt(2 * np.pi)
        
        if norm_sq < 1e-12:
            return 1e12
            
        return (sup_val**2) / norm_sq

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.000308])
    
    best_score = objective(x0)
    best_params = x0
    
    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() - t_start > deadline:
        # Return best found
        pass

    # Construct final array
    c0, c4, c8 = best_params
    # The task asks for "three coefficients of the Hermite combination".
    # The return interface is np.array([c0, c4, c8]).
    return np.array(best_params)
