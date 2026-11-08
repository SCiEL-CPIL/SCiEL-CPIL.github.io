import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _compute_bound(c):
    """
    c: array [c0, c4, c8, c12]
    Returns A(f)*A(Fourier(f)) where A(g) = sup |g(x)|
    for f(x) = sum c_n H_n(x) exp(-x^2/2)
    Key: Fourier transform of H_n(x) exp(-x^2/2) is i^n H_n(x) exp(-x^2/2)
    (with standard Fourier transform convention)
    """
    degrees = [0, 4, 8, 12]
    
    # Use a fine grid to compute sup norms
    # The functions decay as exp(-x^2/2), so we need enough range
    x = np.linspace(-12, 12, 8001)
    exp_term = np.exp(-x**2 / 2.0)
    
    # Compute f(x) = sum c_n H_n(x) e^{-x^2/2}
    f_vals = np.zeros_like(x)
    # Compute F(f)(x) = sum c_n i^n H_n(x) e^{-x^2/2}
    Ff_vals = np.zeros_like(x)
    
    for i, deg in enumerate(degrees):
        H = _hermite_poly(deg, x)
        f_vals += c[i] * H * exp_term
        # i^n: for n=0,4,8,12 all give i^n = 1 (since 4|n)
        Ff_vals += c[i] * (1j)**deg * H * exp_term
    
    # A(f) = sup |f(x)|
    A_f = np.max(np.abs(f_vals))
    # A(F(f)) = sup |F(f)(x)|
    A_Ff = np.max(np.abs(Ff_vals))
    
    if A_f < 1e-30:
        return np.inf
    
    return A_f * A_Ff

def _hermite_poly(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H0 = np.ones_like(x)
        H1 = 2 * x
        for k in range(2, n + 1):
            H2 = 2 * x * H1 - 2 * (k - 1) * H0
            H0 = H1
            H1 = H2
        return H1

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill
    
    # Constraint: f(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=3991680
    # c0 + 12*c4 + 1680*c8 + 3991680*c12 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 3991680
    
    def objective(params):
        c4, c8 = params
        c0 = 1.0
        c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 3991680.0
        c = np.array([c0, c4, c8, c12])
        return _compute_bound(c)
    
    best_score = np.inf
    best_params = np.array([0.0, 0.0])
    
    # Phase 1: Broad global search with differential evolution
    if time.time() - start_time < deadline - 100:
        try:
            bounds = [(-10.0, 10.0), (-5.0, 5.0)]
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=200, 
                popsize=20, tol=1e-10, polish=True,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement from many starting points
    # Generate diverse starting points around the best found
    rng = np.random.default_rng(123)
    start_points = [best_params.copy()]
    
    # Add points at various scales
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for _ in range(10):
            p = best_params + scale * rng.standard_normal(2)
            start_points.append(p)
    
    # Also try some structured points
    for c4 in [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        for c8 in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            start_points.append(np.array([c4, c8]))
    
    for x0 in start_points:
        if time.time() - start_time > deadline - 30:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < deadline - 15 and np.isfinite(best_score):
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more DE run in a tighter neighborhood if time allows
    if time.time() - start_time < deadline - 20 and np.isfinite(best_score):
        try:
            bounds_tight = [(best_params[0]-2.0, best_params[0]+2.0), 
                           (best_params[1]-1.0, best_params[1]+1.0)]
            de_result2 = differential_evolution(
                objective, bounds_tight, seed=99, maxiter=100, 
                popsize=15, tol=1e-12, polish=True
            )
            if de_result2.fun < best_score:
                best_score = de_result2.fun
                best_params = de_result2.x.copy()
                
                # Polish again
                res = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array([1.0, best_params[0], best_params[1]])
