"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x) vectorized."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        H0 = np.ones_like(x)
        H1 = 2.0 * x
        for k in range(1, n):
            H2 = 2.0 * x * H1 - 2.0 * k * H0
            H0, H1 = H1, H2
        return H1

# Precompute on a fixed grid
_x = np.linspace(-15, 15, 30000)
_dx = _x[1] - _x[0]
_gauss = np.exp(-_x**2 / 2)
_H0 = _hermite(0, _x)
_H4 = _hermite(4, _x)
_H8 = _hermite(8, _x)
_H12 = _hermite(12, _x)

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty bound."""
    c12 = (c0 - 3.0*c4 + 105.0*c8) / 10395.0
    
    g = (c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12) * _gauss
    Fg = (c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12) * _gauss  # i^4k = 1
    
    abs_g2 = g*g  # real-valued
    abs_Fg2 = Fg*Fg
    
    norm_g = np.sum(abs_g2) * _dx
    norm_Fg = np.sum(abs_Fg2) * _dx
    
    if norm_g < 1e-30 or norm_Fg < 1e-30:
        return 1e10
    
    x2 = _x**2
    var_g = np.sum(x2 * abs_g2) * _dx / norm_g
    var_Fg = np.sum(x2 * abs_Fg2) * _dx / norm_Fg
    
    return var_g * var_Fg

def construct():
    """Optimize the three coefficients to minimize the uncertainty bound."""
    start_time = time.time()
    deadline = start_time + 185
    
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    seed_bound = _compute_bound(seed[0], seed[1], seed[2])
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Random sampling
    for i in range(300):
        if time.time() > deadline - 60:
            break
        p = rng.uniform(-3, 3, size=3)
        b = _compute_bound(p[0], p[1], p[2])
        if b < best_bound:
            best_bound = b
            best_params = p.copy()
    
    # Phase 2: Local refinement
    starts = [best_params.copy(), seed.copy()]
    for i in range(12):
        if time.time() > deadline - 40:
            break
        pert = best_params + rng.normal(0, 1.0, size=3)
        starts.append(pert)
    
    for s in starts:
        if time.time() > deadline - 15:
            break
        try:
            result = minimize(
                lambda p: _compute_bound(p[0], p[1], p[2]),
                s,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    # Phase 3: Differential evolution
    if time.time() < deadline - 35:
        try:
            de_result = differential_evolution(
                lambda p: _compute_bound(p[0], p[1], p[2]),
                bounds=[(-5, 5), (-5, 5), (-5, 5)],
                maxiter=200,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=True
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_params = de_result.x.copy()
        except:
            pass
    
    # Phase 4: Final polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda p: _compute_bound(p[0], p[1], p[2]),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
