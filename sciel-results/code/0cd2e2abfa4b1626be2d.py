"""
Uncertainty inequality: tune Hermite-Gaussian test function coefficients to minimize the upper bound on C4.

We parameterize g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is derived from the constraint g(0) = 0.

The score is the upper bound on A(f)*A(F(f)) (or C4-related constant).
We use scipy.optimize to minimize this bound over (a0, a4, a8).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution

DEADLINE_FRACTION = 0.90  # safety margin before 210s

# Precompute on a fixed grid for fast evaluation
# The function g(x) decays rapidly due to the Gaussian.
# H12(x) grows as x^12, but exp(-x^2/2) dominates.
# Range [-10, 10] should be sufficient for high accuracy.
_X_GRID = np.linspace(-10, 10, 20000)
_DX = _X_GRID[1] - _X_GRID[0]
_H0 = np.ones_like(_X_GRID)
_H4 = eval_hermite(4, _X_GRID)
_H8 = eval_hermite(8, _X_GRID)
_H12 = eval_hermite(12, _X_GRID)
_GAUSS = np.exp(-0.5 * _X_GRID**2)

# Precompute the basis functions g_k(x) = H_k(x) * exp(-x^2/2)
_B0 = _H0 * _GAUSS
_B4 = _H4 * _GAUSS
_B8 = _H8 * _GAUSS
_B12 = _H12 * _GAUSS

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on the uncertainty product for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    The bound is (||g||_inf / ||g||_2)^2.
    """
    # Derive a12 from g(0) = 0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # g(0) = (a0*1 + a4*(-3) + a8*105 + a12*(-10395)) * 1 = 0
    # a12 = (a0 - 3*a4 + 105*a8) / 10395
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Build g(x) using precomputed arrays
    g = a0*_B0 + a4*_B4 + a8*_B8 + a12*_B12
    
    # ||g||_2^2
    g2 = g**2
    norm2_sq = np.trapz(g2, dx=_DX)
    
    if norm2_sq <= 0 or not np.isfinite(norm2_sq):
        return 1e10  # invalid
    
    norm_inf = np.max(np.abs(g))
    
    bound = (norm_inf / np.sqrt(norm2_sq))**2
    
    if not np.isfinite(bound):
        return 1e10
        
    return bound

def construct():
    start_time = time.time()
    deadline = 210.0 * DEADLINE_FRACTION  # 189s
    
    # Seed coefficients
    seed_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate seed
    try:
        seed_val = _compute_bound(seed_x[0], seed_x[1], seed_x[2])
    except Exception:
        seed_val = 1e10
    
    best_val = seed_val
    best_x = seed_x.copy()
    
    # If seed is invalid, we must find any valid point
    if best_val > 1e9:
        # Try a few random points
        for _ in range(10):
            x_try = np.random.uniform(-2, 2, 3)
            v = _compute_bound(x_try[0], x_try[1], x_try[2])
            if v < best_val:
                best_val = v
                best_x = x_try
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x
    
    # Global Search using Differential Evolution
    # We want to explore a reasonable range around the seed and wider areas.
    # The coefficients a0, a4, a8 determine the shape.
    # a0 is the dominant term (H0=1).
    # a4 and a8 are corrections.
    # Let's search in a box around the seed and also a wider box to catch global minima.
    
    try:
        # Bounds for DE
        # Seed: a0=-1.13, a4=0.04, a8=0.0003
        # Let's try a range that allows significant deviation
        bounds = [
            (best_x[0] - 5.0, best_x[0] + 5.0),
            (best_x[1] - 1.0, best_x[1] + 1.0),
            (best_x[2] - 0.05, best_x[2] + 0.05)
        ]
        
        def _de_obj(x):
            return _compute_bound(x[0], x[1], x[2])
        
        # Use a moderate population and iterations to fit in time
        # popsize=15, maxiter=50 is usually fast enough for 3D
        result = differential_evolution(
            _de_obj,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            popsize=15,
            polish=False  # We will polish manually
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_x
    
    # Refine with Nelder-Mead
    try:
        result = minimize(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_x
    
    # Multi-start refinement from perturbed points to escape local minima
    np.random.seed(42)
    try:
        for _ in range(10):
            if time.time() - start_time > deadline - 5:
                break
            scale = 0.1
            x0 = best_x + scale * np.random.randn(3)
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Final high-precision polish
    if time.time() - start_time < deadline - 2:
        try:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
