import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# Precompute fixed grid for fast integration
_GRID_N = 4096
_GRID = np.linspace(-12, 12, _GRID_N)
_DX = _GRID[1] - _GRID[0]
_EXP = np.exp(-_GRID**2 / 2.0)
_GRID2 = _GRID**2
_GRID4 = _GRID2**2
_GRID6 = _GRID4 * _GRID2
_GRID8 = _GRID4**2
_GRID10 = _GRID8 * _GRID2
_GRID12 = _GRID6**2

def _compute_bound_fast(c0, c4, c8):
    """Fast computation of the upper bound C4 using precomputed grid."""
    # c12 is fixed so g(0) = 0: H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 45360x^2 + 10395
    P = (c0 * 1.0 
         + c4 * (_GRID4 - 6.0*_GRID2 + 3.0)
         + c8 * (_GRID8 - 28.0*_GRID6 + 210.0*_GRID4 - 420.0*_GRID2 + 105.0)
         + c12 * (_GRID12 - 60.0*_GRID10 + 1260.0*_GRID8 - 12600.0*_GRID6 
                  + 45360.0*_GRID4 - 45360.0*_GRID2 + 10395.0))
    
    g = P * _EXP
    integral_g = np.sum(np.abs(g)) * _DX
    
    return integral_g ** 2

def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 on A(f)*A(Fourier(f))."""
    return _compute_bound_fast(c0, c4, c8)

def construct():
    deadline = time.time() + 195  # 195s to be safe (limit is 210s)
    
    # Initial guess: the seed values
    x0 = np.array([-113/100.0, 1/25.0, 1/3240.0])
    
    # Evaluate initial
    best_score = _compute_bound(*x0)
    best_x = x0.copy()
    
    # Phase 1: Differential evolution for global search
    try:
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        
        def objective(x):
            if time.time() > deadline - 20:
                return best_score
            return _compute_bound(x[0], x[1], x[2])
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=False
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start local optimization if time remains
    if time.time() < deadline - 15:
        rng = np.random.RandomState(123)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            try:
                start = rng.uniform(-3, 3, size=3)
                res = minimize(
                    lambda x: _compute_bound(x[0], x[1], x[2]),
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    return np.array(best_x)
