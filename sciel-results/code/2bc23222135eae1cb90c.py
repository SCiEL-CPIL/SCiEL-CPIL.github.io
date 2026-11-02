"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import log, exp

# Precompute on a fine grid for speed
# g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
# H_n are physicists' Hermite polynomials
# g(0) = 0 => a12 = (a0 - 12*a4 + 1680*a8) / 145152

# Use a fine grid
_x_grid = np.linspace(-20, 20, 8001)
_dx = _x_grid[1] - _x_grid[0]

# Precompute Hermite polynomials and Gaussian on the grid
_x2 = _x_grid**2
_x4 = _x2**2
_x6 = _x2**3
_x8 = _x2**4
_x10 = _x2**5
_x12 = _x2**6

_H0 = np.ones_like(_x_grid)
_H4 = _x4 - 6*_x2 + 3
_H8 = _x8 - 28*_x6 + 210*_x4 - 420*_x2 + 105
_H12 = _x12 - 42*_x10 + 630*_x8 - 3780*_x6 + 7560*_x4 - 5040*_x2 + 945

_gauss = np.exp(-_x2 / 2.0)

# Precompute the basis functions: p_n(x) * gauss
_p0 = _H0 * _gauss
_p4 = _H4 * _gauss
_p8 = _H8 * _gauss
_p12 = _H12 * _gauss


def _compute_bound(a0, a4, a8):
    """
    Compute the C4 uncertainty bound for the given coefficients.
    Uses soft-min for smoothness.
    """
    # Compute a12 from g(0) = 0
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 145152.0
    
    # g(x) = a0*p0 + a4*p4 + a8*p8 + a12*p12
    g = a0*_p0 + a4*_p4 + a8*_p8 + a12*_p12
    
    # L1 norm
    l1_norm = np.sum(np.abs(g)) * _dx
    
    # hat{g}(k) has the same structure (since (-i)^n = 1 for n=0,4,8,12)
    abs_hat_g = np.abs(g)
    
    # Use soft-min with temperature parameter for smoothness
    # soft_min = -tau * log(sum(exp(-abs_hat_g / tau)))
    # For very small values, this approaches the true min
    # Use a moderate temperature
    tau = 1e-4
    # Clip to avoid overflow
    clipped = np.clip(abs_hat_g, 1e-15, None)
    neg_log_sum = -tau * np.log(np.sum(np.exp(-clipped / tau)))
    min_val = -neg_log_sum  # This is the soft-min
    
    if min_val < 1e-15:
        return 1e10
    
    return l1_norm / min_val


def _objective(params):
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the C4 uncertainty bound.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # 15 second safety margin
    
    # Start with the known prior
    best_a0, best_a4, best_a8 = -113/100.0, 1/25.0, 1/3240.0
    best_bound = _compute_bound(best_a0, best_a4, best_a8)
    
    # Phase 1: Coarse grid search around the prior
    n_coarse = 20
    a0_range = np.linspace(-2, 2, n_coarse)
    a4_range = np.linspace(-1, 1, n_coarse)
    a8_range = np.linspace(-0.05, 0.05, n_coarse)
    
    for a0 in a0_range:
        if time.time() > deadline:
            break
        for a4 in a4_range:
            if time.time() > deadline:
                break
            for a8 in a8_range:
                if time.time() > deadline:
                    break
                bound = _compute_bound(a0, a4, a8)
                if bound < best_bound:
                    best_bound = bound
                    best_a0, best_a4, best_a8 = a0, a4, a8
    
    # Phase 2: Local refinement using scipy
    try:
        from scipy.optimize import minimize, differential_evolution
        
        # Multiple restarts from different starting points
        restarts = [
            [best_a0, best_a4, best_a8],
            [-1.0, 0.05, 0.0001],
            [-1.2, 0.04, 0.0002],
            [-0.9, 0.06, 0.00005],
            [-1.1, 0.045, 0.00015],
            [-1.05, 0.048, 0.00012],
        ]
        
        for x0 in restarts:
            if time.time() > deadline:
                break
            
            # Nelder-Mead with tight tolerance
            result = minimize(
                _objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14,
                         'adaptive': True, 'initial_simplex': None}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_a0, best_a4, best_a8 = result.x[0], result.x[1], result.x[2]
        
        # Differential evolution for broader search
        bounds = [(best_a0 - 0.3, best_a0 + 0.3),
                  (best_a4 - 0.1, best_a4 + 0.1),
                  (best_a8 - 0.01, best_a8 + 0.01)]
        
        result_de = differential_evolution(
            _objective, bounds, maxiter=50, tol=1e-12,
            seed=42, polish=True, timeout=30, popsize=20
        )
        
        if result_de.fun < best_bound:
            best_bound = result_de.fun
            best_a0, best_a4, best_a8 = result_de.x[0], result_de.x[1], result_de.x[2]
            
    except Exception:
        # Fallback: simple local search
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        for step in step_sizes:
            if time.time() > deadline:
                break
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for da0 in [-step, 0, step]:
                    for da4 in [-step, 0, step]:
                        for da8 in [-step, 0, step]:
                            if da0 == da4 == da8 == 0:
                                continue
                            new_a0 = best_a0 + da0
                            new_a4 = best_a4 + da4
                            new_a8 = best_a8 + da8
                            new_bound = _compute_bound(new_a0, new_a4, new_a8)
                            if new_bound < best_bound:
                                best_bound = new_bound
                                best_a0, best_a4, best_a8 = new_a0, new_a4, new_a8
                                improved = True
    
    # Final high-precision polish
    try:
        from scipy.optimize import minimize
        result_final = minimize(
            _objective, [best_a0, best_a4, best_a8],
            method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16,
                     'adaptive': True}
        )
        if result_final.fun < best_bound:
            best_bound = result_final.fun
            best_a0, best_a4, best_a8 = result_final.x[0], result_final.x[1], result_final.x[2]
    except Exception:
        pass
    
    return np.array([best_a0, best_a4, best_a8])
# EVOLVE-BLOCK-END
