import numpy as np
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Using physicists' convention
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# H_12 = 4096x^12 - 98304x^10 + 1290240x^8 - 6451200x^6 + 12902400x^4 - 6451200x^2 + 665280

# Coefficients for x^(2k) in each Hermite polynomial
_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # H_0
    [12.0, -48.0, 16.0, 0, 0, 0, 0],   # H_4
    [1680.0, -13440.0, 13440.0, -3072.0, 256.0, 0, 0],  # H_8
    [665280.0, -6451200.0, 12902400.0, -6451200.0, 1290240.0, -98304.0, 4096.0]  # H_12
])

# H_n(0) values for the constraint g(0) = 0
_H_AT_ZERO = np.array([1.0, 12.0, 1680.0, 665280.0])

# Precompute I_n = ∫ x^(2n) e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi) for n=0..14
def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

# Matrix M where M[k, i, j] = I_{k+i+j} for k=0,1; i,j=0..6
_M_SPATIAL = np.zeros((2, 7, 7))
_M_FREQ = np.zeros((2, 7, 7))
for k in range(2):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                _M_SPATIAL[k, i, j] = _I_TABLE[idx]
                _M_FREQ[k, i, j] = 2.0 * np.pi * _I_TABLE[idx]

# Signs for frequency domain: q_k = p_k * (-1)^k
_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2><xi^2>) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute a12 from constraint g(0) = 0
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute polynomial coefficients p[k] for x^(2k)
    p = _H_COEFFS @ coeffs  # shape (7,)
    
    # Compute spatial moments m_k = ∫ x^(2k) |P(x)|^2 e^{-x^2} dx
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    
    # Compute frequency moments
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        
        dq = dp * _SIGNS
        
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    """
    Minimize the uncertainty bound using a 2D reduced search space.
    Since the bound is scale-invariant, fix a0 = -1 and search over (a4, a8).
    Returns three coefficients (a0, a4, a8) for H0, H4, H8.
    """
    best_val = np.inf
    best_params = None
    
    def _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14):
        """Helper to try optimization from a starting point."""
        nonlocal best_val, best_params
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, start, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-16, 'fatol': 1e-16})
            else:
                return
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Since the bound is scale-invariant, fix a0 = -1 and search over (a4, a8)
    # Generate a dense grid in the 2D space
    a4_range = np.linspace(-0.1, 0.2, 40)
    a8_range = np.linspace(-0.002, 0.002, 30)
    
    grid_starts = []
    for a4 in a4_range:
        for a8 in a8_range:
            grid_starts.append(np.array([-1.0, a4, a8]))
    
    # Add the known good point and its neighborhood
    known_good = np.array([-1.13, 0.04, 0.000308642])
    # Normalize to a0 = -1
    known_good_norm = known_good / known_good[0] * (-1.0)
    for da4 in np.linspace(-0.05, 0.05, 15):
        for da8 in np.linspace(-0.0005, 0.0005, 10):
            grid_starts.append(np.array([-1.0, known_good_norm[1] + da4, known_good_norm[2] + da8]))
    
    # Random starts in the 2D reduced space
    rng = np.random.default_rng(42)
    for _ in range(50):
        a4 = rng.uniform(-0.2, 0.3)
        a8 = rng.uniform(-0.005, 0.005)
        grid_starts.append(np.array([-1.0, a4, a8]))
    
    # Evaluate grid points
    grid_results = []
    for start in grid_starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            val, _ = _compute_bound_and_grad(start)
            if np.isfinite(val):
                grid_results.append((val, start))
        except Exception:
            pass
    
    # Sort by value and take top candidates for local optimization
    grid_results.sort(key=lambda x: x[0])
    
    # Optimize from top 30 grid points
    top_candidates = [start for _, start in grid_results[:30]]
    
    for start in top_candidates:
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14)
    
    # Phase 2: Local refinement around best found point
    if best_params is not None and time.time() < _DEADLINE - 15:
        # Try Nelder-Mead polish with very tight tolerances
        _try_optimize(best_params, method='Nelder-Mead', maxiter=50000, ftol=1e-16, gtol=1e-16)
        
        # Try L-BFGS-B with tighter tolerances
        if time.time() < _DEADLINE - 10:
            _try_optimize(best_params, method='L-BFGS-B', maxiter=10000, ftol=1e-18, gtol=1e-16)
    
    # Phase 3: If time remains, do a finer grid around the best point
    if best_params is not None and time.time() < _DEADLINE - 5:
        bp = best_params
        # Scale back to original parameter space for fine grid
        fine_starts = []
        for da0 in np.linspace(-0.05, 0.05, 5):
            for da4 in np.linspace(-0.005, 0.005, 5):
                for da8 in np.linspace(-0.00005, 0.00005, 3):
                    fine_starts.append(bp + np.array([da0, da4, da8]))
        
        for start in fine_starts:
            if time.time() > _DEADLINE - 3:
                break
            _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-16, gtol=1e-14)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
