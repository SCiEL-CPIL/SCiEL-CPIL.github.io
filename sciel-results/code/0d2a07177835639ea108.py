import numpy as np
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200

# Use Probabilists' Hermite Polynomials (He_n)
# He_0 = 1
# He_4 = x^4 - 6x^2 + 3
# He_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# He_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficients for x^(2k) in each Hermite polynomial (k=0..6)
# Index 0 is x^0, Index 1 is x^2, ..., Index 6 is x^12
_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # He_0
    [3.0, -6.0, 1.0, 0, 0, 0, 0],      # He_4
    [105.0, -420.0, 210.0, -28.0, 1.0, 0, 0],  # He_8
    [945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0]  # He_12
])

# H_n(0) values for the constraint g(0) = 0
_H_AT_ZERO = np.array([1.0, 3.0, 105.0, 945.0])

# Precompute I_n = \int x^(2n) e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi) for n=0..14
def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            # Use log-gamma for stability
            # log((2n)!) - n*log(4) - log(n!) + 0.5*log(pi)
            log_num = math.lgamma(2*n + 1)
            log_den = n * math.log(4) + math.lgamma(n + 1)
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
    g(x) = (a0*He0 + a4*He4 + a8*He8 + a12*He12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 3*a4 + 105*a8) / 945
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute a12 from constraint g(0) = 0
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute polynomial coefficients p[k] for x^(2k)
    p = _H_COEFFS @ coeffs  # shape (7,)
    
    # Compute spatial moments m_k = \int x^(2k) |P(x)|^2 e^{-x^2} dx
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
    Minimize the uncertainty bound.
    Uses a global grid search to find the best region, then local optimization.
    """
    best_val = np.inf
    best_params = None
    
    def _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14):
        nonlocal best_val, best_params
        try:
            if time.time() > _DEADLINE - 10:
                return
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

    # The bound is scale-invariant. We can fix a0 = 1.
    # We search over a4 and a8.
    # Based on the structure, the optimal coefficients are likely small.
    # Let's do a coarse grid search first.
    
    # Phase 1: Coarse Grid Search
    # a4 in [-1, 1], a8 in [-0.1, 0.1]
    # We fix a0 = 1.0
    a4_vals = np.linspace(-1.0, 1.0, 40)
    a8_vals = np.linspace(-0.1, 0.1, 40)
    
    grid_starts = []
    for a4 in a4_vals:
        for a8 in a8_vals:
            grid_starts.append(np.array([1.0, a4, a8]))
            
    # Add the seed point (scaled to a0=1)
    # Seed: [-1.13, 0.04, 0.000308642] -> [1, -0.0354, -0.000273]
    seed_scaled = np.array([-1.13, 0.04, 0.000308642]) / -1.13
    grid_starts.append(seed_scaled)
    
    # Evaluate grid
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
    
    # Sort and take top candidates
    grid_results.sort(key=lambda x: x[0])
    
    # Phase 2: Local Optimization from top grid points
    top_candidates = [start for _, start in grid_results[:20]]
    
    for start in top_candidates:
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14)
        
    # Phase 3: Fine Grid around the best point found
    if best_params is not None and time.time() < _DEADLINE - 20:
        bp = best_params
        # Create a fine grid around bp
        # Scale step sizes
        scale = np.array([1.0, 1.0, 1.0]) # a0 is fixed effectively, but we perturb all 3
        # Actually, we should perturb in the free space.
        # Since a0 is not strictly fixed in the optimizer (it's free), we can perturb all 3.
        # But scale invariance means the surface is a cone.
        
        fine_starts = []
        for da0 in np.linspace(-0.1, 0.1, 5):
            for da4 in np.linspace(-0.05, 0.05, 5):
                for da8 in np.linspace(-0.005, 0.005, 5):
                    fine_starts.append(bp + np.array([da0, da4, da8]))
                    
        for start in fine_starts:
            if time.time() > _DEADLINE - 10:
                break
            _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-16, gtol=1e-14)
            
    # Phase 4: Final Polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        _try_optimize(best_params, method='Nelder-Mead', maxiter=50000, ftol=1e-16, gtol=1e-16)
        _try_optimize(best_params, method='L-BFGS-B', maxiter=10000, ftol=1e-18, gtol=1e-16)

    if best_params is None:
        # Fallback to seed
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
