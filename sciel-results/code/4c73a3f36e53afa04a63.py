"""
Optimized Uncertainty Bound Minimization for Hermite-Gaussian Test Functions.

Strategy:
1. Identify that for the basis {H0, H4, H8, H12}, the Fourier transform is
   self-reciprocal (up to phase), so Var(xi) = Var(x).
2. The objective reduces to minimizing Var(x) = <x^2 g, g> / <g, g>.
3. Due to the large index gaps (0, 4, 8, 12), the moment matrix M is diagonal.
4. The problem becomes minimizing a Rayleigh quotient with a linear constraint g(0)=0.
5. Solve the resulting 3-variable optimization using a dense grid and local polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 10.0

# Precompute weights W_n = sqrt(pi) * 2^n * n!
# and variances lambda_n = (n+1)/2 for pure H_n
def _get_weights_and_lambdas(orders):
    W = {}
    L = {}
    for n in orders:
        W[n] = math.sqrt(math.pi) * (2.0 ** n) * math.factorial(n)
        L[n] = (n + 1.0) / 2.0
    return W, L

_ORDERS = [0, 4, 8, 12]
_W, _L = _get_weights_and_lambdas(_ORDERS)

# Coefficients for H_n(0)
# H_0(0) = 1
# H_4(0) = 3
# H_8(0) = 105
# H_12(0) = 945
_H0_VALS = {0: 1.0, 4: 3.0, 8: 105.0, 12: 945.0}

def _compute_bound(a4, a8, a12):
    """
    Compute the variance product (which equals Var(x) in this case)
    for g = a0*H0 + a4*H4 + a8*H8 + a12*H12, with a0 determined by g(0)=0.
    """
    a0 = -(_H0_VALS[4] * a4 + _H0_VALS[8] * a8 + _H0_VALS[12] * a12) / _H0_VALS[0]
    
    # Numerator: sum a_n^2 * lambda_n * W_n
    num = 0.0
    num += (a0 ** 2) * _L[0] * _W[0]
    num += (a4 ** 2) * _L[4] * _W[4]
    num += (a8 ** 2) * _L[8] * _W[8]
    num += (a12 ** 2) * _L[12] * _W[12]
    
    # Denominator: sum a_n^2 * W_n
    den = 0.0
    den += (a0 ** 2) * _W[0]
    den += (a4 ** 2) * _W[4]
    den += (a8 ** 2) * _W[8]
    den += (a12 ** 2) * _W[12]
    
    if den < 1e-300:
        return 1e18
        
    return num / den

def _objective(params):
    return _compute_bound(params[0], params[1], params[2])

def construct():
    """
    Return the three Hermite coefficients (degrees 0, 4, 8).
    """
    # Initial guess from seed
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    # Extract a4, a8, a12 from seed? 
    # Seed returns a0, a4, a8. a12 is derived.
    # Let's just optimize a4, a8, a12 directly.
    
    # The seed coefficients:
    # a0 = -1.13
    # a4 = 0.04
    # a8 = 0.0003086
    # a12 = -(a0 + 3a4 + 105a8)/945
    a0_seed = -1.13
    a4_seed = 0.04
    a8_seed = 0.0003086
    a12_seed = -(a0_seed + 3.0*a4_seed + 105.0*a8_seed) / 945.0
    
    start_params = np.array([a4_seed, a8_seed, a12_seed])
    best_val = _objective(start_params)
    best_params = start_params.copy()
    
    # Dense grid search
    # Scale of parameters: a4 ~ 0.04, a8 ~ 0.0003, a12 ~ 0.00001
    # Let's search in relative terms or small absolute ranges
    
    # Ranges based on seed magnitude
    r4 = 0.1
    r8 = 0.001
    r12 = 0.0001
    
    # Coarse grid
    for d4 in np.linspace(-r4, r4, 11):
        for d8 in np.linspace(-r8, r8, 11):
            for d12 in np.linspace(-r12, r12, 11):
                if time.time() > _DEADLINE - _MARGIN:
                    break
                p = start_params + np.array([d4, d8, d12])
                val = _objective(p)
                if val < best_val:
                    best_val = val
                    best_params = p.copy()
    
    # Local polish
    if time.time() < _DEADLINE - _MARGIN:
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - _MARGIN:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-18})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    a4, a8, a12 = best_params
    a0 = -(_H0_VALS[4] * a4 + _H0_VALS[8] * a8 + _H0_VALS[12] * a12) / _H0_VALS[0]
    
    # Return a0, a4, a8
    if not np.all(np.isfinite([a0, a4, a8])):
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        
    return np.array([a0, a4, a8])
