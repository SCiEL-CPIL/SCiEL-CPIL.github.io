import time, math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 600.0
_START = time.time()

def _laguerre_square_integral(m):
    # int_0^inf [L_m(x)]^2 e^{-x} dx = binom(2m, m)
    return math.comb(2*m, m)

def _integral_monic(roots):
    # monic poly P = prod (x - r)^2 ; integral of P e^{-x}
    # build coefficients
    r2 = np.concatenate([roots, roots])
    coeffs = np.poly(r2)  # monic
    # coeffs[0]=1, degree 2m
    k = np.arange(len(coeffs))
    # integral = sum coeffs[k] * k!
    fact = np.array([math.factorial(int(j)) for j in k], dtype=float)
    return float(np.sum(coeffs * fact))

def construct():
    m = 12
    target = _laguerre_square_integral(m)
    # initial guess: spread roots
    best_roots = np.linspace(0.5, 10.0, m)
    best_val = float('inf')

    def obj_and_con(x):
        # x = log roots
        roots = np.exp(x)
        # objective: largest root
        return roots.max()

    # Use penalty for integral constraint
    def penalty_obj(x):
        roots = np.exp(x)
        val = roots.max()
        integ = _integral_monic(roots)
        # relative penalty
        rel = (integ - target)/target
        return val + 1e6*rel*rel + 1e3*max(0.0, -rel)

    x0 = np.log(best_roots)
    res = minimize(penalty_obj, x0, method='Nelder-Mead',
                   options={'maxiter': 4000, 'xatol':1e-10, 'fatol':1e-12})
    roots = np.exp(res.x)
    roots = np.sort(roots)
    # verify positivity
    if np.all(roots > 0) and np.isfinite(roots).all():
        best_roots = roots
        best_val = roots.max()

    # deadline guard
    if time.time() - _START > _DEADLINE - 30:
        pass
    return [float(r) for r in best_roots]
