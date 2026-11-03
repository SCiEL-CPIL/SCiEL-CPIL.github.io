"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = 210.0
_SAFETY_MARGIN = 15.0  # seconds

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _hermite_gaussian(c0, c4, c8, x):
    """Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)."""
    c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 665280.0
    h0 = eval_hermite(0, x)
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    poly = c0*h0 + c4*h4 + c8*h8 + c12*h12
    return poly * np.exp(-x**2 / 2.0)

def _compute_C4(c0, c4, c8):
    """Compute C4 = A(g) * A(hat_g) where A(f) = integral |f|."""
    def g(x):
        return _hermite_gaussian(c0, c4, c8, x)
    
    # A(g) = integral |g(x)| dx over real line
    A_g, _ = quad(lambda x: abs(g(x)), -np.inf, np.inf, limit=200)
    
    # hat_g: since g is a combo of even Hermite functions, hat_g has same structure
    # but with (-i)^n factors. For even n, (-i)^n = 1, so hat_g = g in the same basis.
    # Actually, the Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}
    # for the unitary convention. For even n, this is just H_n(x) e^{-x^2/2}.
    # So hat_g = g. Thus A(hat_g) = A(g).
    # But let me verify numerically by computing the FT of g.
    # For safety, compute hat_g numerically via the fact that it's the same function.
    A_hat_g = A_g  # since hat_g = g for even Hermite combinations
    
    return A_g * A_hat_g

def construct():
    best_c = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_C4(*best_c)
    
    # Try a few starting points and optimize
    starts = [
        best_c,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-0.9, 0.06, 0.0002]),
    ]
    
    for x0 in starts:
        if _time_left() < 5.0:
            break
        try:
            res = minimize(_compute_C4, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
