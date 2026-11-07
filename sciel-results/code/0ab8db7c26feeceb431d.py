"""Optimization-based program: minimize r^2/(2*pi) using differentiable objective and scipy.optimize.minimize."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# H0(0)=1, H4(0)=-12, H8(0)=1008, H12(0)=-103320
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1008.0
_H12_0 = -103320.0


def _coeffs_to_q(c0, c1, c2):
    """Return (c3, q) where g=c0 H0+c1 H4+c2 H8+c3 H12, g(0)=0, q=g/x^2.
    q is a degree-10 polynomial in x with coefficients [a10, a8, a6, a4, a2].
    """
    c3 = -(_H0_0 * c0 + _H4_0 * c1 + _H8_0 * c2) / _H12_0
    # g(x)/x^2 coefficients (descending powers of x):
    # H0/x^2 -> 0 (H0=1, no x^2 term)
    # H4 = 16x^4-48x^2+12 -> H4/x^2 = 16x^2-48+12/x^2... wait, that's not polynomial.
    # Actually q = g/x^2 must divide exactly. Let's recompute.
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 2304x^6 + 4320x^4 - 1512x^2 + 1008
    # H12 = 4096x^12 - 129024x^10 + 1198080x^8 - 4644960x^6 + 6652800x^4 - 2126112x^2 + 103320
    # All have only even powers. g(0)=0 means the constant term is zero.
    # g = c0*1 + c1*(16x^4-48x^2+12) + c2*(256x^8-2304x^6+4320x^4-1512x^2+1008) + c3*(...)
    # g(0) = c0 + 12*c1 + 1008*c2 + 103320*c3 = 0
    # g/x^2: since g has no constant term and all terms have even powers >= 2,
    # g/x^2 is a polynomial of degree 10 in x (degree 12 in g, so degree 10 in g/x^2).
    # g/x^2 = c1*16*x^2 + c2*256*x^6 + c3*4096*x^10 + ... let me collect by power of x.
    # Power x^10: 4096*c3
    # Power x^8: 256*c2 + 1198080*c3
    # Power x^6: -2304*c2 - 4644960*c3
    # Power x^4: 16*c1 + 4320*c2 + 6652800*c3
    # Power x^2: -48*c1 - 1512*c2 - 2126112*c3
    # Power x^0: 0 (by construction)
    a10 = 4096.0 * c3
    a8 = 256.0 * c2 + 1198080.0 * c3
    a6 = -2304.0 * c2 - 4644960.0 * c3
    a4 = 16.0 * c1 + 4320.0 * c2 + 6652800.0 * c3
    a2 = -48.0 * c1 - 1512.0 * c2 - 2126112.0 * c3
    # q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2
    # For np.polyval, coefficients are in descending order: [a10, 0, a8, 0, a6, 0, a4, 0, a2, 0, 0]
    q = np.array([a10, 0.0, a8, 0.0, a6, 0.0, a4, 0.0, a2, 0.0, 0.0])
    return c3, q


def _find_r(q):
    """Largest real root of q at which q changes sign.
    q is a polynomial of degree 10 (in x), only even powers.
    """
    try:
        roots = np.roots(q)
    except Exception:
        return -1.0
    
    best = -1.0
    eps = 1e-7
    for r in roots:
        if abs(r.imag) > 1e-9:
            continue
        x = r.real
        if x <= 1e-9:
            continue
        # Check sign change: evaluate q at x-eps and x+eps
        try:
            qm = np.polyval(q, x - eps)
            qp = np.polyval(q, x + eps)
        except Exception:
            continue
        if qm * qp < 0:
            if x > best:
                best = x
    return best


def _objective(c):
    """Compute value = r^2/(2*pi), or a large penalty."""
    c0, c1, c2 = c
    c3, q = _coeffs_to_q(c0, c1, c2)
    # If leading coeff of q is negative, g tends to -inf, so replace g by -g
    if q[0] < 0:
        q = -q
    r = _find_r(q)
    if r < 0:
        return 1e10
    return (r * r) / (2.0 * np.pi)


def _objective_smooth(c):
    """Smoother objective: use a larger grid-based approach for better numerical stability."""
    c0, c1, c2 = c
    c3, q = _coeffs_to_q(c0, c1, c2)
    if q[0] < 0:
        q = -q
    # Find roots via np.roots
    try:
        roots = np.roots(q)
    except Exception:
        return 1e10
    
    best = -1.0
    eps = 1e-6
    for r in roots:
        if abs(r.imag) > 1e-8:
            continue
        x = r.real
        if x <= 1e-8:
            continue
        # Check sign change with slightly larger eps for robustness
        try:
            qm = np.polyval(q, x - eps)
            qp = np.polyval(q, x + eps)
        except Exception:
            continue
        if qm * qp < 0:
            if x > best:
                best = x
    if best < 0:
        return 1e10
    return (best * best) / (2.0 * np.pi)


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _objective(seed)

    best_c = seed.copy()
    best_val = seed_val

    # Try multiple optimization methods from the seed
    methods_opts = [
        ('Nelder-Mead', {'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}),
        ('Powell', {'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-14}),
        ('BFGS', {'maxiter': 1000, 'gtol': 1e-10}),
    ]

    for method, opts in methods_opts:
        try:
            res = minimize(_objective, best_c.copy(), method=method, options=opts)
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Additional local refinement with tight tolerances
    try:
        res = minimize(_objective, best_c.copy(), method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    # Try a few small perturbations around the best point
    rng = np.random.default_rng(42)
    for i in range(20):
        perturbation = rng.normal(0, 1e-4, size=3)
        cand = best_c + perturbation
        v = _objective(cand)
        if v < best_val:
            best_val = v
            best_c = cand.copy()
            # Refine from this new best
            try:
                res = minimize(_objective, best_c.copy(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
