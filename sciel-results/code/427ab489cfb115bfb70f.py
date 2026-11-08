"""Bootstrap DIRECT solution: derive the Hermite-Gaussian coefficients from the
Fourier-invariant C4 uncertainty-product structure, with a hard wall-clock guard."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

# Wall-clock deadline guard (evaluator kills at 210 s; keep a safety margin).
T_START = time.time()
DEADLINE = 210.0 - 15.0  # 195 s hard stop


def _now():
    return time.time()


def _time_left():
    return DEADLINE - _now()


# Physically normalised probabilists' Hermite polynomials.
# He_n(x) is the probabilists' Hermite polynomial (recurrence He_{n+1}=x He_n - n He_{n-1}).
# The Fourier-invariant (self-dual up to sign) Gaussian-Hermite eigenfunctions use the
# physicists' form; here we work with a fixed basis so only the relative weights matter
# for the uncertainty product (a scale-invariant ratio). We use the standard physicists'
# H_n with the 2^{n/2} sqrt(n! pi) normalisation factor folded into the basis.
def _H(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)


def _norm_factor(n):
    """2^{n/2} * sqrt(n! * pi) : L2 norm of H_n(x) exp(-x^2/2) over R."""
    return (2.0 ** (0.5 * n)) * math.sqrt(math.factorial(n) * math.pi)


def _make_basis():
    """Return a set of orthogonal-ish basis vectors for degrees 0,4,8,12 on a grid,
    each multiplied by the Gaussian weight, pre-normalised to unit L2 norm."""
    xs = np.linspace(-40.0, 40.0, 40001)
    dx = xs[1] - xs[0]
    w = np.exp(-0.5 * xs * xs)
    basis = {}
    for n in (0, 4, 8, 12):
        h = _H(n, xs)
        f = h * w
        # normalise to unit L2 norm
        f = f / np.sqrt(np.sum(f * f) * dx + 1e-300)
        basis[n] = f
    return xs, dx, w, basis


XS, DX, W, BASIS = _make_basis()


def _coeff12(c0, c4, c8):
    """Derive the H12 weight so that the combination has P(0)=0 (g(0)=0).
    g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    => c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0).
    We work in the unnormalised polynomial values at 0."""
    h0 = _H(0, 0.0)
    h4 = _H(4, 0.0)
    h8 = _H(8, 0.0)
    h12 = _H(12, 0.0)
    if abs(h12) < 1e-15:
        return 0.0
    return -(c0 * h0 + c4 * h4 + c8 * h8) / h12


def _uncertainty_product(c0, c4, c8):
    """Compute the C4 upper bound: A(f)*A(Fourier(f)) for the constructed f.

    A(f) = sqrt(Var_x(f)) * sqrt(Var_p(f)) where the Fourier transform of a
    Gaussian-Hermite combination is itself a Gaussian-Hermite combination. For
    Fourier-invariant construction the product is scale-invariant; we return the
    raw product of the two standard deviations (lower is better)."""
    c12 = _coeff12(c0, c4, c8)
    f = (c0 * BASIS[0] + c4 * BASIS[4] + c8 * BASIS[8] + c12 * BASIS[12])
    # Renormalise f to unit L2 (the product is scale invariant, but keep it sane).
    nrm = np.sqrt(np.sum(f * f) * DX + 1e-300)
    f = f / nrm

    # Fourier transform of f on the grid (use FFT with appropriate scaling).
    # f is defined on xs in [-40,40]; its FT is the Gaussian-Hermite dual.
    # For a self-similar basis, FT(H_n * g)(xi) = (-i)^n H_n(xi) g(xi) (with the
    # physicists' convention and unit-variance Gaussian). We exploit this: the
    # Fourier transform of our combination is the same linear combination with
    # the same weights (up to phase) in the SAME basis, because each H_n*g is an
    # eigenfunction of the FT. Hence Var_p = Var_x for the same weighted sum.
    # Therefore the product A(f)*A(FT f) = Var_x(f)^2.
    xf = XS * f
    xf2 = XS * XS * f
    mean_x = np.sum(xf) * DX
    var_x = np.sum(xf2) * DX - mean_x * mean_x
    var_x = max(var_x, 1e-15)
    return var_x  # = (sqrt(var_x)*sqrt(var_x)) ; the C4 product for FT-invariant f


def _score(c0, c4, c8):
    """The upper bound C4 value (lower is better)."""
    try:
        return _uncertainty_product(c0, c4, c8)
    except Exception:
        return float('inf')


def _coarse_grid():
    """Direct structured sweep over the three free coefficients."""
    best = None
    best_val = float('inf')
    # Symmetry: coefficients can be scaled; the product is scale-invariant, so we
    # fix c8 = 1 (unless it would make the combination degenerate) and search c0/c8,
    # c4/c8 ratios. This reduces 3D search to 2D.
    ratios_c0 = np.linspace(-30.0, 30.0, 61)
    ratios_c4 = np.linspace(-30.0, 30.0, 61)
    for r0 in ratios_c0:
        if _time_left() < 15.0:
            break
        for r4 in ratios_c4:
            v = _score(r0, r4, 1.0)
            if v < best_val:
                best_val = v
                best = (r0, r4, 1.0)
    return best, best_val


def _local_refine(c0, c4, c8, iters=400):
    """Deterministic coordinate + Nelder-Mead style refinement around a start point."""
    best = (c0, c4, c8)
    best_val = _score(c0, c4, c8)
    # Simple Nelder-Mead with a few iterations, bounded.
    pts = [
        (c0, c4, c8),
        (c0 + 0.5, c4, c8),
        (c0 - 0.5, c4, c8),
        (c0, c4 + 0.5, c8),
        (c0, c4 - 0.5, c8),
        (c0, c4, c8 + 0.5),
        (c0, c4, c8 - 0.5),
    ]
    vals = [_score(*p) for p in pts]
    for _ in range(iters):
        if _time_left() < 15.0:
            break
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = tuple(np.mean(pts[:-1], axis=0))
        alpha = 1.0
        reflection = tuple(centroid[i] + alpha * (centroid[i] - pts[-1][i]) for i in range(3))
        vr = _score(*reflection)
        if vr < vals[0]:
            # expansion
            gamma = 2.0
            expansion = tuple(centroid[i] + gamma * (reflection[i] - pts[-1][i]) for i in range(3))
            ve = _score(*expansion)
            if ve < vr:
                pts[-1], vals[-1] = expansion, ve
            else:
                pts[-1], vals[-1] = reflection, vr
        elif vr < vals[-2]:
            pts[-1], vals[-1] = reflection, vr
        else:
            # contraction
            rho = 0.5
            contraction = tuple(centroid[i] + rho * (pts[-1][i] - centroid[i]) for i in range(3))
            vc = _score(*contraction)
            if vc < vals[-1]:
                pts[-1], vals[-1] = contraction, vc
            else:
                # shrink toward best
                for i in range(1, 3):
                    pts[i] = tuple(pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(3))
                    vals[i] = _score(*pts[i])
        cur_best = min(range(3), key=lambda i: vals[i])
        if vals[cur_best] < best_val:
            best_val = vals[cur_best]
            best = pts[cur_best]
    return best, best_val


def construct():
    from scipy.optimize import minimize

    # Seed prior-art coefficients as a fallback / initial guess.
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])

    best = tuple(seed.tolist())
    best_val = _score(*best)

    # Phase 1: coarse structured sweep (2D via scale-invariance, c8 fixed to 1).
    if _time_left() > 30.0:
        grid_best, grid_val = _coarse_grid()
        if grid_best is not None and grid_val < best_val:
            best, best_val = grid_best, grid_val
        # Also try the seed-scaled point and a few hand-derived structured points.
        for cand in [
            (-113.0, 1.0, 3240.0 / 100.0),
            (-20.0, 4.0, 1.0),
            (-50.0, 10.0, 1.0),
            (0.0, 0.0, 1.0),
        ]:
            v = _score(*cand)
            if v < best_val:
                best, best_val = cand, v

    # Phase 2: Use scipy.optimize.minimize for thorough local search from multiple starts.
    def _obj_2d(params):
        """Objective in 2D: params = [c0/c8, c4/c8], c8=1."""
        return _score(params[0], params[1], 1.0)

    # Multiple starting points for the 2D optimization
    starts = [
        np.array([best[0] / best[2], best[1] / best[2]]) if abs(best[2]) > 1e-15 else np.array([0.0, 0.0]),
        np.array([-1.13, 0.04]),  # seed ratios
        np.array([-2.0, 0.5]),
        np.array([-5.0, 1.0]),
        np.array([-10.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([-3.0, -0.5]),
        np.array([3.0, 0.5]),
    ]

    for start in starts:
        if _time_left() < 10.0:
            break
        try:
            res = minimize(_obj_2d, start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best = (res.x[0], res.x[1], 1.0)
                best_val = res.fun
        except Exception:
            pass

    # Phase 3: Final Nelder-Mead refinement on the 3D space from the best 2D point.
    if _time_left() > 15.0:
        ref_best, ref_val = _local_refine(*best, iters=500)
        if ref_val < best_val:
            best, best_val = ref_best, ref_val

    # Return the three free coefficients (degrees 0, 4, 8).
    return np.array([best[0], best[1], best[2]])
# EVOLVE-BLOCK-END
