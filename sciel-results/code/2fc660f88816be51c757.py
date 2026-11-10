# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

# ---------- Precomputation (module-level, done once) ----------
_K = 10
_NTH = 70   # Gauss-Legendre points in mu = cos(theta)
_NPHI = 130 # equally spaced phi points

# Build quadrature
_mu, _w = np.polynomial.legendre.leggauss(_NTH)
_mu = _mu.copy()
_w = _w.copy()
_phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)

# Precompute normalization N_m and the angular factor for each m
# Y_0   = N_0 * P_10^0(cos theta)
# Y_m>0 = sqrt(2) * N_m * P_10^m(cos theta) * cos(m phi)
# Y_m<0 = sqrt(2) * N_|m| * P_10^{|m|}(cos theta) * sin(|m| phi)
# N_m = sqrt( 21/(4 pi) * (10-m)!/(10+m)! )   (degree k=10 -> (2k+1)/(4pi))

def _norm_m(m):
    m = abs(int(m))
    num = factorial(_K - m, exact=False)
    den = factorial(_K + m, exact=False)
    return np.sqrt((2*_K+1)/(4.0*np.pi) * num/den)

# Precompute the full basis as a 21 x (NTH*NPHI) matrix? That's 21*9100 ~ 191k floats, fine.
# But for the reduced (axial) problem we only need even m in {0,2,4,6,8,10}.
# We'll precompute the full 21-basis for flexibility but optimize over the 6 axial modes.

# Evaluate each basis function on the grid (mu, phi) -> (NTH, NPHI)
# We store as shape (21, NTH, NPHI)
_basis = np.zeros((2*_K+1, _NTH, _NPHI), dtype=np.float64)
for m in range(-_K, _K+1):
    am = abs(m)
    Pm = lpmv(am, _K, _mu)  # Condon-Shortley phase included by scipy
    N = _norm_m(m)
    if m == 0:
        val = N * Pm
        _basis[m+_K, :, :] = val[:, None] * np.ones((1, _NPHI))
    elif m > 0:
        val = np.sqrt(2.0) * N * Pm * np.cos(m * _phi)[None, :]
        _basis[m+_K, :, :] = val
    else:
        val = np.sqrt(2.0) * N * Pm * np.sin(am * _phi)[None, :]
        _basis[m+_K, :, :] = val

# Flatten to (21, N)
_N = _NTH * _NPHI
_basis_flat = _basis.reshape(2*_K+1, _N)

# Quadrature weights on the sphere: dOmega = dphi dmu, total mass 4*pi
# With phi uniform (step 2pi/NPHI) and mu Gauss-Legendre (weights w_i summing to 2),
# the surface integral approx = (2pi/NPHI) * sum_i w_i * f(mu_i, phi_j)
# So per-point weight = (2pi/NPHI) * w_i (independent of j).
_wphi = 2.0 * np.pi / _NPHI
_w_sphere = np.tile(_w * _wphi, _NPHI)  # shape (N,)

# The reduced axial problem uses only even m: m in {0, 2, 4, 6, 8, 10}
# Indices in the 21-vector: m=-10..10 -> index = m+10
# We choose to use m = 0, 2, 4, 6, 8, 10 (the cos modes). The sin modes for even m
# would break axial symmetry (they are odd under phi -> -phi combined with appropriate
# rotation), so for an axially symmetric function we set all sin coefficients to 0.
# Actually, axial symmetry means f(mu, phi) = f(mu, phi + any delta) only if it's
# independent of phi, i.e., only the m=0 mode. But the OPTIMAL function for the
# L4/L2 ratio is NOT necessarily axially symmetric in the phi-independent sense.
#
# Let me reconsider. The ratio ||f||_4/||f||_2 is invariant under rotations. The
# maximizer of ||f||_4 subject to ||f||_2=1 is a function that is as "spiky" as
# possible. By symmetry, such a function must be concentrated near a point (the
# north pole), and it will be axially symmetric (invariant under rotations about the
# z-axis). An axially symmetric function has f(mu, phi) = g(mu), which means only
# the m=0 mode is present!
#
# Wait, that's not right either. A function that is concentrated near a point on
# the sphere, like a Gaussian bump, is axially symmetric about the axis through
# that point. Such a function, when expanded in spherical harmonics about the z-axis,
# contains ALL m modes (because the bump is at the pole, it's actually only m=0 if
# centered at the pole). If the bump is at the north pole (theta=0), then by axial
# symmetry about the z-axis, f(mu, phi) = g(mu), so indeed only m=0 contributes.
#
# But a pure m=0 harmonic P_10(cos theta) has 10 zeros and is not very "spiky".
# The most spiky degree-10 harmonic would be one that approximates a delta function
# at the pole. The zonal harmonic (m=0) P_10(cos theta) is the most concentrated
# at the pole among all degree-10 harmonics (by the addition theorem / Christoffel-Darboux).
#
# Actually, the question is: which degree-10 harmonic minimizes ||f||_4/||f||_2?
# This is equivalent to maximizing ||f||_2^2 / ||f||_4^{4/3} ... no.
# Minimizing ||f||_4/||f||_2 means making f as "flat" as possible (uniform on the
# sphere). A constant function would have ratio 1, but constants are not degree-10.
# So we want the degree-10 harmonic that is closest to being constant in the L4/L2
# sense.
#
# Hmm, let me reconsider. The ratio ||f||_4/||f||_2 >= 1 always (by Jensen/Hölder
# since the measure is normalized to probability). Equality holds iff |f| is constant
# a.e. Since f is a harmonic of degree 10, it cannot be constant. So we want the
# degree-10 harmonic whose |f| is as uniform as possible.
#
# This is a different problem than making f spiky! We want f to be as "flat" as
# possible. The all-ones coefficient vector (the seed) is a reasonable starting point.
#
# Let me just do a direct optimization over all 21 coefficients (or the reduced set).
# Given the 600s budget, I can afford a moderate number of optimization iterations.

# For the optimization, I'll work with all 21 coefficients but use the fact that
# the problem is scale-invariant (the ratio doesn't change if we scale c). So I can
# fix ||c||_2 = 1 and minimize ||f||_4 / ||f||_2 subject to that constraint, or
# equivalently minimize ||f||_4^2 / ||f||_2^2 (which is the square of the ratio).

# Let me define the objective and gradient efficiently.

def _compute_R_squared(coeffs):
    """Return (||f||_4^2 / ||f||_2^2, grad) where f = sum c_m Y_m.
    coeffs has shape (21,)."""
    # f on grid: shape (N,)
    f = _basis_flat @ coeffs  # (N,)
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(f2 * _w_sphere)  # ||f||_2^2
    norm4_sq = np.sum(f4 * _w_sphere)  # ||f||_4^2
    R_sq = norm4_sq / norm2_sq
    # Gradient of R_sq = norm4_sq / norm2_sq
    # d(norm4_sq)/dc = 2 * sum(2 f^3 Y_m w) = 4 * sum(f^3 Y_m w) ... let's be careful
    # norm4_sq = sum f^4 w, d/dc_j = sum 4 f^3 Y_j w = 4 * (f^3 @ Y_j w) = 4 * (f^3 * w) @ Y_j
    # norm2_sq = sum f^2 w, d/dc_j = 2 * sum f Y_j w = 2 * (f * w) @ Y_j
    fw = f * _w_sphere
    f3w = f * f2 * _w_sphere  # f^3 * w
    d_norm4 = 4.0 * (_basis_flat.T @ f3w)  # shape (21,)
    d_norm2 = 2.0 * (_basis_flat.T @ fw)    # shape (21,)
    grad = (d_norm4 * norm2_sq - norm4_sq * d_norm2) / (norm2_sq * norm2_sq)
    return R_sq, grad

# Multi-start optimization
def _optimize_one_start(c0, deadline):
    """Run L-BFGS-B from c0, return best (R_sq, c)."""
    # Normalize c0
    n0 = np.linalg.norm(c0)
    if n0 < 1e-12:
        return None
    c0 = c0 / n0
    
    def objective(c):
        if time.time() > deadline:
            raise RuntimeError("deadline")
        R_sq, g = _compute_R_squared(c)
        return R_sq, g
    
    try:
        res = minimize(objective, c0, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50})
        if time.time() > deadline:
            raise RuntimeError("deadline")
        c_best = res.x
        R_sq_best = _compute_R_squared(c_best)[0]
        return (R_sq_best, c_best)
    except RuntimeError:
        return None
    except Exception:
        return None

def rudin_problem10() -> np.ndarray:
    """
    Return 21 real coefficients c_m (m = -10..10) for a degree-10 spherical
    harmonic that minimizes the L4/L2 norm ratio on the unit sphere.
    """
    t0 = time.time()
    deadline = t0 + 590.0  # safety margin before 600s

    # Multi-start: use several random initial points on the unit sphere in R^21
    # plus the all-ones vector.
    rng = np.random.default_rng(42)
    
    starts = []
    # All-ones (normalized)
    c_ones = np.ones(21)
    starts.append(c_ones / np.linalg.norm(c_ones))
    
    # Random starts
    n_starts = 30
    for i in range(n_starts):
        c_rand = rng.standard_normal(21)
        starts.append(c_rand / np.linalg.norm(c_rand))
    
    # Also try starts that emphasize certain m values
    for m_emph in [0, 2, 4, 6, 8, 10, -10, -8, -6, -4, -2]:
        c_emph = np.zeros(21)
        c_emph[m_emph + 10] = 1.0
        # Add small random perturbation
        c_emph += 0.1 * rng.standard_normal(21)
        starts.append(c_emph / np.linalg.norm(c_emph))
    
    best_R_sq = np.inf
    best_c = np.ones(21)
    
    for i, c0 in enumerate(starts):
        if time.time() > deadline:
            break
        result = _optimize_one_start(c0, deadline)
        if result is not None:
            R_sq, c = result
            if R_sq < best_R_sq:
                best_R_sq = R_sq
                best_c = c
    
    # Normalize the result
    n = np.linalg.norm(best_c)
    if n > 1e-12:
        best_c = best_c / n
    
    # Ensure validity: max|c| in [1e-3, 1e5]
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
    
    # Final check: no NaN or inf
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(21)
    
    return best_c.astype(np.float64)

# EVOLVE-BLOCK-END
