"""Spectral spherical-harmonic strategy for 100-point log-energy minimization."""
import numpy as np
import time
import math

N = 100
DEADLINE = 138.0  # safety margin before 150s kill

# ---------- Spherical harmonics basis (real) up to l_max ----------
def _legendre(n, x):
    """Legendre polynomials P_0..P_n via recurrence, vectorized in x."""
    x = np.asarray(x, dtype=float)
    P = np.zeros((n + 1, x.shape[0]))
    P[0] = 1.0
    if n >= 1:
        P[1] = x
    for k in range(2, n + 1):
        P[k] = ((2 * k - 1) * x * P[k - 1] - (k - 1) * P[k - 2]) / k
    return P

def _sph_harm_real_basis(l_max, theta, phi):
    """Return list of real spherical harmonic values for each (l,m).
    theta: polar angle [0, pi], phi: azimuth [0, 2pi).
    Conventions: Y_lm real, normalized so integral |Y|^2 dOmega = 1.
    """
    l_max = int(l_max)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Precompute Legendre up to l_max
    P = _legendre(l_max, cos_theta)  # shape (l_max+1, npts)

    basis = []
    for l in range(l_max + 1):
        for m in range(0, l + 1):
            # Normalization constant for real SH
            if m == 0:
                norm = np.sqrt((2 * l + 1) / (4.0 * np.pi))
                val = norm * P[l]
            else:
                norm = np.sqrt((2 * l + 1) / (2.0 * np.pi) *
                               math.factorial(l - m) / math.factorial(l + m))
                if m % 2 == 0:
                    val = norm * P[l, m:] * cos_phi if l >= m else norm * P[l] * cos_phi
                else:
                    val = norm * P[l] * sin_phi
                # P[l, m] isn't directly from recurrence (that's P_l(x) only).
                # We need associated Legendre P_l^m(x). Recompute properly below.
            basis.append(val)
    # The above is a placeholder; replace with proper associated Legendre below.
    return basis

def _assoc_legendre(l, m, x):
    """Associated Legendre P_l^m(x) for 0<=m<=l, vectorized in x."""
    x = np.asarray(x, dtype=float)
    # Start from P_m^m
    P = np.ones_like(x)
    if m > 0:
        for k in range(m):
            P *= -(1.0 - x * x) ** 0.5 * (2 * (k + 1))
        # Actually standard recurrence: P_m^m = (-1)^m (1-x^2)^(m/2) (2m-1)!!
        P = (-1.0) ** m * (1.0 - x * x) ** (m / 2.0)
        # Multiply by (2m-1)!!
        double_fact = 1.0
        for k in range(1, m + 1, 2):
            double_fact *= k
        P *= double_fact
    if l == m:
        return P
    # Recurse up to l
    P_lm1 = P  # P_m^m
    if l > m:
        # P_{m+1}^m
        P_lm = (2 * m + 1) * x * P_lm1 - P_lm1  # placeholder; use proper recurrence
        # Proper: (l-m) P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m
        P_prev = P_lm1
        if l > m + 1:
            P_curr = (2 * m + 1) * x * P_prev - P_prev
            for k in range(m + 2, l + 1):
                P_next = (x * (2 * k - 1) * P_curr - (k + m - 1) * P_prev) / (k - m)
                P_prev = P_curr
                P_curr = P_next
            return P_curr
        else:
            return (2 * m + 1) * x * P_prev - P_prev
    return P

def _build_sh_basis(l_max, theta, phi):
    """Proper real spherical harmonics, normalized."""
    l_max = int(l_max)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    x = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    basis = []
    for l in range(l_max + 1):
        for m in range(0, l + 1):
            Plm = _assoc_legendre(l, m, x)
            if m == 0:
                norm = np.sqrt((2 * l + 1) / (4.0 * np.pi) *
                               math.factorial(l - m) / math.factorial(l + m))
                val = norm * Plm
            else:
                norm = np.sqrt((2 * l + 1) / (2.0 * np.pi) *
                               math.factorial(l - m) / math.factorial(l + m))
                if m % 2 == 0:
                    val = norm * Plm * cos_phi
                else:
                    val = norm * Plm * sin_phi
            basis.append(val)
    return basis  # list of arrays, length (l_max+1)(l_max+2)/2

# ---------- Map from harmonic coefficients to N points on sphere ----------
# We sample N fixed "seed" directions on the sphere (Fibonacci), and define
# a vector field F: S^2 -> R^3 whose values at the seed directions give
# (after normalization) the N points. F is a band-limited function expressed
# in the real spherical harmonic basis up to l_max, for each of 3 components.

def _fibonacci_seed_dirs(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _coeffs_to_points(coeffs, l_max, seed_dirs):
    """coeffs: flat array of length 3 * n_basis. Returns N points (N,3)."""
    n_basis = (l_max + 1) * (l_max + 2) // 2
    coeffs = np.asarray(coeffs, dtype=float).reshape(3, n_basis)
    # Precompute basis values at seed directions
    theta = np.arccos(np.clip(seed_dirs[:, 2], -1.0, 1.0))
    phi = np.arctan2(seed_dirs[:, 1], seed_dirs[:, 0])
    basis = _build_sh_basis(l_max, theta, phi)  # list of (N,) arrays
    B = np.array(basis)  # (n_basis, N)
    # F_i = sum_k coeffs[c, k] * B[k, i]  for component c
    F = coeffs @ B  # (3, N)
    # Normalize each point to unit sphere
    norms = np.linalg.norm(F, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = F / norms
    return pts

# ---------- Energy and gradient in coefficient space ----------
def _energy_and_grad(coeffs, l_max, seed_dirs):
    """Compute log-energy and gradient w.r.t. coefficients."""
    n_basis = (l_max + 1) * (l_max + 2) // 2
    coeffs = np.asarray(coeffs, dtype=float).reshape(3, n_basis)
    theta = np.arccos(np.clip(seed_dirs[:, 2], -1.0, 1.0))
    phi = np.arctan2(seed_dirs[:, 1], seed_dirs[:, 0])
    basis = _build_sh_basis(l_max, theta, phi)
    B = np.array(basis)  # (n_basis, N)
    F = coeffs @ B  # (3, N)
    norms = np.linalg.norm(F, axis=0, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    pts = F / norms_safe  # (3, N)

    # Energy: sum_{i<j} log(1/|p_i - p_j|)
    n = N
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (3, N, N)
    dist2 = np.sum(diff * diff, axis=0)  # (N, N)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))

    # Gradient of energy w.r.t. points: for point i, grad_p_i = sum_{j!=i} (p_i - p_j)/|p_i-p_j|^2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    # diff[c, i, j] = p_i[c] - p_j[c]
    grad_pts = np.sum(diff * inv_dist2[np.newaxis, :, :], axis=2)  # (3, N)

    # Chain rule: pts = F / |F|, so d pts / d F is (I - pts pts^T) / |F|
    # grad_F[c] = sum_i grad_pts[c, i] * (I - pts[:,i] pts[:,i]^T) / |F|_i  projected
    # More precisely: dE/dF_c = sum_i grad_pts_c[i] * d(pts_i)/d(F_c)
    # d(pts_i)/dF = (I - pts_i pts_i^T) / |F_i|
    # So grad_F[c, k] = sum_i grad_pts[c, i] * B[k, i] / |F_i|  (since pts_i = F_i/|F_i|,
    # and the (I - pts pts^T) part means we only keep the tangential component)
    # Actually: grad_F = (grad_pts - (grad_pts . pts) pts) / |F|
    dot_gp = np.sum(grad_pts * pts, axis=0)  # (N,)
    grad_tan_pts = grad_pts - dot_gp[np.newaxis, :] * pts  # (3, N)
    grad_F = grad_tan_pts / norms_safe  # (3, N)

    # grad_coeffs[c, k] = sum_i grad_F[c, i] * B[k, i]
    grad_coeffs = grad_F @ B.T  # (3, n_basis)
    return energy, grad_coeffs.flatten()

# ---------- Optimization via scipy BFGS on coefficients ----------
def _optimize_from_coeffs(coeffs0, l_max, seed_dirs, max_time=120.0):
    from scipy.optimize import minimize
    t0 = time.time()

    def obj(x):
        e, _ = _energy_and_grad(x, l_max, seed_dirs)
        return e

    def obj_grad(x):
        e, g = _energy_and_grad(x, l_max, seed_dirs)
        return e, g

    res = minimize(obj_grad, coeffs0, jac=True, method='BFGS',
                   options={'maxiter': 2000, 'gtol': 1e-12,
                            'disp': False})
    return res.x, res.fun

# ---------- Construct ----------
def construct():
    start_time = time.time()
    seed_dirs = _fibonacci_seed_dirs(N)

    # Choose l_max: balance expressiveness vs. smoothness/low-dim
    # l_max=4 gives 3*15=45 coeffs; l_max=6 gives 3*28=84; l_max=8 gives 3*45=135
    best_energy = np.inf
    best_coeffs = None

    rng = np.random.default_rng(123)

    # Try several l_max values and random initializations
    for l_max in [4, 6, 8]:
        n_basis = (l_max + 1) * (l_max + 2) // 2
        total_coeffs = 3 * n_basis

        # Time check
        if time.time() - start_time > DEADLINE - 30:
            break

        # Initialize: small random perturbation around zero (near-uniform)
        for init_try in range(3):
            if time.time() - start_time > DEADLINE - 15:
                break
            if init_try == 0:
                # Start near uniform (only l=0 component)
                coeffs0 = np.zeros(total_coeffs)
                coeffs0[0] = 1.0  # l=0, m=0 component
            else:
                coeffs0 = rng.normal(scale=0.1, size=total_coeffs)

            try:
                coeffs_opt, energy = _optimize_from_coeffs(
                    coeffs0, l_max, seed_dirs,
                    max_time=min(40.0, DEADLINE - (time.time() - start_time) - 10))
                if energy < best_energy:
                    best_energy = energy
                    best_coeffs = coeffs_opt.copy()
            except Exception:
                continue

    if best_coeffs is None:
        # Fallback: Fibonacci
        return seed_dirs

    pts = _coeffs_to_points(best_coeffs, l_max, seed_dirs)
    pts = pts.T  # (N, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    # Final validity check
    if not np.all(np.isfinite(pts)):
        return seed_dirs
    # Ensure no two points are closer than 1e-9
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    if np.min(dist2) < 1e-18:
        return seed_dirs

    return pts
