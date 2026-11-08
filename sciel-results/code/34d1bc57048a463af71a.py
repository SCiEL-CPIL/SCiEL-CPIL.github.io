"""Fourier-invariant Hermite-Gaussian test function tuned via analytic spectral moments."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 175.0  # safety margin before 210s kill


def _hermite_moments_analytic(a0, a4, a8):
    """
    Compute the uncertainty bound analytically using exact Hermite-Gaussian
    orthogonality relations with cross-terms.
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    
    Uses: x^2 H_n = (1/4)H_{n+2} + (n+1/2)H_n + n(n-1)H_{n-2}
    and: ∫ H_m H_n e^{-x^2} dx = sqrt(pi) 2^n n! δ_mn
    """
    # Enforce g(0) = 0: H_n(0) for even n
    # H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-47880
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    # a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    c = np.array([a0, a4, a8, a12])
    ns = np.array([0, 4, 8, 12])
    sqrt_pi = np.sqrt(np.pi)
    
    def norm_sq_n(n):
        return sqrt_pi * (2.0 ** n) * float(np.math.factorial(int(n)))
    
    # Norm squared: ||g||^2 = sum |c_n|^2 * ||H_n e^{-x^2/2}||^2
    norm_sq = 0.0
    for ci, ni in zip(c, ns):
        norm_sq += ci**2 * norm_sq_n(ni)
    
    if norm_sq < 1e-30:
        return 1e10
    
    # Compute ∫ x^2 |g|^2 dx using cross-terms
    # x^2 H_j = (1/4)H_{j+2} + (j+0.5)H_j + j(j-1)H_{j-2}
    # ∫ x^2 H_i H_j e^{-x^2} dx = sqrt_pi * [
    #   (1/4) * 2^j * j! * δ_{i,j+2} +
    #   (j+0.5) * 2^j * j! * δ_{i,j} +
    #   j(j-1) * 2^j * j! * δ_{i,j-2}
    # ]
    
    def moment_x2(i, j):
        """∫ x^2 H_i H_j e^{-x^2} dx"""
        result = 0.0
        wj = (2.0 ** j) * float(np.math.factorial(int(j)))
        if i == j + 2:
            result += 0.25 * sqrt_pi * wj
        if i == j:
            result += (j + 0.5) * sqrt_pi * wj
        if j >= 2 and i == j - 2:
            result += j * (j - 1) * sqrt_pi * wj
        return result
    
    moment_x2_total = 0.0
    for i_idx in range(4):
        for j_idx in range(4):
            moment_x2_total += c[i_idx] * c[j_idx] * moment_x2(int(ns[i_idx]), int(ns[j_idx]))
    
    # Uncertainty bound: A(f)*A(F) ~ norm_sq * moment_x2_total (up to constants)
    bound = norm_sq * moment_x2_total
    return bound


def _compute_bound_with_fourier(a0, a4, a8):
    """
    Compute a more accurate bound that accounts for the Fourier-side moments.
    For Hermite-Gaussian functions, F[H_n(x)e^{-x^2/2}] = i^n H_n(x)e^{-x^2/2},
    so the Fourier side has the same structure. The product A(f)A(F) involves
    the geometric mean of spatial and Fourier variances.
    
    For this self-similar family, <x^2> = <ξ^2>, so the product scales as <x^2>.
    But the actual bound C4 involves a specific combination. We use the ratio
    form which is scale-invariant: <x^2> = moment_x2 / norm_sq.
    """
    # Enforce g(0) = 0
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    c = np.array([a0, a4, a8, a12])
    ns = np.array([0, 4, 8, 12])
    sqrt_pi = np.sqrt(np.pi)
    
    def norm_sq_n(n):
        return sqrt_pi * (2.0 ** n) * float(np.math.factorial(int(n)))
    
    norm_sq = sum(ci**2 * norm_sq_n(int(ni)) for ci, ni in zip(c, ns))
    if norm_sq < 1e-30:
        return 1e10
    
    def moment_x2(i, j):
        result = 0.0
        wj = (2.0 ** j) * float(np.math.factorial(int(j)))
        if i == j + 2:
            result += 0.25 * sqrt_pi * wj
        if i == j:
            result += (j + 0.5) * sqrt_pi * wj
        if j >= 2 and i == j - 2:
            result += j * (j - 1) * sqrt_pi * wj
        return result
    
    moment_x2_total = 0.0
    for i_idx in range(4):
        for j_idx in range(4):
            moment_x2_total += c[i_idx] * c[j_idx] * moment_x2(int(ns[i_idx]), int(ns[j_idx]))
    
    # The bound is proportional to the variance <x^2> = moment_x2 / norm_sq
    # Since the problem asks for the upper bound on C4, and for this family
    # the product A(f)A(F) is controlled by <x^2>, we minimize <x^2>.
    variance = moment_x2_total / norm_sq
    return variance


def _objective(a):
    """Objective function for optimization."""
    return _compute_bound_with_fourier(a[0], a[1], a[2])


def _solve_eigenvalue():
    """
    Solve the generalized eigenvalue problem for the Rayleigh quotient
    <x^2> = (a^T Nc a) / (a^T Nw a) subject to g(0)=0.
    """
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    # a12 = (a0 - 12*a4 + 1680*a8) / 47880
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0/47880.0, -12.0/47880.0, 1680.0/47880.0],
    ])
    
    k = np.array([0, 4, 8, 12], dtype=float)
    fact = np.array([float(np.math.factorial(int(j))) for j in [0, 4, 8, 12]])
    w = (2.0 ** k) * fact
    c = (2.0 * k + 1.0) * w  # diagonal approximation (no cross-terms)
    
    Nc = M.T @ np.diag(c) @ M
    Nw = M.T @ np.diag(w) @ M
    
    try:
        eigvals, eigvecs = eigh(Nc, Nw)
        idx = np.argmin(eigvals)
        v = eigvecs[:, idx]
        a = v[:3]
        scale = np.max(np.abs(a))
        if scale > 0:
            a = a / scale
        return a
    except Exception:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])


def construct():
    start = time.time()
    deadline = start + _TIME_LIMIT
    
    # Step 1: Get initial guess from eigenvalue solve
    best_a = _solve_eigenvalue()
    best_val = _compute_bound_with_fourier(best_a[0], best_a[1], best_a[2])
    
    # Step 2: Refine with local optimization from multiple starting points
    initial_points = [
        best_a.copy(),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-0.5, 0.5, -0.5]),
        np.array([0.5, -0.5, 0.5]),
    ]
    
    rng = np.random.default_rng(42)
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(5):
            initial_points.append(rng.uniform(-scale, scale, size=3))
    
    for x0 in initial_points:
        elapsed = time.time() - start
        if elapsed > deadline - 15:
            break
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_a = result.x.copy()
        except Exception:
            continue
    
    # Step 3: Final high-precision polish
    elapsed = time.time() - start
    if elapsed < deadline - 5:
        try:
            result = minimize(_objective, best_a, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_a = result.x.copy()
        except Exception:
            pass
    
    # Ensure finite result
    if not np.all(np.isfinite(best_a)):
        best_a = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    return best_a
# EVOLVE-BLOCK-END
