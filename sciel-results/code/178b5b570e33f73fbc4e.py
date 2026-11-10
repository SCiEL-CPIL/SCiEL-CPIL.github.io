# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 620.0  # safety margin before 630s kill


def _check_deadline():
    return time.time() > _DEADLINE


def _build_quadrature():
    """70-point GL in cos(theta), 130 equally spaced phi."""
    x, w = np.polynomial.legendre.leggauss(70)
    theta = np.arccos(x)
    phi = np.linspace(0.0, 2.0 * np.pi, 130, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    # weights: w_i * (dphi)
    dphi = 2.0 * np.pi / 130
    W = np.outer(w, np.full(130, dphi))
    return TH, PH, W


def _real_harmonics(Y, k=10):
    """
    Build real orthonormal degree-k harmonics as a (N, 2k+1) matrix.
    Column j corresponds to m = j - k.
    Y_0 = N_0 P^0
    Y_m = sqrt(2) N_m P^m cos(m phi)  for m > 0
    Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi)  for m < 0
    N_m = sqrt((2k+1)/(4 pi) * (k-m)!/(k+m)!)
    """
    cos_t = np.cos(Y[:, 0])
    sin_t = np.sin(Y[:, 0])
    phi = Y[:, 1]
    Npts = Y.shape[0]
    cols = np.empty((Npts, 2 * k + 1))
    # factorial ratios (k-m)!/(k+m)! computed via log to avoid overflow
    from math import lgamma
    for m in range(-k, k + 1):
        abs_m = abs(m)
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                      np.exp(lgamma(k - abs_m + 1) - lgamma(k + abs_m + 1)))
        P = lpmv(abs_m, k, cos_t)  # includes Condon-Shortley phase
        if m == 0:
            cols[:, m + k] = N_m * P
        elif m > 0:
            cols[:, m + k] = np.sqrt(2.0) * N_m * P * np.cos(m * phi)
        else:
            cols[:, m + k] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi)
    return cols


def _evaluate(coeffs, Y):
    """
    Compute R = ||f||_4 / ||f||_2 given coefficients and harmonic matrix Y.
    Returns (R, f, f2, f4) for reuse.
    """
    f = Y @ coeffs
    f2 = np.sum(f * f)
    f4 = np.sum(f * f * f * f)
    n2 = np.sqrt(max(f2, 1e-300))
    n4 = f4 ** 0.25
    R = n4 / n2
    return R, f, f2, f4


def _objective(c, Y):
    R, _, _, _ = _evaluate(c, Y)
    return R


def _grad(c, Y):
    """
    Gradient of R = (sum f^4)^{1/4} / (sum f^2)^{1/2}
    Let A = sum f^4, B = sum f^2. R = A^{1/4} B^{-1/2}.
    dR/dc_j = (1/4) A^{-3/4} B^{-1/2} * sum 4 f^3 Y_j
             + A^{1/4} * (-1/2) B^{-3/2} * sum 2 f Y_j
    = A^{-3/4} B^{-1/2} sum f^3 Y_j  -  A^{1/4} B^{-3/2} sum f Y_j
    """
    f = Y @ c
    A = np.sum(f ** 4)
    B = np.sum(f ** 2)
    if A < 1e-300 or B < 1e-300:
        return np.zeros_like(c)
    A_34 = A ** (-0.75)
    B_2 = B ** (-0.5)
    A_4 = A ** 0.25
    B_32 = B ** (-1.5)
    s3 = Y.T @ (f ** 3)
    s1 = Y.T @ f
    g = A_34 * B_2 * s3 - A_4 * B_32 * s1
    return g


def _hessian(c, Y, f=None, A=None, B=None):
    """
    Hessian of R.
    R = A^{1/4} B^{-1/2}, A = sum f^4, B = sum f^2.
    Let a = A^{-3/4} B^{-1/2}, b = A^{1/4} B^{-3/2}.
    R = a * sum f^3 Y_j - b * sum f Y_j  (linear in c via f = Y c)
    Actually R is a function of c through f. Let's compute Hessian properly.
    
    R(c) = (sum (Yc)^4)^{1/4} (sum (Yc)^2)^{-1/2}
    Let F = Yc (vector). A = F^T F_3 where F_3 = F^3 elementwise... 
    Simpler: use autodiff-like via finite differences is too slow.
    Use the formula:
    dR = a * dA - b * dB  where a = (1/4)A^{-3/4}B^{-1/2}, b = (1/2)A^{1/4}B^{-3/2}
    dA = 4 sum F^3 dF = 4 sum F^3 Y_j dc_j
    dB = 2 sum F dF = 2 sum F Y_j dc_j
    So dR = A^{-3/4}B^{-1/2} sum F^3 Y_j dc_j - A^{1/4}B^{-3/2} sum F Y_j dc_j
    For Hessian, differentiate again. This is complex; use finite-difference
    Hessian-vector products instead, or just use BFGS with analytic grad.
    """
    # Return None to indicate we don't have analytic Hessian; use BFGS.
    return None


def _trust_region_step(c, g, Y, f, A, B, delta):
    """
    One trust-region-like step: minimize quadratic model on sphere.
    We'll use a simpler approach: steepest descent with backtracking line search
    on the sphere, which is more robust than full TR for this problem.
    """
    # Project gradient to tangent space
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return c
    g_tangent = g - np.dot(g, c) * c
    # Step direction: negative projected gradient
    d = -g_tangent
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        return c
    d = d / d_norm
    # Backtracking line search on sphere
    R0, _, _, _ = _evaluate(c, Y)
    c_new = c.copy()
    for lr in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]:
        step = lr * d
        c_trial = c + step
        c_trial = c_trial / np.linalg.norm(c_trial)
        R_trial, _, _, _ = _evaluate(c_trial, Y)
        if R_trial < R0:
            c_new = c_trial
            break
        c_new = c_trial
    return c_new


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical
    harmonic that MINIMIZE the norm ratio ||f||_4 / ||f||_2 on the unit sphere.
    Returns np.ndarray of shape (21,).
    """
    k = 10
    TH, PH, W = _build_quadrature()
    # Build harmonic matrix at quadrature points
    # Y matrix: shape (Npts, 21)
    cos_t = np.cos(TH)
    sin_t = np.sin(TH)
    phi = PH
    Npts = TH.size
    from math import lgamma
    Y = np.empty((Npts, 2 * k + 1))
    for m in range(-k, k + 1):
        abs_m = abs(m)
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                      np.exp(lgamma(k - abs_m + 1) - lgamma(k + abs_m + 1)))
        P = lpmv(abs_m, k, cos_t)
        if m == 0:
            Y[:, m + k] = N_m * P
        elif m > 0:
            Y[:, m + k] = np.sqrt(2.0) * N_m * P * np.cos(m * phi)
        else:
            Y[:, m + k] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi)

    # Weights for integration: W (70, 130) -> flatten to (Npts,)
    w_flat = W.ravel()
    # Orthonormality check: Y^T W Y should be ~ I
    # For now, proceed.

    def _eval_weighted(c):
        f = Y @ c
        f2 = np.sum(w_flat * f * f)
        f4 = np.sum(w_flat * f * f * f * f)
        n2 = np.sqrt(max(f2, 1e-300))
        n4 = f4 ** 0.25
        R = n4 / n2
        return R, f, f2, f4

    def _objective_w(c):
        R, _, _, _ = _eval_weighted(c)
        return R

    def _grad_w(c):
        f = Y @ c
        A = np.sum(w_flat * f ** 4)
        B = np.sum(w_flat * f * f)
        if A < 1e-300 or B < 1e-300:
            return np.zeros_like(c)
        A_34 = A ** (-0.75)
        B_2 = B ** (-0.5)
        A_4 = A ** 0.25
        B_32 = B ** (-1.5)
        s3 = Y.T @ (w_flat * f ** 3)
        s1 = Y.T @ (w_flat * f)
        g = A_34 * B_2 * s3 - A_4 * B_32 * s1
        return g

    # Start from all-ones
    c = np.ones(21, dtype=np.float64)
    c = c / np.linalg.norm(c)

    # Phase 1: BFGS on the sphere (normalize after each step)
    best_c = c.copy()
    best_R, _, _, _ = _eval_weighted(c)

    # Use scipy BFGS with a constraint that ||c||=1 via normalization trick
    # We'll minimize R(c) subject to ||c||=1 by using a penalty-free approach:
    # parameterize c = u / ||u|| and minimize R(u/||u||) over u in R^21.
    # But simpler: just do projected gradient descent.

    # Projected gradient descent with backtracking
    max_iters = 5000
    for it in range(max_iters):
        if _check_deadline():
            break
        g = _grad_w(c)
        c_norm = np.linalg.norm(c)
        g_tangent = g - np.dot(g, c) * c
        d = -g_tangent
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-12:
            break
        d = d / d_norm
        R0, _, _, _ = _eval_weighted(c)
        c_new = c.copy()
        for lr in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                   0.0078125, 0.00390625, 0.001953125]:
            step = lr * d
            c_trial = c + step
            c_trial = c_trial / np.linalg.norm(c_trial)
            R_trial, _, _, _ = _eval_weighted(c_trial)
            if R_trial < R0:
                c_new = c_trial
                break
            c_new = c_trial
        if np.allclose(c_new, c, atol=1e-14):
            break
        c = c_new
        R, _, _, _ = _eval_weighted(c)
        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Phase 2: Try multiple random restarts if time permits
    n_restarts = 20
    for r in range(n_restarts):
        if _check_deadline():
            break
        # Random start on sphere
        c_rand = np.random.randn(21)
        c_rand = c_rand / np.linalg.norm(c_rand)
        R_rand, _, _, _ = _eval_weighted(c_rand)
        if R_rand < best_R:
            best_R = R_rand
            best_c = c_rand.copy()
        # Short projected gradient descent from this start
        for it in range(2000):
            if _check_deadline():
                break
            g = _grad_w(c_rand)
            c_norm = np.linalg.norm(c_rand)
            g_tangent = g - np.dot(g, c_rand) * c_rand
            d = -g_tangent
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-12:
                break
            d = d / d_norm
            R0, _, _, _ = _eval_weighted(c_rand)
            c_new = c_rand.copy()
            for lr in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                       0.0078125, 0.00390625, 0.001953125]:
                step = lr * d
                c_trial = c_rand + step
                c_trial = c_trial / np.linalg.norm(c_trial)
                R_trial, _, _, _ = _eval_weighted(c_trial)
                if R_trial < R0:
                    c_new = c_trial
                    break
                c_new = c_trial
            if np.allclose(c_new, c_rand, atol=1e-14):
                break
            c_rand = c_new
            R, _, _, _ = _eval_weighted(c_rand)
            if R < best_R:
                best_R = R
                best_c = c_rand.copy()

    # Phase 3: Final polish with very small steps
    c = best_c.copy()
    for it in range(2000):
        if _check_deadline():
            break
        g = _grad_w(c)
        c_norm = np.linalg.norm(c)
        g_tangent = g - np.dot(g, c) * c
        d = -g_tangent
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-14:
            break
        d = d / d_norm
        R0, _, _, _ = _eval_weighted(c)
        c_new = c.copy()
        for lr in [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025]:
            step = lr * d
            c_trial = c + step
            c_trial = c_trial / np.linalg.norm(c_trial)
            R_trial, _, _, _ = _eval_weighted(c_trial)
            if R_trial < R0:
                c_new = c_trial
                break
            c_new = c_trial
        if np.allclose(c_new, c, atol=1e-15):
            break
        c = c_new
        R, _, _, _ = _eval_weighted(c)
        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Scale to satisfy validity: max|c| between 1e-3 and 1e5
    # Currently ||c||=1, so max|c| <= 1. Scale up if needed.
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)

    # Final validation
    if best_c.size != 21:
        best_c = np.ones(21)
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(21)
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        best_c = np.ones(21)

    return best_c


# EVOLVE-BLOCK-END
