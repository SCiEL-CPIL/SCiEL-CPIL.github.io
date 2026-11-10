# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time
from math import factorial, pi


def _build_basis(k=10):
    """
    Build real orthonormal spherical harmonic basis values on a fixed quadrature grid.
    Returns:
        Y: array of shape (2k+1, n_theta, n_phi)
        weights: array of shape (n_theta, n_phi) for surface integration
    """
    n_theta = 70
    n_phi = 130

    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    phi_weights = np.full(n_phi, 2 * np.pi / n_phi)
    weights = np.outer(w, phi_weights)

    coeffs = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        num = factorial(k - abs_m)
        den = factorial(k + abs_m)
        N_m = np.sqrt((2 * k + 1) / (4 * pi) * num / den)
        
        P = lpmv(abs_m, k, x)
        
        if m == 0:
            coeffs[k, :, :] = N_m * P[:, np.newaxis]
        elif m > 0:
            coeffs[k + m, :, :] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            coeffs[k + m, :, :] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return coeffs, weights


def _normalize_unit_l2(c, Y, weights):
    """Normalize c so that ||f||_2 = 1 (unit L2 norm on the sphere)."""
    f = np.tensordot(c, Y, axes=([0], [0]))
    norm2_sq = np.sum(f * f * weights)
    if norm2_sq > 0:
        return c / np.sqrt(norm2_sq)
    return c


def _compute_ratio(c, Y, weights):
    """Compute R = ||f||_4 / ||f||_2."""
    f = np.tensordot(c, Y, axes=([0], [0]))
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    norm4_4 = np.sum(f2 * f2 * weights)
    if norm2_sq <= 0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    return norm4 / norm2


def _objective_and_grad(c, Y, weights):
    """Return (R, grad_R) for the L4/L2 ratio."""
    f = np.tensordot(c, Y, axes=([0], [0]))
    f2 = f * f
    f3 = f * f2
    f4 = f2 * f2
    
    norm2_sq = np.sum(f2 * weights)
    norm4_sq = np.sum(f4 * weights)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_sq ** 0.25
    R = norm4 / norm2
    
    dA_dc = 4.0 * np.tensordot(weights * f3, Y, axes=([0, 1], [1, 2]))
    dB_dc = 2.0 * np.tensordot(weights * f, Y, axes=([0, 1], [1, 2]))
    
    A = norm4_sq
    B = norm2_sq
    
    dR_dc = (0.25 * A**(-0.75) * dA_dc) / np.sqrt(B) - \
            (0.5 * A**0.25 * B**(-1.5) * dB_dc)
    
    return R, dR_dc


def _lgbfsb_optimize(c0, Y, weights, maxiter=2000, ftol=1e-15, gtol=1e-14):
    """Run L-BFGS-B on the objective with gradient."""
    from scipy.optimize import minimize
    
    def obj_and_grad(c):
        R, g = _objective_and_grad(c, Y, weights)
        return R, g
    
    res = minimize(obj_and_grad, c0, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
    return res.x, res.fun


def _nelder_mead_minimize(c0, Y, weights, maxiter=3000, xatol=1e-14, fatol=1e-14):
    """Run Nelder-Mead on the objective (no gradient needed)."""
    from scipy.optimize import minimize
    
    res = minimize(lambda c: _compute_ratio(c, Y, weights), c0, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
    return res.x, res.fun


def _gradient_descent_polish(c0, Y, weights, lr=1e-3, n_steps=2000):
    """Simple gradient descent with adaptive step size for final polish."""
    c = c0.copy()
    best_R = _compute_ratio(c, Y, weights)
    best_c = c.copy()
    
    for i in range(n_steps):
        R, g = _objective_and_grad(c, Y, weights)
        if not np.isfinite(R):
            break
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
        step = lr / g_norm
        c_new = c - step * g
        R_new = _compute_ratio(c_new, Y, weights)
        if np.isfinite(R_new) and R_new < best_R:
            best_R = R_new
            best_c = c_new.copy()
            c = c_new
        else:
            lr *= 0.5
            if lr < 1e-16:
                break
    
    return best_c, best_R


def _get_eigenvector_start(Y, weights):
    """
    Get a start from the eigenvector of the quadratic form associated with the L4/L2 ratio.
    For a fixed L2 norm, minimizing L4 is related to the smallest eigenvalue of a matrix
    that encodes the quartic interaction. We approximate by using the mode that most
    evenly distributes energy.
    """
    # Build the matrix M where M_{mn} = sum w Y_m Y_n Y_p Y_q ... 
    # Actually, let's use a simpler approach: the vector that minimizes max|f|/mean|f|
    # is approximately the one with equal weights across all modes.
    # A better start: use the eigenvector of the "energy distribution" matrix.
    
    # Compute f for each basis element and look at their L4/L2 ratios
    n_modes = Y.shape[0]
    ratios = np.zeros(n_modes)
    for i in range(n_modes):
        c = np.zeros(n_modes)
        c[i] = 1.0
        ratios[i] = _compute_ratio(c, Y, weights)
    
    # The mode with the lowest ratio is the flattest single mode
    best_mode = np.argmin(ratios)
    
    # Try combinations of the best few modes
    start = np.zeros(n_modes)
    # Use the 3 best modes with equal weight
    top3 = np.argsort(ratios)[:3]
    for idx in top3:
        start[idx] = 1.0
    
    return start / np.linalg.norm(start)


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic that minimize ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    k = 10
    Y, weights = _build_basis(k)
    n_modes = 2 * k + 1
    
    # Objective helpers
    def obj(c):
        return _compute_ratio(c, Y, weights)
    
    def obj_grad(c):
        return _objective_and_grad(c, Y, weights)
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    starts = []
    
    # 1. All ones (normalized)
    c = np.ones(n_modes)
    starts.append(_normalize_unit_l2(c, Y, weights))
    
    # 2. Single modes (normalized)
    for m in range(-k, k + 1):
        c = np.zeros(n_modes)
        c[k + m] = 1.0
        starts.append(_normalize_unit_l2(c, Y, weights))
    
    # 3. Symmetric pairs
    for m in range(1, k + 1):
        c = np.zeros(n_modes)
        c[k + m] = 1.0
        c[k - m] = 1.0
        starts.append(_normalize_unit_l2(c, Y, weights))
    
    # 4. Alternating even m
    c = np.zeros(n_modes)
    for m in range(0, k + 1, 2):
        c[k + m] = 1.0
        if m > 0:
            c[k - m] = 1.0
    starts.append(_normalize_unit_l2(c, Y, weights))
    
    # 5. Random starts
    for _ in range(25):
        c = rng.normal(size=n_modes)
        starts.append(_normalize_unit_l2(c, Y, weights))
    
    # 6. Low-m bias
    for _ in range(15):
        c = rng.normal(size=n_modes)
        for i in range(n_modes):
            m = i - k
            c[i] *= np.exp(-0.15 * abs(m))
        starts.append(_normalize_unit_l2(c, Y, weights))
    
    # 7. High-m bias (sometimes helps)
    for _ in range(5):
        c = rng.normal(size=n_modes)
        for i in range(n_modes):
            m = i - k
            c[i] *= np.exp(0.1 * abs(m))
        starts.append(_normalize_unit_l2(c, Y, weights))
    
    # 8. Eigenvector-based start
    starts.append(_get_eigenvector_start(Y, weights))
    
    # 9. Sparse starts (few modes active)
    for _ in range(10):
        c = np.zeros(n_modes)
        n_active = rng.integers(2, 6)
        active = rng.choice(n_modes, size=n_active, replace=False)
        c[active] = rng.choice([-1, 1], size=n_active)
        starts.append(_normalize_unit_l2(c, Y, weights))
    
    # Evaluate all starts
    best_R = np.inf
    best_c = None
    start_scores = []
    
    for c0 in starts:
        if time.time() > deadline:
            break
        try:
            R = obj(c0)
            if np.isfinite(R):
                start_scores.append((R, c0.copy()))
                if R < best_R:
                    best_R = R
                    best_c = c0.copy()
        except Exception:
            pass
    
    if best_c is None:
        return np.ones(n_modes)
    
    # Sort and take top starts for optimization
    start_scores.sort(key=lambda x: x[0])
    top_starts = [c for _, c in start_scores[:20]]
    
    if not any(np.allclose(c, best_c, atol=1e-10) for c in top_starts):
        top_starts.append(best_c.copy())
    
    # Optimize from each top start
    for c0 in top_starts:
        if time.time() > deadline - 60:
            break
        
        # Phase 1: L-BFGS-B
        try:
            c_opt, R_opt = _lgbfsb_optimize(c0, Y, weights, maxiter=2000, ftol=1e-15, gtol=1e-14)
            if np.isfinite(R_opt) and R_opt < best_R:
                best_R = R_opt
                best_c = c_opt.copy()
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead from the L-BFGS-B result (robust local search)
        if time.time() < deadline - 40:
            try:
                c_nm, R_nm = _nelder_mead_minimize(c0, Y, weights, maxiter=2000, xatol=1e-13, fatol=1e-13)
                if np.isfinite(R_nm) and R_nm < best_R:
                    best_R = R_nm
                    best_c = c_nm.copy()
            except Exception:
                pass
    
    # Final polish: gradient descent with adaptive step
    if time.time() < deadline - 20:
        try:
            c_polish, R_polish = _gradient_descent_polish(best_c, Y, weights, lr=1e-2, n_steps=3000)
            if np.isfinite(R_polish) and R_polish < best_R:
                best_R = R_polish
                best_c = c_polish.copy()
        except Exception:
            pass
    
    # Ultra-tight final L-BFGS-B
    if time.time() < deadline - 10:
        try:
            c_final, R_final = _lgbfsb_optimize(best_c, Y, weights, maxiter=5000, ftol=1e-16, gtol=1e-15)
            if np.isfinite(R_final) and R_final < best_R:
                best_R = R_final
                best_c = c_final.copy()
        except Exception:
            pass
    
    # Verify validity
    if len(best_c) != n_modes:
        return np.ones(n_modes)
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_modes)
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1.0 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1.0 / max_abs)
        
    return best_c


# EVOLVE-BLOCK-END
