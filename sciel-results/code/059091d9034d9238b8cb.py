# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_legendre

_DEADLINE_SECONDS = 620.0


def _build_harmonic_matrix():
    """Build the real orthonormal degree-10 spherical harmonic evaluation matrix."""
    k = 10
    Ntheta = 70
    Nphi = 130

    x, w = np.polynomial.legendre.leggauss(Ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi = 2.0 * np.pi / Nphi

    H = np.zeros((Ntheta, Nphi, 2 * k + 1), dtype=np.float64)

    P = np.zeros((Ntheta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = eval_legendre(k, m, x)

    def N_const(m):
        m = abs(int(m))
        lo = k - m + 1
        hi = k + m
        den = 1.0
        for j in range(lo, hi + 1):
            den *= j
        return np.sqrt((2 * k + 1) / (4.0 * np.pi) * (1.0 / den))

    for m in range(-k, k + 1):
        j = m + k
        if m == 0:
            H[:, :, j] = N_const(0) * P[:, 0][:, None] * np.ones((1, Nphi))
        elif m > 0:
            H[:, :, j] = np.sqrt(2.0) * N_const(m) * P[:, m][:, None] * np.cos(m * phi)[None, :]
        else:
            am = -m
            H[:, :, j] = np.sqrt(2.0) * N_const(am) * P[:, am][:, None] * np.sin(am * phi)[None, :]

    weights = np.outer(w, np.full(Nphi, dphi))
    return H, weights


def _ratio_from_coeffs(c, H_flat, w_flat):
    """Compute R = ||f||_4 / ||f||_2."""
    f = H_flat @ c
    f2 = f * f
    f4 = f2 * f2
    A = np.dot(f4, w_flat)
    B = np.dot(f2, w_flat)
    if A <= 0 or B <= 0:
        return np.inf
    return A ** 0.25 / np.sqrt(B)


def _variance_objective_and_grad(c, H_flat, w_flat):
    """
    Minimize Var(f^2) = E[f^4] - (E[f^2])^2 on the unit sphere.
    This is equivalent to minimizing L4/L2 ratio (up to monotone transform).
    Returns (obj, grad, hessian_diag_approx).
    """
    n = len(c)
    f = H_flat @ c
    f2 = f * f
    f4 = f2 * f2

    A = np.dot(f4, w_flat)  # E[f^4]
    B = np.dot(f2, w_flat)  # E[f^2]
    total_w = np.dot(w_flat, np.ones_like(w_flat))  # should be ~4*pi

    obj = A - B * B / total_w  # Var(f^2)

    # Gradient of A: dA/dc = 4 * H^T (f^3 * w)
    # Gradient of B: dB/dc = 2 * H^T (f * w)
    # d(obj)/dc = 4*H^T(f^3*w) - 2*B/total_w * 2*H^T(f*w)
    #           = 4*H^T(f^3*w) - 4*B/total_w * H^T(f*w)

    f3w = f * f2 * w_flat
    fw = f * w_flat

    grad_A = 4.0 * (H_flat.T @ f3w)
    grad_B = 2.0 * (H_flat.T @ fw)

    grad = grad_A - 2.0 * B / total_w * grad_B

    # Project gradient to tangent space of unit sphere: grad_tan = grad - (c . grad) * c
    c_dot_grad = np.dot(c, grad)
    grad_tan = grad - c_dot_grad * c

    return obj, grad_tan, f, f2, f4, A, B


def _hessian_at(c, H_flat, w_flat, f, f2, f4, A, B):
    """
    Approximate Hessian of Var(f^2) restricted to tangent space.
    Hessian of A = 4 * H^T diag(f^2 * w) H  (from second derivative of sum f^4 w)
    Actually: d2A/dc_i dc_j = 4 * sum_k f_k^2 w_k H_ki H_kj  (when f = Hc)
    Wait: A = sum f^4 w, dA/dc_i = 4 sum f^3 w H_i, d2A/dc_i dc_j = 4 sum (3 f^2 H_i H_j + f * 0) w
    Actually f depends on c: df/dc_i = H_i (the i-th column evaluated at all points)
    dA/dc_i = 4 sum f^3 w H_i
    d2A/dc_i dc_j = 4 sum (3 f^2 H_i H_j) w  -- no, let me redo.
    
    A = sum_k f_k^4 w_k where f_k = sum_j c_j H_kj
    dA/dc_i = 4 sum_k f_k^3 w_k H_ki
    d2A/dc_i dc_j = 4 sum_k (3 f_k^2 H_ki H_kj + f_k^3 * 0) w_k
                 = 12 sum_k f_k^2 H_ki H_kj w_k
                 = 12 * (H^T diag(f^2 * w) H)_{ij}
    
    B = sum f^2 w
    dB/dc_i = 2 sum f H_i w
    d2B/dc_i dc_j = 2 sum H_i H_j w = 2 * (H^T W H)_{ij}
    
    obj = A - B^2/total_w
    d2obj = 12 H^T diag(f^2 w) H - (2B/total_w) * 2 * H^T W H - (2/total_w)(dB/dc)(dB/dc)^T
    
    For trust-region on sphere, we need Hessian projected to tangent space.
    But computing full 21x21 Hessian is affordable.
    """
    n = len(c)
    N = H_flat.shape[0]

    # H^T diag(f^2 * w) H
    f2w = f2 * w_flat  # (N,)
    Ht_f2w = H_flat.T * f2w  # (N, n) -- actually H_flat.T @ (f2w * H_flat)
    # More efficiently: H^T (diag(f2w) H) = (H.T * f2w) @ H  -- no
    # H^T diag(f2w) H: first compute diag(f2w) @ H = H * f2w[:, None], then H.T @ that
    H_diag = H_flat * f2w[:, None]  # (N, n)
    Ht_H_f2w = H_flat.T @ H_diag  # (n, n) = H^T diag(f2w) H

    # H^T W H
    HW = H_flat * w_flat[:, None]
    Ht_HW = H_flat.T @ HW  # (n, n) = H^T W H

    # (dB/dc)(dB/dc)^T
    dB_dc = 2.0 * (H_flat.T @ (f * w_flat))  # (n,)
    outer_B = np.outer(dB_dc, dB_dc)

    total_w = np.dot(w_flat, w_flat)  # this is sum w_i^2, not what we want
    # Actually total_w should be sum of all weights = 4*pi
    # Let me recompute: the weights are the quadrature weights, sum(w) = 4*pi
    total_w = np.sum(w_flat)

    Hess = 12.0 * Ht_H_f2w - 4.0 * B / total_w * Ht_HW - 2.0 / total_w * outer_B

    # Project to tangent space: H_tan = (I - cc^T) H (I - cc^T)
    P = np.eye(n) - np.outer(c, c)
    Hess_tan = P @ Hess @ P

    return Hess_tan


def _trust_region_step(c, grad_tan, Hess_tan, delta, rng):
    """
    Solve the trust-region subproblem:
    min p(s) = grad_tan^T s + 0.5 s^T Hess_tan s  s.t. ||s|| <= delta
    Using Levenberg-Marquardt: (Hess_tan + mu * I) s = -grad_tan
    """
    n = len(c)
    I = np.eye(n)

    # Bisection on mu for LM
    mu_lo = 1e-10
    mu_hi = 1e10

    # First try mu=0 (Newton step)
    try:
        s = np.linalg.solve(Hess_tan + 1e-8 * I, -grad_tan)
        if np.linalg.norm(s) <= delta:
            return s, 0.0
    except np.linalg.LinAlgError:
        pass

    # Bisection
    for _ in range(60):
        mu = np.sqrt(mu_lo * mu_hi)
        try:
            s = np.linalg.solve(Hess_tan + mu * I, -grad_tan)
        except np.linalg.LinAlgError:
            mu_lo = mu
            continue

        s_norm = np.linalg.norm(s)
        if s_norm <= delta:
            return s, mu
        else:
            mu_lo = mu
        # Check if mu is large enough
        s_lm = -grad_tan / (mu + 1e-30)
        if np.linalg.norm(s_lm) <= delta:
            mu_hi = mu

    # Fallback: gradient descent direction
    g_norm = np.linalg.norm(grad_tan)
    if g_norm < 1e-15:
        return np.zeros(n), mu_hi
    s = -grad_tan / g_norm * min(delta, g_norm)
    return s, mu_hi


def _make_initializations(k, rng, n):
    """Generate diverse starting points on the unit sphere."""
    inits = []

    # Single modes
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v.copy())

    # Pairs (cos + sin for same |m|)
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0 / np.sqrt(2)
        v[-m + k] = 1.0 / np.sqrt(2)
        inits.append(v.copy())

    # All ones normalized
    v = np.ones(n)
    v /= np.linalg.norm(v)
    inits.append(v)

    # Even m only
    v = np.zeros(n)
    for m in range(-k, k + 1, 2):
        v[m + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v)

    # Odd m only
    v = np.zeros(n)
    for m in range(-k + 1, k + 1, 2):
        v[m + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v)

    # m=0 + high m
    for m in [4, 6, 8, 10]:
        v = np.zeros(n)
        v[0 + k] = 1.0
        v[m + k] = 1.0
        v[-m + k] = 1.0
        v /= np.linalg.norm(v)
        inits.append(v)

    # Random
    for _ in range(40):
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)
        inits.append(v)

    # Sparse random
    for _ in range(20):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 7), replace=False)
        v[idx] = rng.normal(size=len(idx))
        v /= np.linalg.norm(v)
        inits.append(v)

    return inits


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) of a degree-10 real spherical
    harmonic f = sum_m c_m Y_m that MINIMIZES the norm ratio
        R = ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}
    Uses trust-region (Levenberg-Marquardt) on the unit sphere with
    Var(f^2) as the smooth surrogate objective.
    """
    start_time = time.time()
    k = 10
    n = 2 * k + 1

    H, weights = _build_harmonic_matrix()
    H_flat = H.reshape(-1, n)
    w_flat = weights.reshape(-1)
    total_w = np.sum(w_flat)

    rng = np.random.default_rng(42)

    def remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    inits = _make_initializations(k, rng, n)

    best_c = np.ones(n, dtype=np.float64) / np.sqrt(n)
    best_R = _ratio_from_coeffs(best_c, H_flat, w_flat)
    best_obj = np.inf

    # Trust-region parameters
    delta_init = 1.0  # initial trust region radius
    delta_max = 5.0
    delta_min = 1e-8
    eta = 0.1  # acceptance threshold for actual/predicted ratio

    for init_idx, c0 in enumerate(inits):
        if remaining() < 15.0:
            break

        c = c0.copy()
        # Ensure unit norm
        cn = np.linalg.norm(c)
        if cn > 1e-15:
            c = c / cn

        delta = delta_init
        n_trust_iters = 0
        max_trust_iters = 80

        for tr_iter in range(max_trust_iters):
            if remaining() < 5.0:
                break
            n_trust_iters += 1

            obj, grad_tan, f, f2, f4, A, B = _variance_objective_and_grad(c, H_flat, w_flat)

            # Track best by R (the actual metric)
            R = _ratio_from_coeffs(c, H_flat, w_flat)
            if R < best_R:
                best_R = R
                best_c = c.copy()
                best_obj = obj

            # Convergence check
            g_norm = np.linalg.norm(grad_tan)
            if g_norm < 1e-13:
                break

            # Compute Hessian (projected to tangent space)
            Hess_tan = _hessian_at(c, H_flat, w_flat, f, f2, f4, A, B)

            # Solve trust-region subproblem
            s, mu = _trust_region_step(c, grad_tan, Hess_tan, delta, rng)

            # Ensure step is in tangent space
            s = s - np.dot(c, s) * c

            # Proposed point: c + s, then project back to unit sphere
            c_trial = c + s
            cn_trial = np.linalg.norm(c_trial)
            if cn_trial < 1e-15:
                break
            c_trial = c_trial / cn_trial

            # Evaluate at trial point
            obj_trial, grad_trial, f_t, f2_t, f4_t, A_t, B_t = _variance_objective_and_grad(
                c_trial, H_flat, w_flat
            )
            R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)

            # Actual reduction
            actual = obj - obj_trial

            # Predicted reduction from quadratic model
            predicted = -np.dot(grad_tan, s) - 0.5 * np.dot(s, Hess_tan @ s)

            # Acceptance test
            if actual > eta * max(predicted, 1e-30) and actual > 1e-15:
                # Accept
                c = c_trial
                delta = min(delta_max, 2.0 * delta)
            else:
                # Reject, shrink trust region
                delta *= 0.5
                if delta < delta_min:
                    break

            # If R improved, update best
            if R_trial < best_R:
                best_R = R_trial
                best_c = c_trial.copy()

        # If trust region converged, do a few more GD steps for polishing
        if remaining() > 10.0:
            c = best_c.copy()
            for _ in range(50):
                if remaining() < 5.0:
                    break
                obj, grad_tan, f, f2, f4, A, B = _variance_objective_and_grad(c, H_flat, w_flat)
                g_norm = np.linalg.norm(grad_tan)
                if g_norm < 1e-14:
                    break
                # Backtracking GD
                step_size = 0.1 / (g_norm + 1e-10)
                improved = False
                for bt in range(20):
                    c_t = c - step_size * grad_tan
                    c_t = c_t / (np.linalg.norm(c_t) + 1e-15)
                    R_t = _ratio_from_coeffs(c_t, H_flat, w_flat)
                    R_cur = _ratio_from_coeffs(c, H_flat, w_flat)
                    if R_t < R_cur:
                        c = c_t
                        improved = True
                        break
                    step_size *= 0.5
                if not improved:
                    break
                R_final = _ratio_from_coeffs(c, H_flat, w_flat)
                if R_final < best_R:
                    best_R = R_final
                    best_c = c.copy()

    # Final polishing with LM from best
    if remaining() > 30.0:
        c = best_c.copy()
        cn = np.linalg.norm(c)
        if cn > 1e-15:
            c = c / cn
        delta = 0.5
        for _ in range(100):
            if remaining() < 10.0:
                break
            obj, grad_tan, f, f2, f4, A, B = _variance_objective_and_grad(c, H_flat, w_flat)
            R = _ratio_from_coeffs(c, H_flat, w_flat)
            if R < best_R:
                best_R = R
                best_c = c.copy()

            g_norm = np.linalg.norm(grad_tan)
            if g_norm < 1e-14:
                break

            Hess_tan = _hessian_at(c, H_flat, w_flat, f, f2, f4, A, B)
            s, mu = _trust_region_step(c, grad_tan, Hess_tan, delta, rng)
            s = s - np.dot(c, s) * c

            c_trial = c + s
            cn_t = np.linalg.norm(c_trial)
            if cn_t < 1e-15:
                break
            c_trial = c_trial / cn_t

            obj_trial, _, _, _, _, _, _ = _variance_objective_and_grad(c_trial, H_flat, w_flat)
            actual = obj - obj_trial
            predicted = -np.dot(grad_tan, s) - 0.5 * np.dot(s, Hess_tan @ s)

            if actual > eta * max(predicted, 1e-30) and actual > 1e-15:
                c = c_trial
                delta = min(delta_max, 2.0 * delta)
            else:
                delta *= 0.5
                if delta < delta_min:
                    break

            R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
            if R_trial < best_R:
                best_R = R_trial
                best_c = c_trial.copy()

    # Validate
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c


# EVOLVE-BLOCK-END
