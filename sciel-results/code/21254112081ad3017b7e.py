# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10

    # Gauss-Legendre quadrature for cos(theta) in [-1, 1]
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)

    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Precompute P_10^m(cos(theta)) for m = 0..10 at all quadrature nodes
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)

    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)

    # Build basis matrix: B[m_idx, i, j] where m_idx=0..20 corresponds to m=-10..10
    B = np.zeros((21, n_theta, n_phi))

    for m_idx in range(21):
        m = m_idx - 10
        if m == 0:
            for j in range(n_phi):
                B[m_idx, :, j] = N[0] * P[0, :]
        elif m > 0:
            for j in range(n_phi):
                B[m_idx, :, j] = np.sqrt(2) * N[m] * P[m, :] * np.cos(m * phi[j])
        else:
            am = abs(m)
            for j in range(n_phi):
                B[m_idx, :, j] = np.sqrt(2) * N[am] * P[am, :] * np.sin(am * phi[j])

    # Quadrature weights
    W = np.outer(w, np.ones(n_phi)) * dphi

    return B, W, x, phi, n_theta, n_phi


def _evaluate_ratio(coeffs, B, W):
    """Compute ||f||_4 / ||f||_2 using precomputed basis and weights."""
    f = np.tensordot(coeffs, B, axes=(0, 0))
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)

    if norm2_sq <= 0:
        return np.inf

    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))

    return norm4 / norm2


def _objective_and_grad(c, B, W):
    """Return objective (||f||_4^4) and its gradient for given coefficients."""
    f = np.tensordot(c, B, axes=(0, 0))
    f4 = f**4
    obj = np.sum(W * f4)
    grad = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
    return obj, grad


def _objective_sphere(c, B, W):
    """Objective on the unit sphere: minimize ||f||_4^4 given ||c||=1."""
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return 1e10
    c_unit = c / c_norm
    f = np.tensordot(c_unit, B, axes=(0, 0))
    f4 = f**4
    return np.sum(W * f4)


def _objective_sphere_grad(c, B, W):
    """Objective and gradient on the unit sphere."""
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return 1e10, np.zeros_like(c)
    c_unit = c / c_norm
    f = np.tensordot(c_unit, B, axes=(0, 0))
    f4 = f**4
    obj = np.sum(W * f4)
    grad_unit = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
    grad = grad_unit - np.dot(grad_unit, c_unit) * c_unit
    return obj, grad


def _objective_sphere_hess(c, B, W):
    """Compute Hessian of the quartic objective projected to the tangent space.
    
    f = B^T c_unit where c_unit = c / ||c||
    obj = sum W f^4
    d_obj/d_c_unit = 4 sum W f^3 B_m  (vector g)
    d^2_obj/d_c_unit^2 = 12 * B * diag(W f^2) * B^T  (matrix H)
    
    For the constrained problem on the sphere, we project to tangent space:
    H_tangent = (I - c c^T) H (I - c c^T)
    """
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return np.eye(len(c))
    c_unit = c / c_norm
    f = np.tensordot(c_unit, B, axes=(0, 0))
    
    # H = 12 * B * diag(W * f^2) * B^T
    # Compute as: H[i,j] = 12 * sum_{a,b} B[i,a,b] * W[a,b] * f[a,b]^2 * B[j,a,b]
    wf2 = W * f * f  # shape (n_theta, n_phi)
    
    # H[i,j] = 12 * sum_{a,b} B[i,a,b] * wf2[a,b] * B[j,a,b]
    # = 12 * (B * wf2) contracted with B over spatial indices
    # Efficient: for each i, H[i,:] = 12 * sum_{a,b} wf2[a,b] * B[i,a,b] * B[j,a,b] for all j
    # = 12 * B_i_weighted @ B^T where B_i_weighted[a,b] = wf2[a,b] * B[i,a,b]
    
    # Use einsum for efficiency
    # H[i,j] = 12 * einsum('iab,ab,jab->ij', B, wf2, B)
    H = 12.0 * np.einsum('iab,ab,jab->ij', B, wf2, B)
    
    # Project to tangent space: H_tan = (I - cc^T) H (I - cc^T)
    # = H - c (c^T H) - H c (c^T) + c (c^T H c) (c^T)
    Hc = H @ c_unit
    cTH = c_unit @ H
    cTHc = c_unit @ Hc
    H_tan = H - np.outer(c_unit, Hc) - np.outer(cTH, c_unit) + cTHc * np.outer(c_unit, c_unit)
    
    return H_tan


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimize the norm ratio ||f||_4 / ||f||_2 on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin

    B, W, x, phi, n_theta, n_phi = _precompute()

    dim = 21

    def check_time():
        return time.time() < deadline - 10

    best_coeffs = None
    best_ratio = np.inf

    # Strategy: Use L-BFGS-B on the sphere with multiple restarts
    # Since the ratio is scale-invariant, we optimize on the unit sphere
    
    def make_objective(c):
        """Return objective value for L-BFGS-B (uses normalized c)."""
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return 1e10
        c_unit = c / c_norm
        f = np.tensordot(c_unit, B, axes=(0, 0))
        f4 = f**4
        return np.sum(W * f4)

    def make_grad(c):
        """Return gradient for L-BFGS-B."""
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return np.zeros(dim)
        c_unit = c / c_norm
        f = np.tensordot(c_unit, B, axes=(0, 0))
        grad_unit = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
        # Project to tangent space of sphere
        grad = grad_unit - np.dot(grad_unit, c_unit) * c_unit
        return grad

    # Generate diverse initializations
    np.random.seed(12345)
    initializations = []

    # 1. All ones
    initializations.append(np.ones(dim) / np.sqrt(dim))

    # 2. Only m=0
    c0 = np.zeros(dim)
    c0[10] = 1.0
    initializations.append(c0.copy())

    # 3. Random unit vectors
    for trial in range(30):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        initializations.append(c)

    # 4. Alternating signs
    c_alt = np.ones(dim)
    c_alt[::2] = -1.0
    initializations.append(c_alt / np.linalg.norm(c_alt))

    # 5. Reduced m=0
    c_low_m0 = np.ones(dim)
    c_low_m0[10] = 0.1
    initializations.append(c_low_m0 / np.linalg.norm(c_low_m0))

    # 6. Pure high-m modes
    for m_val in [5, 8, 10]:
        for sign in [1, -1]:
            c = np.zeros(dim)
            idx_pos = m_val + 10
            idx_neg = -m_val + 10
            c[idx_pos] = sign
            c[idx_neg] = sign
            initializations.append(c / np.linalg.norm(c))

    # 7. Mix of modes with random weights
    for trial in range(20):
        c = np.random.randn(dim)
        c[10] *= 0.3  # downweight m=0
        c /= np.linalg.norm(c)
        initializations.append(c)

    # 8. Pure single-m modes (both cos and sin)
    for m_val in range(1, 11):
        c_cos = np.zeros(dim)
        c_cos[m_val + 10] = 1.0
        initializations.append(c_cos.copy())
        c_sin = np.zeros(dim)
        c_sin[-m_val + 10] = 1.0
        initializations.append(c_sin.copy())

    # 9. Symmetric cos-only combinations (even m)
    for m_val in [2, 4, 6, 8, 10]:
        c = np.zeros(dim)
        c[m_val + 10] = 1.0
        c[-m_val + 10] = 1.0
        initializations.append(c / np.linalg.norm(c))
        c2 = np.zeros(dim)
        c2[m_val + 10] = 1.0
        c2[-m_val + 10] = -1.0
        initializations.append(c2 / np.linalg.norm(c2))

    # Phase 1: L-BFGS-B from all initializations (fast, good for coarse convergence)
    for i, c_init in enumerate(initializations):
        if not check_time():
            break

        try:
            result = minimize(
                make_objective,
                c_init,
                jac=make_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )

            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm < 1e-15:
                continue
            c_opt = c_opt / c_norm

            ratio = _evaluate_ratio(c_opt, B, W)

            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = c_opt.copy()

        except Exception:
            continue

    # Phase 2: trust-constr with analytic Hessian from best L-BFGS-B result
    if best_coeffs is not None and check_time():
        try:
            # Use trust-constr with the sphere constraint ||c|| = 1
            # The constraint is g(c) = ||c||^2 - 1 = 0
            sphere_constraint = NonlinearConstraint(
                lambda c: np.dot(c, c) - 1.0,
                0.0, 0.0,
                jac=lambda c: 2.0 * c
            )
            
            result = minimize(
                make_objective,
                best_coeffs,
                jac=make_grad,
                hess=_objective_sphere_hess,
                constraints=[sphere_constraint],
                method='trust-constr',
                options={'maxiter': 300, 'gtol': 1e-10, 'xtol': 1e-12, 'disp': False}
            )

            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm >= 1e-15:
                c_opt = c_opt / c_norm
                ratio = _evaluate_ratio(c_opt, B, W)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 3: Additional random restarts with trust-constr if time permits
    if check_time():
        np.random.seed(99999)
        for trial in range(30):
            if not check_time():
                break
            c = np.random.randn(dim)
            c /= np.linalg.norm(c)
            try:
                # First do a quick L-BFGS-B
                result = minimize(
                    make_objective,
                    c,
                    jac=make_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                c_opt = result.x
                c_norm = np.linalg.norm(c_opt)
                if c_norm < 1e-15:
                    continue
                c_opt = c_opt / c_norm

                # Then polish with trust-constr
                if check_time():
                    sphere_constraint = NonlinearConstraint(
                        lambda c: np.dot(c, c) - 1.0,
                        0.0, 0.0,
                        jac=lambda c: 2.0 * c
                    )
                    result2 = minimize(
                        make_objective,
                        c_opt,
                        jac=make_grad,
                        hess=_objective_sphere_hess,
                        constraints=[sphere_constraint],
                        method='trust-constr',
                        options={'maxiter': 200, 'gtol': 1e-10, 'xtol': 1e-12, 'disp': False}
                    )
                    c_opt = result2.x
                    c_norm = np.linalg.norm(c_opt)
                    if c_norm < 1e-15:
                        continue
                    c_opt = c_opt / c_norm

                ratio = _evaluate_ratio(c_opt, B, W)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = c_opt.copy()
            except Exception:
                continue

    # Phase 4: Final high-precision polish on the best solution
    if best_coeffs is not None and check_time():
        try:
            sphere_constraint = NonlinearConstraint(
                lambda c: np.dot(c, c) - 1.0,
                0.0, 0.0,
                jac=lambda c: 2.0 * c
            )
            result = minimize(
                make_objective,
                best_coeffs,
                jac=make_grad,
                hess=_objective_sphere_hess,
                constraints=[sphere_constraint],
                method='trust-constr',
                options={'maxiter': 500, 'gtol': 1e-14, 'xtol': 1e-15, 'disp': False}
            )
            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm >= 1e-15:
                c_opt = c_opt / c_norm
                ratio = _evaluate_ratio(c_opt, B, W)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = c_opt.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.ones(dim) / np.sqrt(dim)

    # Ensure validity: max|c| in [1e-3, 1e5]
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1e-3 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)

    # Remove any NaN or inf
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(dim)

    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
