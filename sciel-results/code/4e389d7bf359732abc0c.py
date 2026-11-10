import numpy as np
from scipy.special import lpmv, factorial, lpmv as legendre_associated
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses fixed-point iteration on the Euler-Lagrange condition plus gradient polish.
    """
    start_time = time.time()
    deadline = start_time + 560  # Safety margin

    k = 10
    n_m = 2 * k + 1

    # Setup quadrature grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # Construct basis functions
    A = np.zeros((n_points, n_m))

    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)

    # m = 0
    A[:, 10] = get_norm(0) * lpmv(0, 10, mu_flat)

    # m > 0: cos and sin modes
    for m in range(1, k + 1):
        P = lpmv(m, 10, mu_flat)
        N = np.sqrt(2.0) * get_norm(m)
        A[:, 10 + m] = N * P * np.cos(m * phi_flat)
        A[:, 10 - m] = N * P * np.sin(m * phi_flat)

    # Precompute weighted basis
    B = A * w_flat[:, None]

    def compute_R(c, A, w_flat, d_phi):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-14:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f = A @ u
        f2 = f * f
        f4 = f2 * f2
        S2 = d_phi * np.dot(w_flat, f2)
        S4 = d_phi * np.dot(w_flat, f4)
        if S2 < 1e-14:
            return 1e10, np.zeros(n_m)
        R = S4 ** 0.25 / np.sqrt(S2)
        # Gradient on sphere
        w_f3 = d_phi * (B.T @ (f ** 3))
        w_f = d_phi * (B.T @ f)
        # ∂R/∂u_j = S4^{-3/4} S2^{-1/2} w_f3[j] - S4^{1/4} S2^{-3/2} w_f[j]
        coef1 = S4 ** (-0.75) * S2 ** (-0.5)
        coef2 = S4 ** 0.25 * S2 ** (-1.5)
        grad_u = coef1 * w_f3 - coef2 * w_f
        u_dot_grad = np.dot(u, grad_u)
        grad_tangent = grad_u - u * u_dot_grad
        grad_c = grad_tangent / norm_c
        return R, grad_c

    def objective(c):
        R, g = compute_R(c, A, w_flat, d_phi)
        return R, g

    def sphere_polish(c, max_iter=800, initial_lr=0.05, tol=1e-12):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-14:
            return c
        u = c / norm_c
        lr = initial_lr
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            R_curr, grad_u = compute_R(u, A, w_flat, d_phi)
            grad_tangent = grad_u - np.dot(u, grad_u) * u
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < tol:
                break

            # Backtracking
            best_R = R_curr
            best_u = u
            step = lr
            for _ in range(30):
                if step < 1e-15:
                    break
                u_new = u - step * grad_tangent
                n_new = np.linalg.norm(u_new)
                if n_new < 1e-14:
                    step *= 0.5
                    continue
                u_new = u_new / n_new
                R_new, _ = compute_R(u_new, A, w_flat, d_phi)
                if R_new < best_R:
                    best_R = R_new
                    best_u = u_new
                step *= 0.7

            if best_R < R_curr:
                u = best_u
                lr = step / 0.7
                lr = min(lr, initial_lr * 3)
            else:
                lr *= 0.5
            if lr < 1e-12:
                break

        return u

    def fixed_point_iteration(u, max_iter=2000, omega=None):
        """
        Solve the Euler-Lagrange condition for L4/L2 minimization.
        At the optimum with L2 normalization, f should satisfy f ∝ |f|^3 pointwise.
        We iterate: u_new ∝ A^T (w * |A u|^3), then normalize.
        Use adaptive over-relaxation.
        """
        u = u.copy()
        if omega is None:
            omega = 1.0
        prev_diff = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            f = A @ u
            w_f3 = d_phi * (B.T @ (f ** 3))
            norm_w_f3 = np.linalg.norm(w_f3)
            if norm_w_f3 < 1e-14:
                break
            u_fp = w_f3 / norm_w_f3
            
            # Adaptive relaxation: increase omega if converging, decrease if oscillating
            if it > 10:
                cur_diff = np.linalg.norm(u_fp - u)
                if cur_diff < prev_diff * 0.8:
                    omega = min(omega * 1.05, 1.8)
                elif cur_diff > prev_diff * 1.2:
                    omega = max(omega * 0.9, 0.5)
                prev_diff = cur_diff
            
            u_new = (1 - omega) * u + omega * u_fp
            n_new = np.linalg.norm(u_new)
            if n_new < 1e-14:
                break
            u_new = u_new / n_new
            
            if np.linalg.norm(u_new - u) < 1e-12:
                u = u_new
                break
            u = u_new
        return u

    # Generate diverse initial candidates
    rng = np.random.default_rng(123)
    candidates = []

    # Pure modes
    for m in range(-k, k + 1):
        c = np.zeros(n_m)
        c[10 + m] = 1.0
        candidates.append(c)

    # Pairs with various amplitudes (even m for cos symmetry)
    for m1 in range(0, k + 1, 2):
        for m2 in range(m1 + 2, k + 1, 2):
            for amp in [0.3, 0.7, 1.0, 1.5]:
                c = np.zeros(n_m)
                c[10 + m1] = 1.0
                c[10 + m2] = amp
                c /= np.linalg.norm(c)
                candidates.append(c)

    # Even-m symmetric patterns (cosine modes only, symmetric in m -> -m)
    for pattern_seed in range(60):
        c = np.zeros(n_m)
        # Use only even m values for cos symmetry
        even_ms = list(range(0, k + 1, 2))
        n_nonzero = rng.integers(2, len(even_ms) + 1)
        indices_m = rng.choice(len(even_ms), size=n_nonzero, replace=False)
        for idx_m in indices_m:
            m_val = even_ms[idx_m]
            amp = rng.standard_normal()
            c[10 + m_val] = amp
            c[10 - m_val] = amp  # symmetric: same cos mode
        nrm = np.linalg.norm(c)
        if nrm > 1e-12:
            c /= nrm
            candidates.append(c)

    # Antisymmetric patterns (sin modes)
    for pattern_seed in range(30):
        c = np.zeros(n_m)
        odd_ms = list(range(1, k + 1, 2))
        n_nonzero = rng.integers(2, len(odd_ms) + 1)
        indices_m = rng.choice(len(odd_ms), size=n_nonzero, replace=False)
        for idx_m in indices_m:
            m_val = odd_ms[idx_m]
            amp = rng.standard_normal()
            c[10 + m_val] = amp
            c[10 - m_val] = -amp  # antisymmetric
        nrm = np.linalg.norm(c)
        if nrm > 1e-12:
            c /= nrm
            candidates.append(c)

    # Low-m emphasis
    for _ in range(50):
        c = np.zeros(n_m)
        n_nonzero = rng.integers(2, 10)
        probs = np.array([1.0 / (1 + abs(m)) for m in range(-k, k + 1)])
        probs /= probs.sum()
        indices = rng.choice(2 * k + 1, size=n_nonzero, replace=False, p=probs)
        for idx in indices:
            c[idx] = rng.standard_normal() * (1.0 / (1 + abs(idx - 10)))
        nrm = np.linalg.norm(c)
        if nrm > 1e-12:
            c /= nrm
            candidates.append(c)

    # All random
    for _ in range(100):
        c = rng.standard_normal(n_m)
        c /= np.linalg.norm(c)
        candidates.append(c)

    best_c = None
    best_val = np.inf

    # Strategy: Fixed point + Polish for each candidate
    for c_init in candidates:
        if time.time() > deadline - 40:
            break

        try:
            # 1. Fixed point iteration with adaptive relaxation
            u_fp = fixed_point_iteration(c_init, max_iter=1500)
            val_fp, _ = objective(u_fp)
            
            # 2. Polish with gradient
            u_polished = sphere_polish(u_fp, max_iter=600, initial_lr=0.05)
            val_polished, _ = objective(u_polished)
            
            if val_polished < best_val and np.all(np.isfinite(u_polished)):
                best_val = val_polished
                best_c = u_polished.copy()
                
            # Also try polishing the raw initial guess in case FP got stuck
            if time.time() < deadline - 15:
                u_grad = sphere_polish(c_init, max_iter=400, initial_lr=0.05)
                val_grad, _ = objective(u_grad)
                if val_grad < best_val and np.all(np.isfinite(u_grad)):
                    best_val = val_grad
                    best_c = u_grad.copy()

        except Exception:
            pass

    # Second round: fixed-point from best gradient-polished result
    if best_c is not None and time.time() < deadline - 20:
        u_fp2 = fixed_point_iteration(best_c, max_iter=2000)
        val_fp2, _ = objective(u_fp2)
        if val_fp2 < best_val:
            best_val = val_fp2
            best_c = u_fp2.copy()
        # Polish again
        polished2 = sphere_polish(best_c, max_iter=800, initial_lr=0.03)
        val_p2, _ = objective(polished2)
        if val_p2 < best_val:
            best_val = val_p2
            best_c = polished2.copy()

    # Final high-precision polish
    if best_c is not None and time.time() < deadline - 5:
        polished = sphere_polish(best_c, max_iter=1500, initial_lr=0.01, tol=1e-14)
        val_p, _ = objective(polished)
        if val_p < best_val:
            best_val = val_p
            best_c = polished.copy()

    # Validity check and scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0

    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c *= (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c *= (1e-2 / max_abs)

    return best_c
