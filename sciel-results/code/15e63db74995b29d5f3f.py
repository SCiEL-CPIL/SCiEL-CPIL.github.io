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

    def sphere_polish(c, max_iter=1000, tol=1e-12):
        """
        Adam optimizer on the unit sphere.
        """
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-14:
            return c
        u = c / norm_c
        
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = 0.05
        
        m_v = np.zeros_like(u)
        v_v = np.zeros_like(u)
        
        best_R = np.inf
        best_u = u.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
                
            R_curr, grad_u = compute_R(u, A, w_flat, d_phi)
            grad_tangent = grad_u - np.dot(u, grad_u) * u
            
            # Update Adam moments
            m_v = beta1 * m_v + (1 - beta1) * grad_tangent
            v_v = beta2 * v_v + (1 - beta2) * (grad_tangent ** 2)
            
            m_hat = m_v / (1 - beta1 ** (it + 1))
            v_hat = v_v / (1 - beta2 ** (it + 1))
            
            # Adaptive step size
            step = lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Update and project to sphere
            u_new = u - step
            n_new = np.linalg.norm(u_new)
            if n_new < 1e-14:
                break
            u_new = u_new / n_new
            
            R_new, _ = compute_R(u_new, A, w_flat, d_phi)
            
            if R_new < best_R:
                best_R = R_new
                best_u = u_new.copy()
                
            # Early stopping if gradient is small
            if np.linalg.norm(grad_tangent) < tol:
                break
                
            u = u_new
            
        return best_u

    def fixed_point_iteration(u, max_iter=2000):
        """
        Heavy Ball method for the fixed point map T(u) = norm(A^T (w * |Au|^3)).
        """
        u = u.copy()
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            f_val = A @ u
            w_f3 = d_phi * (B.T @ (f_val ** 3))
            norm_w_f3 = np.linalg.norm(w_f3)
            if norm_w_f3 < 1e-14:
                break
            u_next = w_f3 / norm_w_f3
            
            # Check convergence
            diff = np.linalg.norm(u_next - u)
            if diff < 1e-12:
                return u_next
            
            # Heavy Ball update: u_new = u_next + alpha * (u_next - u)
            # alpha = 0.5 is a standard choice
            u = u_next + 0.5 * (u_next - u)
            u = u / np.linalg.norm(u)
            
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

    # Key structured candidates: m=0 and m=10 superposition (known to flatten)
    for ratio in [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        c = np.zeros(n_m)
        c[10] = 1.0  # m=0
        c[20] = ratio  # m=10
        c[0] = ratio  # m=-10
        c /= np.linalg.norm(c)
        candidates.append(c)
        
        # Also try with sin components
        c2 = np.zeros(n_m)
        c2[10] = 1.0
        c2[20] = ratio * 0.5
        c2[0] = ratio * 0.5
        c2[19] = ratio * 0.3  # m=9 cos
        c2[1] = ratio * 0.3   # m=-9 sin
        c2 /= np.linalg.norm(c2)
        candidates.append(c2)

    # Even-m symmetric patterns (cosine modes only, symmetric in m -> -m)
    for pattern_seed in range(80):
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
    for pattern_seed in range(40):
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
    for _ in range(60):
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
    for _ in range(120):
        c = rng.standard_normal(n_m)
        c /= np.linalg.norm(c)
        candidates.append(c)

    best_c = None
    best_val = np.inf

    # Strategy: Fixed point + Polish for each candidate
    # Process candidates in batches, prioritizing structured ones first
    n_candidates = len(candidates)
    for i, c_init in enumerate(candidates):
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

    # Final high-precision polish with very tight tolerance
    if best_c is not None and time.time() < deadline - 5:
        polished = sphere_polish(best_c, max_iter=2000, initial_lr=0.01, tol=1e-15)
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
