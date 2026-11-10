import numpy as np
from scipy.special import lpmv, factorial, lpmv as legendre_associated
from scipy.optimize import minimize
import time

# Precompute spectral tensor for L4 norm: T[a,b,c,d] = ∫ Y_a Y_b Y_c Y_d dΩ
# This allows computing ||f||_4^4 = sum_{abcd} T[a,b,c,d] c_a c_b c_c c_d
# without evaluating on a grid at each iteration.
def _precompute_spectral_tensors():
    """Precompute the 4-index tensor for L4 norm via grid quadrature."""
    k = 10
    n_m = 2 * k + 1
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # Build basis matrix A (n_points x n_m)
    A = np.zeros((n_points, n_m))

    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)

    A[:, 10] = get_norm(0) * lpmv(0, 10, mu_flat)
    for m in range(1, k + 1):
        P = lpmv(m, 10, mu_flat)
        N = np.sqrt(2.0) * get_norm(m)
        A[:, 10 + m] = N * P * np.cos(m * phi_flat)
        A[:, 10 - m] = N * P * np.sin(m * phi_flat)

    # Weighted basis: B = A * w_flat[:, None] * d_phi
    B = A * (w_flat * d_phi)[:, None]

    # Compute T[a,b,c,d] = sum_points B[p,a] * B[p,b] * B[p,c] * B[p,d]
    # Use einsum for efficiency: T = einsum('pa,pb,pc,pd->abcd', B, B, B, B)
    # This is 9100 * 21^4 ≈ 1.7e9 operations... too slow.
    # Instead, use a two-step: M1[p,a,b] = B[p,a]*B[p,b], then T[a,b,c,d] = sum_p M1[p,a,b]*M1[p,c,d]
    # M1 shape: (9100, 21, 21) = 4M elements, fine.
    # T = einsum('pab,pcd->abcd', M1, M1) -> 21^4 = 194481 elements, fine.
    M1 = np.einsum('pa,pb->pab', B, B)  # (n_points, 21, 21)
    T = np.einsum('pab,pcd->abcd', M1, M1)  # (21, 21, 21, 21)

    # Also precompute G[a,b] = sum_p B[p,a]*B[p,b] for L2 norm
    G = np.einsum('pa,pb->ab', B, B)  # (21, 21)

    return T, G, A, B, w_flat, d_phi, n_points

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

    # Precompute spectral tensors for fast L4/L2 evaluation in coefficient space
    # G[a,b] = integral Y_a Y_b dOmega (L2 Gram matrix)
    G = np.einsum('pa,pb->ab', B, B)
    # T[a,b,c,d] = integral Y_a Y_b Y_c Y_d dOmega (L4 tensor)
    # Compute via intermediate: M1[p,a,b] = B[p,a]*B[p,b]
    M1 = np.einsum('pa,pb->pab', B, B)
    T = np.einsum('pab,pcd->abcd', M1, M1)
    # Free memory
    del M1

    def compute_R(c, A, w_flat, d_phi):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-14:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        # Use spectral tensors: S2 = u^T G u, S4 = u^T T u (contracted)
        S2 = u @ G @ u
        # S4 = sum_abcd T[a,b,c,d] u_a u_b u_c u_d
        # Efficient: first compute v_b = sum_a T[a,b,0,0]... no, need full contraction
        # S4 = einsum('abcd,a,b,c,d->', T, u, u, u, u)
        S4 = np.einsum('abcd,a,b,c,d->', T, u, u, u, u)
        if S2 < 1e-14:
            return 1e10, np.zeros(n_m)
        R = S4 ** 0.25 / np.sqrt(S2)
        # Gradient: dS4/du_j = 4 * sum_abc T[a,b,c,j] u_a u_b u_c
        # dS2/du_j = 2 * (G @ u)_j
        # grad_u_j = (1/4) S4^{-3/4} S2^{-1/2} * dS4/du_j - S4^{1/4} S2^{-3/2} * dS2/du_j
        # = S4^{-3/4} S2^{-1/2} * sum_abc T[a,b,c,j] u_a u_b u_c - S4^{1/4} S2^{-3/2} * (G @ u)_j
        # Compute grad_S4[j] = sum_abc T[a,b,c,j] u_a u_b u_c
        # = einsum('abcd,a,b,c->d', T, u, u, u)
        grad_S4 = np.einsum('abcd,a,b,c->d', T, u, u, u)
        grad_S2 = G @ u
        coef1 = S4 ** (-0.75) * S2 ** (-0.5)
        coef2 = S4 ** 0.25 * S2 ** (-1.5)
        grad_u = coef1 * grad_S4 - coef2 * grad_S2
        u_dot_grad = np.dot(u, grad_u)
        grad_tangent = grad_u - u * u_dot_grad
        grad_c = grad_tangent / norm_c
        return R, grad_c

    def objective(c):
        R, g = compute_R(c, A, w_flat, d_phi)
        return R, g

    def sphere_polish(c, max_iter=800, initial_lr=0.05, tol=1e-13):
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
                lr = step / 0.7  # reset to a value that worked
                lr = min(lr, initial_lr * 2)
            else:
                lr *= 0.5
            if lr < 1e-10:
                break

        return u

    def fixed_point_iteration(u, max_iter=2000, omega=1.5):
        """
        Solve the Euler-Lagrange condition for L4/L2 minimization in spectral domain.
        Stationarity: einsum('abcd,a,b,c->d',T,u,u,u) ∝ G @ u
        For orthonormal basis G≈I, so u ∝ einsum('abcd,a,b,c->d',T,u,u,u)
        Use over-relaxation: u_new = (1-omega)*u + omega*u_fp
        """
        u = u.copy()
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Compute spectral fixed point: v_j = sum_abc T[a,b,c,j] u_a u_b u_c
            v = np.einsum('abcd,a,b,c->d', T, u, u, u)
            # Project onto constraint G@u = u (for orthonormal, just normalize)
            # More generally, solve: v = λ G u, i.e., u ∝ G^{-1} v
            # For well-conditioned G, just use v directly
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-14:
                break
            u_fp = v / norm_v

            # Over-relaxation
            u_new = (1 - omega) * u + omega * u_fp
            n_new = np.linalg.norm(u_new)
            if n_new < 1e-14:
                break
            u_new = u_new / n_new

            # Check convergence
            if np.linalg.norm(u_new - u) < 1e-10:
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

    # Pairs with various amplitudes
    for m1 in range(-k, k + 1, 2):
        for m2 in range(m1 + 1, k + 1, 2):
            for amp in [0.3, 0.7, 1.0]:
                c = np.zeros(n_m)
                c[10 + m1] = 1.0
                c[10 + m2] = amp
                c /= np.linalg.norm(c)
                candidates.append(c)

    # Symmetric patterns
    for pattern_seed in range(50):
        c = np.zeros(n_m)
        n_nonzero = rng.integers(3, 15)
        indices = rng.choice(2 * k + 1, size=n_nonzero, replace=False)
        for idx in indices:
            c[idx] = rng.standard_normal()
        # Enforce even/odd symmetry options
        if pattern_seed % 3 == 0:
            # Make symmetric: c[i] = c[19-i]
            c_sym = (c + c[::-1]) / 2
            c = c_sym
        elif pattern_seed % 3 == 1:
            # Antisymmetric
            c_sym = (c - c[::-1]) / 2
            c = c_sym
        nrm = np.linalg.norm(c)
        if nrm > 1e-12:
            c /= nrm
            candidates.append(c)

    # Low-m emphasis
    for _ in range(80):
        c = np.zeros(n_m)
        n_nonzero = rng.integers(2, 10)
        # Bias towards low m
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
    for _ in range(150):
        c = rng.standard_normal(n_m)
        c /= np.linalg.norm(c)
        candidates.append(c)

    best_c = None
    best_val = np.inf

    # Strategy: Fixed point + Polish for each candidate
    for c_init in candidates:
        if time.time() > deadline - 30:
            break

        try:
            # 1. Fixed point iteration
            u_fp = fixed_point_iteration(c_init, max_iter=1000, omega=1.2)
            val_fp, _ = objective(u_fp)
            
            # 2. Polish with gradient
            u_polished = sphere_polish(u_fp, max_iter=500, initial_lr=0.05)
            val_polished, _ = objective(u_polished)
            
            if val_polished < best_val and np.all(np.isfinite(u_polished)):
                best_val = val_polished
                best_c = u_polished.copy()
                
            # Also try polishing the raw initial guess in case FP got stuck
            if time.time() < deadline - 10:
                u_grad = sphere_polish(c_init, max_iter=300, initial_lr=0.05)
                val_grad, _ = objective(u_grad)
                if val_grad < best_val and np.all(np.isfinite(u_grad)):
                    best_val = val_grad
                    best_c = u_grad.copy()

        except Exception:
            pass

    # Final polish on the best found with tighter tolerance
    if best_c is not None and time.time() < deadline - 5:
        polished = sphere_polish(best_c, max_iter=2000, initial_lr=0.01, tol=1e-14)
        val_p, _ = objective(polished)
        if val_p < best_val:
            best_val = val_p
            best_c = polished.copy()
        # Second ultra-fine polish
        if time.time() < deadline - 3:
            polished2 = sphere_polish(best_c, max_iter=3000, initial_lr=0.005, tol=1e-15)
            val_p2, _ = objective(polished2)
            if val_p2 < best_val:
                best_val = val_p2
                best_c = polished2.copy()

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
