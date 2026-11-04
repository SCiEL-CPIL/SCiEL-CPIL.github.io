"""Rewritten program: exact Hermite-Gaussian Fourier identities + analytic-gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Physicist's Hermite polynomial Hn(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 195  # wall-clock safety margin before evaluator kill at 210s

    def elapsed():
        return time.time() - start_time

    # ------------------------------------------------------------------
    # Position-space grid (used only for the norm/position Gram matrices).
    # ------------------------------------------------------------------
    N = 16384
    x = np.linspace(-22, 22, N)
    dx = x[1] - x[0]

    # Hermite polynomials at the grid
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2.0)

    # Effective basis vectors imposing g(0)=0:
    #   g = c0*H0 + c4*H4 + c8*H8 + c12*H12,  c12 = -(c0+12c4+1680c8)/40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # ------------------------------------------------------------------
    # Position Gram matrices (M = <g|g>, Mx = <g|x^2|g>)
    # ------------------------------------------------------------------
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            a = [v0, v4, v8][i]
            b = [v0, v4, v8][j]
            M[i, j] = np.sum(a * b) * dx
            Mx[i, j] = np.sum(x**2 * a * b) * dx
            M[j, i] = M[i, j]
            Mx[j, i] = Mx[i, j]

    # ------------------------------------------------------------------
    # Momentum-space Gram matrix (Mf = <Fg|xi^2|Fg>) computed EXACTLY.
    #   F[Hn(x) e^{-x^2/2}] = i^n Hn(xi) e^{-xi^2/2}
    #   => |Fg|^2 integrates to  <g|Hn Hm|g>  with e^{-xi^2} weight.
    # ------------------------------------------------------------------
    Mf = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            a = [H0, H4, H8][i]
            b = [H0, H4, H8][j]
            Mf[i, j] = np.sum(a * b * gauss**2) * dx
            Mf[j, i] = Mf[i, j]

    # ------------------------------------------------------------------
    # Objective and EXACT analytic gradient (Rayleigh quotient of two
    # positive-definite quadratic forms).
    # ------------------------------------------------------------------
    def compute_bound_from_coeffs(c):
        c = np.asarray(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2 = c @ Mx @ c
        f2 = c @ Mf @ c
        if (x2 < 0 or f2 < 0 or not np.isfinite(x2) or not np.isfinite(f2)):
            return np.inf
        return np.sqrt(x2 * f2) / norm2

    def compute_bound_from_spherical(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    def analytic_gradient_spherical(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        Mc = M @ c
        Mxc = Mx @ c
        Mfc = Mf @ c
        norm2 = c @ Mc
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.array([0.0, 0.0])
        x2 = c @ Mxc
        f2 = c @ Mfc
        if x2 < 0 or f2 < 0 or not np.isfinite(x2) or not np.isfinite(f2):
            return np.array([0.0, 0.0])
        # d/dc [ sqrt(x2*f2) / norm2 ]
        dfdc = (0.5 * np.sqrt(f2 / x2) * (2.0 * Mxc) / norm2
                - np.sqrt(x2 * f2) * (2.0 * Mc) / (norm2 * norm2))
        # d/dtheta, d/dphi of c
        dtheta = np.array([np.cos(theta) * np.cos(phi),
                           np.cos(theta) * np.sin(phi),
                           -np.sin(theta)])
        dphi = np.array([-np.sin(theta) * np.sin(phi),
                         np.sin(theta) * np.cos(phi),
                         0.0])
        return np.array([dfdc @ dtheta, dfdc @ dphi])

    # ------------------------------------------------------------------
    # Seed point
    # ------------------------------------------------------------------
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(np.clip(c8, -0.999999, 0.999999))
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta, best_phi = theta_init, phi_init

    # ------------------------------------------------------------------
    # Global survey: diverse + structured starting points
    # ------------------------------------------------------------------
    rng = np.random.RandomState(42)
    starts = [(theta_init, phi_init)]
    for _ in range(40):
        t = rng.uniform(0.005, np.pi - 0.005)
        p = rng.uniform(0.005, 2 * np.pi - 0.005)
        starts.append((t, p))
    for t in np.linspace(0.15, np.pi - 0.15, 20):
        for p in np.linspace(0.05, 2 * np.pi - 0.05, 20):
            starts.append((t, p))

    scored = []
    for t, p in starts:
        b = compute_bound_from_spherical(t, p)
        if np.isfinite(b):
            scored.append((b, t, p))
            if b < best_bound:
                best_bound = b
                best_theta, best_phi = t, p

    scored.sort(key=lambda z: z[0])
    top = [(t, p) for _, t, p in scored[:20]]
    if (best_theta, best_phi) not in top:
        top.append((best_theta, best_phi))

    # ------------------------------------------------------------------
    # Stage 1: Nelder-Mead local refinement from top starts
    # ------------------------------------------------------------------
    for t, p in top:
        if elapsed() >= deadline - 45:
            break
        try:
            res = minimize(
                compute_bound_from_spherical,
                x0=[t, p],
                method='Nelder-Mead',
                options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 15000, 'adaptive': True},
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stage 2: SLSQP with EXACT analytic gradient (fast, high precision)
    # ------------------------------------------------------------------
    if elapsed() < deadline - 20:
        try:
            res = minimize(
                compute_bound_from_spherical,
                x0=[best_theta, best_phi],
                jac=analytic_gradient_spherical,
                method='SLSQP',
                bounds=[(1e-4, np.pi - 1e-4), (1e-4, 2 * np.pi - 1e-4)],
                options={'maxiter': 4000, 'ftol': 1e-18, 'disp': False},
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stage 3: tight Nelder-Mead polish
    # ------------------------------------------------------------------
    if elapsed() < deadline - 8:
        try:
            res = minimize(
                compute_bound_from_spherical,
                x0=[best_theta, best_phi],
                method='Nelder-Mead',
                options={'xatol': 1e-16, 'fatol': 1e-19, 'maxiter': 60000, 'adaptive': True},
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stage 4: final analytic-gradient SLSQP polish (highest precision)
    # ------------------------------------------------------------------
    if elapsed() < deadline - 3:
        try:
            res = minimize(
                compute_bound_from_spherical,
                x0=[best_theta, best_phi],
                jac=analytic_gradient_spherical,
                method='SLSQP',
                bounds=[(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)],
                options={'maxiter': 6000, 'ftol': 1e-22, 'disp': False},
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Return Cartesian coefficients (degrees 0, 4, 8)
    # ------------------------------------------------------------------
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
