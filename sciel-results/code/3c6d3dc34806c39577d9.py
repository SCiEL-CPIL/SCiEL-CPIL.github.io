"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.88
TIME_LIMIT = 150.0


def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient w.r.t. X (shape (N,3)).
    Uses vectorized operations for speed.
    """
    # Compute pairwise squared distances
    dots = X @ X.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)

    # Energy: -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij) = -0.5 * sum_{i!=j} 0.5*log(d2_ij)
    # = -0.25 * sum_{i!=j} log(d2_ij)
    # But we want sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # = -0.5 * sum_{i!=j} 0.5 * log(d2_ij) = -0.25 * sum_{i!=j} log(d2_ij)
    
    iu = np.triu_indices(N, 1)
    log_d2_upper = np.log(d2[iu])
    E = -0.5 * math.fsum(log_d2_upper.tolist())

    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_i - x_j) / d2_ij
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j inv_d2[i,j] * (x_i - x_j) = x_i * sum_j inv_d2[i,j] - sum_j inv_d2[i,j] * x_j
    row_sums = inv_d2.sum(axis=1)
    term2 = inv_d2 @ X
    grad = X * row_sums[:, None] - term2

    return E, grad


def _retract(X, step):
    """Riemannian retraction: move along tangent direction, then project back to sphere."""
    step_dot_X = np.sum(step * X, axis=1, keepdims=True)
    tangent_step = step - step_dot_X * X
    X_new = X + tangent_step
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_new = X_new / norms
    return X_new


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral with optional phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed_offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    theta = rng.uniform(0, 2 * np.pi, size=n)
    cos_phi = rng.uniform(-1, 1, size=n)
    sin_phi = np.sqrt(1 - cos_phi ** 2)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = cos_phi
    return np.column_stack([x, y, z])


def _scipy_lbfgs(X0, deadline, max_iter=8000):
    """
    Use scipy L-BFGS-B on the sphere with projection.
    Returns (best_points, best_energy).
    """
    X = X0.copy()
    x0 = X.ravel()

    def func(x):
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        E, _ = _energy_and_grad(p)
        return E

    def jac(x):
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        _, g = _energy_and_grad(p)
        return g.ravel()

    best_x = x0.copy()
    best_e = func(x0)

    # Run L-BFGS-B with tighter tolerances
    try:
        result = minimize(
            func, x0, jac=jac,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-16, 'maxfun': 20000, 'disp': False}
        )
        if result.fun < best_e:
            best_e = result.fun
            best_x = result.x
    except Exception:
        pass

    # If time remains, run another L-BFGS-B from the improved point
    if time.time() < deadline - 5.0:
        try:
            result2 = minimize(
                func, best_x, jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxfun': 10000, 'disp': False}
            )
            if result2.fun < best_e:
                best_e = result2.fun
                best_x = result2.x
        except Exception:
            pass

    X_best = best_x.reshape(N, 3)
    norms = np.linalg.norm(X_best, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_best = X_best / norms
    return X_best, best_e


def _polish_steepest(X0, deadline, max_iter=50000, tol=1e-14):
    """
    Steepest descent polish with backtracking on the sphere manifold.
    Uses conjugate gradient direction for faster convergence.
    """
    X = X0.copy()
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()

    step_size = 1e-2
    prev_g = g.copy()
    direction = None

    for it in range(max_iter):
        if time.time() > deadline:
            break

        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break

        # Polak-Ribiere beta for conjugate direction
        if direction is None:
            direction = -g
        else:
            g_dot_prev = np.dot(g.ravel(), prev_g.ravel())
            prev_g_norm_sq = np.dot(prev_g.ravel(), prev_g.ravel())
            if prev_g_norm_sq > 1e-30:
                beta = max(0.0, g_dot_prev / prev_g_norm_sq)
            else:
                beta = 0.0
            direction = -g + beta * direction

        # Project to tangent space
        dir_dot_X = np.sum(direction * X, axis=1, keepdims=True)
        direction = direction - dir_dot_X * X

        # Backtracking line search with Armijo condition
        accepted = False
        for ls in range(60):
            if time.time() > deadline:
                break
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)

            # Armijo condition: E_trial < E + c * step_size * grad^T * direction
            c = 1e-4
            grad_dir = np.sum(g * direction)
            if E_trial < E + c * step_size * grad_dir:
                X = X_trial
                E = E_trial
                g = g_trial
                accepted = True
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                step_size = min(step_size * 1.5, 1e-1)
                break
            step_size *= 0.5

        if not accepted:
            step_size *= 0.1
            if step_size < 1e-25:
                break
            direction = None
            continue

        prev_g = g.copy()

    return best_X, best_E


def _newton_polish(X0, deadline, max_iter=200, tol=1e-15):
    """
    Newton-Raphson style polish using approximate Hessian via finite differences.
    For the final few points of convergence.
    """
    X = X0.copy()
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()

    # Use a damped Newton step: solve (H + lambda*I) dx = -g in tangent space
    # Approximate Hessian via finite differences on a small subset
    eps = 1e-5

    for it in range(max_iter):
        if time.time() > deadline:
            break

        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break

        # Build approximate Hessian using random finite differences
        # Use only a few directions to keep cost manageable
        n_params = N * 3
        # Sample a few random tangent directions for Hessian approximation
        n_samples = 30
        rng = np.random.RandomState(42 + it)

        H_approx = np.zeros((n_params, n_params))
        for s in range(n_samples):
            # Random direction
            v = rng.normal(size=n_params)
            # Project to tangent space
            v_3d = v.reshape(N, 3)
            v_dot_X = np.sum(v_3d * X, axis=1, keepdims=True)
            v_tangent = (v_3d - v_dot_X * X).ravel()

            vn = np.linalg.norm(v_tangent)
            if vn < 1e-15:
                continue
            v_tangent = v_tangent / vn

            X_plus = _retract(X, eps * v_tangent.reshape(N, 3))
            _, g_plus = _energy_and_grad(X_plus)
            X_minus = _retract(X, -eps * v_tangent.reshape(N, 3))
            _, g_minus = _energy_and_grad(X_minus)

            # Hessian-vector product: (g_plus - g_minus) / (2*eps)
            Hv = (g_plus - g_minus).ravel() / (2.0 * eps)
            # Accumulate rank-1 update (not full Hessian, but helps with direction)
            H_approx += np.outer(v_tangent, Hv)

        # Damped Newton step: solve (H_approx + mu*I) dx = -g
        mu = 0.01
        A = H_approx + mu * np.eye(n_params)

        try:
            dx = np.linalg.solve(A, -g_flat)
        except np.linalg.LinAlgError:
            dx = -g_flat

        # Project to tangent space
        dx_3d = dx.reshape(N, 3)
        dx_dot_X = np.sum(dx_3d * X, axis=1, keepdims=True)
        dx_tangent = dx_3d - dx_dot_X * X

        # Line search
        step_size = 1.0
        accepted = False
        for ls in range(30):
            if time.time() > deadline:
                break
            X_trial = _retract(X, step_size * dx_tangent)
            E_trial, g_trial = _energy_and_grad(X_trial)

            if E_trial < E:
                X = X_trial
                E = E_trial
                g = g_trial
                accepted = True
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                break
            step_size *= 0.5

        if not accepted:
            break

    return best_X, best_E


def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION

    rng = np.random.RandomState(42)

    # Generate diverse starting points
    starts = []

    # Fibonacci with different phase offsets
    for offset in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        starts.append(_fibonacci_sphere(N, seed_offset=offset))

    # Random starts
    for _ in range(5):
        starts.append(_random_sphere(N, rng))

    # Perturbed Fibonacci
    for _ in range(3):
        base = _fibonacci_sphere(N)
        noise = rng.normal(0, 0.1, size=(N, 3))
        perturbed = base + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        starts.append(perturbed)

    # Optimize each start and keep the best
    best_pts = None
    best_energy = float('inf')
    n_starts = len(starts)

    for i, start in enumerate(starts):
        if time.time() > deadline - 10.0:
            break

        # Allocate time per start with buffer for final polish
        remaining = deadline - time.time() - 10.0
        starts_remaining = n_starts - i
        time_per_start = remaining / max(1, starts_remaining) * 0.75
        start_deadline = time.time() + max(2.0, time_per_start)

        # scipy L-BFGS-B optimization with more iterations
        pts, e = _scipy_lbfgs(start, start_deadline, max_iter=5000)

        # Steepest descent polish with tighter tolerance
        pts, e = _polish_steepest(pts, start_deadline, max_iter=30000, tol=1e-14)

        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()

    # Final Newton polish on the best result
    if best_pts is not None:
        remaining = deadline - time.time()
        if remaining > 10.0:
            newton_deadline = deadline - 5.0
            best_pts, best_energy = _newton_polish(best_pts, newton_deadline, max_iter=150, tol=1e-15)

        # Final high-precision steepest polish with very tight tolerance
        if time.time() < deadline - 2.0:
            best_pts, best_energy = _polish_steepest(best_pts, deadline - 1.0, max_iter=50000, tol=1e-16)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms

    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)

    # Check minimum distance
    min_d = 1e18
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-9:
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
