import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_and_grad(x_flat):
    """
    Calculate energy and gradient (projected to tangent space).
    """
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_proj = x / norms

    diff = x_proj[:, np.newaxis, :] - x_proj[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + 1e-30
    dist = np.sqrt(dist_sq)

    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = dist[mask]

    e = -math.fsum(np.log(d_pairs).tolist())

    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)

    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x_proj

    grad_raw = -x_proj * w_sum[:, np.newaxis] + w_x

    g_dot_x = np.sum(grad_raw * x_proj, axis=1, keepdims=True)
    grad_tangent = grad_raw - g_dot_x * x_proj

    return e, grad_tangent.flatten()

def _energy_only(x):
    """Fast energy computation for line search."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + 1e-30
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask])
    return -math.fsum(np.log(d_pairs).tolist())

def _grad_tangent(x):
    """Tangent-space gradient."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + 1e-30
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    g = -x * w_sum[:, np.newaxis] + w_x
    g_dot_x = np.sum(g * x, axis=1, keepdims=True)
    return g - g_dot_x * x

def _polish(x, max_iter=800, tol=1e-15):
    """
    Unified high-precision polish: gradient descent with backtracking line search,
    plus Newton-like step acceleration when gradient is small.
    """
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    curr_e = _energy_only(x)
    step = 0.02
    prev_grad_norm = np.inf

    for it in range(max_iter):
        g = _grad_tangent(x)
        g_norm = np.linalg.norm(g)

        if g_norm < tol:
            break

        # Newton-like acceleration: if gradient is decreasing rapidly,
        # try a larger step to accelerate convergence
        if prev_grad_norm > 0 and g_norm < 0.1 * prev_grad_norm:
            step = min(step * 1.5, 0.05)
        prev_grad_norm = g_norm

        alpha = step
        improved = False
        for _ in range(50):
            x_new = x - alpha * g
            norms_new = np.linalg.norm(x_new, axis=1, keepdims=True)
            norms_new = np.where(norms_new < 1e-12, 1.0, norms_new)
            x_new = x_new / norms_new

            new_e = _energy_only(x_new)
            if new_e < curr_e - 1e-7 * alpha * g_norm**2:
                x = x_new
                curr_e = new_e
                improved = True
                break
            alpha *= 0.5

        if not improved:
            break

        # Gradually decrease step for final convergence
        if it % 40 == 39 and step > 0.002:
            step *= 0.85

    return x

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    all_inits = []

    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    all_inits.append(x_fib)

    # 2. Fibonacci + small noise (multiple scales)
    for noise_scale in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        noise = rng.normal(0, noise_scale, (N, 3))
        x_fib_noise = x_fib + noise
        x_fib_noise /= np.linalg.norm(x_fib_noise, axis=1, keepdims=True)
        all_inits.append(x_fib_noise)

    # 3. Spherical design via evenly spaced latitudes + golden angle
    for offset in [0.0, 0.3, 0.7]:
        i2 = np.arange(N) + 0.5 + offset
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + offset * 0.1
        r2 = np.sqrt(1.0 - z2 * z2)
        x_sd = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        all_inits.append(x_sd)

    # 4. Antipodal pair initialization (75 pairs + 1 pole)
    # Place 75 points on hemisphere, then their antipodes
    i3 = np.arange(75) + 0.5
    z3 = 1.0 - 2.0 * i3 / 75
    z3 = z3 * 0.95  # keep in upper hemisphere
    phi3 = np.pi * (1.0 + np.sqrt(5.0)) * i3
    r3 = np.sqrt(1.0 - z3 * z3)
    x_hemi = np.column_stack([r3 * np.cos(phi3), r3 * np.sin(phi3), z3])
    x_anti = np.vstack([x_hemi, -x_hemi])
    # Add one more point near the pole to make 150
    x_anti = np.vstack([x_anti, np.array([0.0, 0.0, 1.0])])
    x_anti /= np.linalg.norm(x_anti, axis=1, keepdims=True)
    all_inits.append(x_anti)

    # 5. Random initializations
    num_randoms = 25
    for k in range(num_randoms):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        all_inits.append(x_rand)

    best_x = all_inits[0].copy()
    best_energy = _energy_and_grad(best_x.flatten())[0]

    for idx, x0 in enumerate(all_inits):
        if time_left() < 12.0:
            break

        try:
            result = minimize(
                _energy_and_grad,
                x0.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100, 'm': 40},
            )

            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)

            # Polish with gradient descent
            if time_left() > 8.0:
                x_opt = _polish(x_opt, max_iter=400, tol=1e-15)

            e_opt = _energy_and_grad(x_opt.flatten())[0]

            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()

        except Exception:
            continue

    # Final high-precision polish on the best solution
    if time_left() > 5.0:
        try:
            best_x = _polish(best_x, max_iter=800, tol=1e-15)
            final_e = _energy_and_grad(best_x.flatten())[0]
            if final_e < best_energy:
                best_energy = final_e
        except Exception:
            pass

    # Second polish pass for even tighter convergence
    if time_left() > 2.0:
        try:
            best_x = _polish(best_x, max_iter=500, tol=1e-16)
            final_e = _energy_and_grad(best_x.flatten())[0]
            if final_e < best_energy:
                best_energy = final_e
        except Exception:
            pass

    # Final verification: ensure all points are on unit sphere and finite
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms

    return best_x
