"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)

    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = np.maximum(dist_upper, 1e-15)
    energy = -np.sum(np.log(dist_upper))

    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = pts * row_sums[:, np.newaxis] - weighted_sum

    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = np.maximum(dist_upper, 1e-15)
    return -np.sum(np.log(dist_upper))


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts


def _fibonacci_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _icosahedron_based(n):
    """Generate points using icosahedron structure for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)

    # Use Fibonacci as base, but pull closest points to icosahedron vertices
    fib = _fibonacci_sphere(n)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _so3_rotate(pts, rng):
    """Apply random SO(3) rotation to points."""
    u = rng.standard_normal(3)
    v = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    v = v - np.dot(v, u) * u
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    R = np.column_stack([u, v, w])
    return pts @ R.T


def _exp_map(pts, tangent):
    """Exponential map on S^2: move points along tangent vectors."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    small = norms < 1e-12
    norms_safe = np.where(small, 1.0, norms)
    
    cos_n = np.cos(norms_safe)
    sin_n_over_n = np.sin(norms_safe) / norms_safe
    sin_n_over_n[small] = 1.0
    
    result = pts * cos_n + tangent * sin_n_over_n
    return _project_to_sphere(result)


def _riemannian_cg(pts0, time_limit=10.0, maxiter=5000):
    """Riemannian Conjugate Gradient with exponential map updates."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)

    energy, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy

    beta_method = 'PR'
    max_beta = 10.0

    p = _tangent_project(pts, grad)
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-12:
        return pts, energy

    alpha_prev = 0.01

    for k in range(maxiter):
        if time.time() - start_time >= time_limit:
            break

        e_k = energy
        g_k = grad
        p_tangent = _tangent_project(pts, p)

        # Backtracking line search
        alpha = alpha_prev
        rho = 0.5
        max_ls = 30
        found = False

        for ls_iter in range(max_ls):
            trial = _exp_map(pts, alpha * p_tangent)
            e_trial = _energy_only(trial)

            if e_trial < e_k:
                found = True
                break
            else:
                alpha *= rho
                if alpha < 1e-10:
                    break

        if not found:
            alpha = 1e-8
            trial = _exp_map(pts, alpha * p_tangent)
            e_trial = _energy_only(trial)
            if e_trial > e_k + 1e-10:
                break

        pts = trial
        energy = e_trial
        grad_new, _ = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        g_tangent = _tangent_project(pts, grad_new)
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < 1e-12:
            break

        if beta_method == 'PR':
            g_new_tangent = g_tangent
            beta = max(0.0, min(max_beta, np.sum(g_new_tangent * (g_new_tangent - g_tangent)) / max(g_norm * g_norm, 1e-12)))
        else:
            beta = max(0.0, min(max_beta, np.sum((g_new_tangent - g_tangent) * p_tangent) / max(np.sum(p_tangent * p_tangent), 1e-12)))

        p = -g_tangent + beta * p
        alpha_prev = alpha

    return best_pts, best_energy


def _lbfgs_optimize(pts, time_limit=5.0, ftol=1e-12, gtol=1e-10, maxiter=2000):
    """L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(p, g)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': maxiter,
            'maxfun': 5000,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _adam_optimize(pts, time_limit=10.0):
    """Adam optimizer on the sphere with adaptive learning rate."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    lr_initial = 0.02
    lr_min = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_decay = 0.995

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    lr = lr_initial

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    no_improve_count = 0
    patience = 500

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        grad_tangent = _tangent_project(pts, grad)
        m_tangent = _tangent_project(pts, m_hat)

        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)

        lr = max(lr_min, lr * lr_decay)

        if t % 100 == 0 and np.linalg.norm(grad_tangent) < 1e-10:
            break

    return best_pts, best_energy


def _pair_swap_perturb(pts, rng, scale=0.1):
    """Perturb by swapping nearby pairs and adding noise."""
    n = len(pts)
    new_pts = pts.copy()
    
    # Select random pairs to swap/blend
    for _ in range(3):
        i, j = rng.choice(n, size=2, replace=False)
        # Blend the two points
        t_val = rng.uniform(0.3, 0.7)
        new_i = t_val * pts[i] + (1 - t_val) * pts[j]
        new_j = (1 - t_val) * pts[i] + t_val * pts[j]
        new_pts[i] = new_i + rng.normal(0, scale * 0.1, 3)
        new_pts[j] = new_j + rng.normal(0, scale * 0.1, 3)
    
    # Add small global noise
    new_pts += rng.normal(0, scale * 0.05, (n, 3))
    return _project_to_sphere(new_pts)


def _simulated_annealing(pts, time_limit=15.0, seed=42):
    """Simulated annealing with pair-swap perturbations to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-8
    cooling = 0.998
    max_steps = 10000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        # Use pair-swap perturbation
        new_pts = _pair_swap_perturb(pts, rng, scale=T * 2.0)
        new_energy = _energy_only(new_pts)

        delta = new_energy - current_energy
        if delta < 0 or rng.rand() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _polish(pts, time_limit=5.0):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(p, g)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'maxfun': 10000,
            'ftol': 1e-15,
            'gtol': 1e-12,
            'maxls': 100
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _momentum_gradient_descent(pts0, max_time):
    """Optimize using projected gradient descent with momentum and adaptive step size."""
    start_time = time.time()

    pts = _project_to_sphere(pts0.copy())
    e, g = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e

    # Momentum buffer
    velocity = np.zeros_like(pts)
    momentum = 0.9

    # Adaptive step size
    step_size = 0.01

    for iteration in range(100000):
        if time.time() - start_time >= max_time:
            break

        # Compute tangent gradient
        g_tan = _tangent_project(pts, g)

        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-10:
            break

        # Momentum update
        velocity = momentum * velocity - step_size * g_tan

        # Trial step
        new_pts = pts + velocity
        new_pts = _project_to_sphere(new_pts)

        # Compute new energy
        new_e, new_g = _energy_and_grad(new_pts)

        # Backtracking line search if needed
        if new_e > e + 1e-8:
            # Reduce step and retry
            for _ in range(10):
                step_size *= 0.5
                velocity = momentum * velocity - step_size * g_tan
                new_pts = pts + velocity
                new_pts = _project_to_sphere(new_pts)
                new_e, new_g = _energy_and_grad(new_pts)
                if new_e <= e + 1e-8:
                    break
            else:
                # Too many failures, reduce step more aggressively
                step_size *= 0.1
                continue

        # Accept step
        pts = new_pts
        e = new_e
        g = new_g

        if e < best_e:
            best_e = e
            best_pts = pts.copy()

        # Gradually increase step size if making progress
        if iteration > 100 and step_size < 0.05:
            step_size *= 1.05

    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    rng = np.random.default_rng(42)

    # Generate diverse starting points
    starts = []

    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))

    # 2. Icosahedron-based
    starts.append(_icosahedron_based(N))

    # 3. Perturbed Fibonacci (multiple seeds)
    for seed in [1, 7, 13]:
        fib = _fibonacci_sphere(N, seed=seed)
        noise = 0.05 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))

    # 4. Random starts
    for _ in range(3):
        starts.append(_random_sphere(N, rng))

    best_pts = None
    best_energy = np.inf

    # Phase 1: Optimize each start with L-BFGS-B, keep best
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 25:
            break

        time_limit = min(8.0, deadline - elapsed - 20)
        if time_limit < 2.0:
            break

        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts

    # Phase 2: Riemannian CG on the best
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        time_limit = min(12.0, deadline - elapsed - 15)
        candidate_pts, candidate_energy = _riemannian_cg(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 3: Simulated annealing with pair-swap perturbations (multiple seeds)
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        for seed in [42, 123, 789, 555]:
            remaining = deadline - (time.time() - start_time) - 5
            if remaining < 2.0:
                break
            time_limit = min(5.0, remaining)
            candidate_pts, candidate_energy = _simulated_annealing(best_pts, time_limit=time_limit, seed=seed)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 4: Riemannian CG after annealing
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        time_limit = min(8.0, deadline - elapsed - 10)
        candidate_pts, candidate_energy = _riemannian_cg(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 5: L-BFGS-B re-polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 8:
        time_limit = min(6.0, deadline - elapsed - 4)
        candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 6: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        time_limit = min(4.0, deadline - elapsed - 1)
        candidate_pts, candidate_energy = _polish(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
