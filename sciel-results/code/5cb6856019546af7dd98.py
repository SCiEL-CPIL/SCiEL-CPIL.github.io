"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)

    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq[iu])
    dist = dist[dist > 1e-15]
    if len(dist) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(dist))

    # Gradient of log(1/|xi-xj|) = -log|xi-xj|
    # d/dxi [-log|xi-xj|] = (xi-xj)/|xi-xj|^2 for each j != i
    # Sum over all j: grad_i = sum_{j!=i} (xi-xj)/|xi-xj|^2
    # = sum_{j!=i} xi/|xi-xj|^2 - sum_{j!=i} xj/|xi-xj|^2
    # = xi * sum_{j!=i} 1/|xi-xj|^2 - sum_{j!=i} xj/|xi-xj|^2

    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum

    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq[iu])
    dist = dist[dist > 1e-15]
    if len(dist) == 0:
        return 0.0
    return -np.sum(np.log(dist))


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


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


def _random_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


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
        # Project gradient to tangent space of sphere
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
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
    """Adam optimizer on the sphere."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        m_tangent = m_hat - np.sum(m_hat * pts, axis=1, keepdims=True) * pts

        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)

        if t % 100 == 0 and np.linalg.norm(grad_tangent) < 1e-10:
            break

    return best_pts, best_energy


def _simulated_annealing(pts, time_limit=15.0, seed=42):
    """Simulated annealing to escape local minima with adaptive step size."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.005
    T_min = 1e-7
    cooling = 0.997
    max_steps = 8000
    base_noise = 0.05

    rng = np.random.RandomState(seed)
    accept_count = 0
    step_count = 0

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        step_count += 1
        # Adaptive noise: larger when accepting, smaller when rejecting
        noise_scale = base_noise * (T / 0.005) ** 0.5
        perturbation = rng.randn(n, 3) * noise_scale
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only(new_pts)

        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / T):
            pts = new_pts
            current_energy = new_energy
            accept_count += 1
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
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'maxfun': 20000,
            'ftol': 1e-16,
            'gtol': 1e-13,
            'maxls': 100
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _restart_optimize(pts, time_limit=20.0, n_restarts=5, seed=0):
    """Try multiple restarts from slightly perturbed versions of pts."""
    start_time = time.time()
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    rng = np.random.RandomState(seed)

    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break

        # Create a perturbed version with varying noise scale
        noise_scale = 0.03 + 0.02 * i
        noise = rng.randn(N, 3) * noise_scale
        perturbed = _project_to_sphere(pts + noise)

        # Quick L-BFGS
        remaining = time_limit - (time.time() - start_time) - 2
        if remaining < 3:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(
            perturbed, time_limit=min(remaining, 8.0), ftol=1e-13, gtol=1e-11
        )

        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts

    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = 145.0

    # Start from fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    # Phase 1: L-BFGS-B from fibonacci
    elapsed = time.time() - start_time
    if elapsed < deadline - 50:
        time_limit = min(15.0, deadline - elapsed - 35)
        candidate_pts, candidate_energy = _lbfgs_optimize(pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 2: Adam fine-tuning
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        time_limit = min(10.0, deadline - elapsed - 25)
        candidate_pts, candidate_energy = _adam_optimize(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 3: Simulated annealing to explore
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        time_limit = min(18.0, deadline - elapsed - 15)
        candidate_pts, candidate_energy = _simulated_annealing(
            best_pts, time_limit=time_limit, seed=42
        )
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 4: Restart optimization from best
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        time_limit = min(20.0, deadline - elapsed - 10)
        candidate_pts, candidate_energy = _restart_optimize(
            best_pts, time_limit=time_limit, n_restarts=5, seed=123
        )
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 5: L-BFGS-B re-polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 12:
        time_limit = min(10.0, deadline - elapsed - 8)
        candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 6: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 3:
        time_limit = min(8.0, deadline - elapsed - 1)
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
