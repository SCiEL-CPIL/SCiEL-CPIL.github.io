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
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)

    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(dist_upper))

    # Gradient: dE/dx_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    dist_sq = dist ** 2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = weighted_sum - pts * row_sums[:, np.newaxis]

    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        return 0.0
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


def _rsgd_optimize(pts, time_limit=10.0, seed=0):
    """Riemannian Stochastic Gradient Descent with mini-batches of pairs.

    Uses a small random subset of pairwise interactions per step to get a cheap
    (noisy) tangent gradient, enabling many cheap escape steps between
    expensive full-gradient passes. Returns the best configuration seen.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    rng = np.random.RandomState(seed)

    # Precompute the full upper-triangle pair list for mini-batch sampling.
    iu = np.triu_indices(n, k=1)
    all_i = iu[0]
    all_j = iu[1]
    n_pairs = len(all_i)

    batch = max(64, min(600, n_pairs // 8))
    lr = 0.015
    lr_decay = 0.997
    lr_min = 0.0005
    beta = 0.9  # momentum on the tangent direction

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)
    cur_energy = best_energy

    momentum = np.zeros_like(pts)
    step = 0
    no_improve = 0

    while time.time() - start_time < time_limit:
        # Sample a random mini-batch of pairs.
        if n_pairs > batch:
            sel = rng.choice(n_pairs, size=batch, replace=False)
            bi = all_i[sel]
            bj = all_j[sel]
        else:
            bi = all_i
            bj = all_j

        d = pts[bi] - pts[bj]
        dist = np.linalg.norm(d, axis=1)
        dist = np.maximum(dist, 1e-12)
        w = 1.0 / (dist * dist)
        # dE/dx_i += (x_i - x_j)/|..|^2 ; dE/dx_j += (x_j - x_i)/|..|^2
        g = np.zeros_like(pts)
        np.add.at(g, bi, d * w[:, None])
        np.add.at(g, bj, -d * w[:, None])

        g = g / batch  # scale to unbiased estimate of full gradient
        g_tan = _tangent_project(pts, g)

        momentum = beta * momentum + (1 - beta) * g_tan
        pts = pts - lr * momentum
        pts = _project_to_sphere(pts)

        # Occasionally recompute the true energy to track best.
        step += 1
        if step % 5 == 0 or step < 20:
            cur_energy = _energy_only(pts)
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 400:
                break

        lr = max(lr_min, lr * lr_decay)

    final_pts = _project_to_sphere(best_pts)
    final_energy, _ = _energy_and_grad(final_pts)
    return final_pts, final_energy


def _simulated_annealing(pts, time_limit=15.0, seed=42):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    max_steps = 5000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        noise_scale = T * 0.1
        perturbation = rng.randn(n, 3) * noise_scale
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only(new_pts)

        if new_energy < current_energy or rng.rand() < math.exp(-(new_energy - current_energy) / T):
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
            'gtol': 1e-13,
            'maxls': 100
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _exact_energy(pts):
    """Exact logarithmic energy via math.fsum for final reporting."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(np.dot(pts[i] - pts[j], pts[i] - pts[j]))
            if d > 1e-15:
                terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin from 150s limit

    # Generate diverse starting points (Fibonacci, icosahedron, random, rotated).
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_icosahedron_based(N))

    rng = np.random.RandomState(42)
    for _ in range(2):
        fib = _fibonacci_sphere(N)
        noise = 0.05 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))

    # A couple of pure-random and rotated starts for diversity.
    starts.append(_random_sphere(N, rng))
    starts.append(_so3_rotate(_fibonacci_sphere(N), rng))

    best_pts = starts[0].copy()
    best_energy, _ = _energy_and_grad(best_pts)

    # Phase 1: Optimize each starting point with L-BFGS-B.
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 45:
            break
        time_limit = min(6.0, deadline - elapsed - 35)
        if time_limit < 2.0:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 2: RSGD escape (cheap mini-batch) then re-polish with L-BFGS-B.
    for seed in [1, 2, 3]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 35:
            break
        time_limit = min(6.0, deadline - elapsed - 20)
        if time_limit < 2.0:
            break
        esc_pts, esc_energy = _rsgd_optimize(best_pts, time_limit=time_limit, seed=seed)
        # Re-polish the escaped configuration with a short L-BFGS pass.
        elapsed = time.time() - start_time
        if elapsed > deadline - 25:
            break
        tl = min(5.0, deadline - elapsed - 15)
        if tl < 1.5:
            break
        cand_pts, cand_energy = _lbfgs_optimize(esc_pts, time_limit=tl)
        if cand_energy < best_energy:
            best_pts = cand_pts
            best_energy = cand_energy

    # Phase 3: Adam refinement.
    elapsed = time.time() - start_time
    if elapsed < deadline - 25:
        time_limit = min(8.0, deadline - elapsed - 12)
        if time_limit >= 2.0:
            candidate_pts, candidate_energy = _adam_optimize(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 4: Simulated annealing with multiple seeds.
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        for seed in [42, 123, 789]:
            remaining = deadline - (time.time() - start_time) - 5
            if remaining < 3.0:
                break
            time_limit = min(5.0, remaining)
            candidate_pts, candidate_energy = _simulated_annealing(best_pts, time_limit=time_limit, seed=seed)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 5: L-BFGS-B re-polish.
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        time_limit = min(8.0, deadline - elapsed - 5)
        if time_limit >= 1.5:
            candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 6: Final high-precision polish.
    elapsed = time.time() - start_time
    if elapsed < deadline - 3:
        time_limit = min(5.0, deadline - elapsed - 1)
        if time_limit >= 1.0:
            candidate_pts, candidate_energy = _polish(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points.
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9 or not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
