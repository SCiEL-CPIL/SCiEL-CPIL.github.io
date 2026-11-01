"""
Structural redesign for 100-point logarithmic energy on the unit sphere.

Instead of many short restarts + perturbations, this uses a single long
optimization trajectory driven by:
  1. Simulated annealing (accepts uphill moves -> escapes local minima).
  2. Periodic Delaunay-neighbor Laplacian smoothing (a low-pass geometric
     filter that enforces global consistency and smooths clustering).
  3. Riemannian gradient descent (cooling phase).
  4. L-BFGS-B polish for high-precision convergence.
"""
import time
import math
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 135.0  # safety margin before 150s kill


def _energy_fsum(points):
    """Exact scoring function: sum_{i<j} log(1/|x_i - x_j|) via math.fsum."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad_vec(points):
    """Fast vectorized energy + tangent gradient on the sphere."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))

    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    return energy, g_tangent


def _project(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _delaunay_smooth(points, strength=0.1):
    """Laplacian smoothing using Delaunay neighbors.

    A low-pass filter on the point distribution: each point is pulled toward
    the average of its Delaunay neighbors, enforcing global geometric
    consistency and smoothing out clustering. This can escape local minima
    that trap coordinate descent.
    """
    n = points.shape[0]
    try:
        tri = Delaunay(points)
        neighbors = np.zeros((n, n), dtype=bool)
        for simplex in tri.simplices:
            for a in simplex:
                for b in simplex:
                    if a != b:
                        neighbors[a, b] = True
    except Exception:
        # Fallback: use k-nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(points, k=10)
        for i in range(n):
            for j in idx[i]:
                if j != i:
                    neighbors[i, j] = True

    # Smooth: move each point toward neighbor average (tangent projection)
    new_points = points.copy()
    for i in range(n):
        nbrs = np.where(neighbors[i])[0]
        if len(nbrs) == 0:
            continue
        avg = points[nbrs].mean(axis=0)
        # Tangent direction
        d = avg - points[i]
        d = d - np.dot(d, points[i]) * points[i]
        new_points[i] = points[i] + strength * d
    return _project(new_points)


def _simulated_annealing(points, time_budget, T_start=0.05, T_end=1e-5,
                         base_lr=0.03):
    """Simulated annealing on the sphere.

    Accepts uphill moves with probability exp(-deltaE / T), allowing escape
    from local minima. Temperature cools geometrically over the budget.
    """
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    current = points.copy()
    current_energy, _ = _energy_grad_vec(current)

    total_steps_est = 20000
    cool_rate = (T_end / T_start) ** (1.0 / max(total_steps_est, 1))

    step = 0
    while time.time() - start_time < time_budget:
        T = T_start * (cool_rate ** step)
        energy, grad = _energy_grad_vec(current)

        # Adaptive learning rate based on temperature
        lr = base_lr * (0.5 + 0.5 * (T / T_start))

        # Riemannian step
        new_points = current - lr * grad
        new_points = _project(new_points)
        new_energy, _ = _energy_grad_vec(new_points)

        delta = new_energy - energy
        if delta < 0:
            accept = True
        else:
            # Metropolis acceptance
            if T > 1e-12:
                accept = np.random.random() < math.exp(-delta / T)
            else:
                accept = False

        if accept:
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current.copy()
        else:
            # On rejection, occasionally apply a small random kick
            if np.random.random() < 0.1:
                noise = np.random.randn(N, 3) * 0.02
                current = _project(current + noise)
                current_energy, _ = _energy_grad_vec(current)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current.copy()

        step += 1

    return best_points


def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-8,
                     patience=3000):
    """Riemannian gradient descent with backtracking line search."""
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0

    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_vec(best_points)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break

        alpha = lr
        improved = False
        for _ in range(20):
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            new_energy, _ = _energy_grad_vec(new_points)
            if new_energy < energy - 1e-12:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.95
            if no_improve > patience:
                break

    return best_points


def _polish_lbfgs(points, time_budget):
    """L-BFGS-B polish with tight tolerances."""
    start_time = time.time()

    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()

    x0 = points.flatten().copy()
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 20000, 'ftol': 1e-16,
                                'gtol': 1e-13, 'maxls': 100})
        if hasattr(res, 'x'):
            return _project(res.x.reshape(-1, 3))
    except Exception:
        pass
    return points


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)


def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)


def construct():
    start_time = time.time()

    def _time_left():
        return max(0.0, TIME_LIMIT - (time.time() - start_time))

    # Phase 1: Initialize with a well-distributed starting point
    # Use Fibonacci spiral (near-optimal for uniform distribution)
    points = _fibonacci_spiral(N, 0.5)
    best_points = points.copy()
    best_energy = _energy_fsum(best_points)

    # Phase 2: Quick gradient descent to get into a good basin
    if _time_left() > 5.0:
        budget = min(_time_left() * 0.1, 10.0)
        refined = _riemannian_flow(points, budget, initial_lr=0.05)
        e = _energy_fsum(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined

    # Phase 3: Simulated annealing (long horizon, escapes local minima)
    if _time_left() > 10.0:
        sa_budget = _time_left() * 0.5
        sa_result = _simulated_annealing(best_points, sa_budget,
                                         T_start=0.03, T_end=1e-6,
                                         base_lr=0.02)
        e_sa = _energy_fsum(sa_result)
        if e_sa < best_energy:
            best_energy = e_sa
            best_points = sa_result

    # Phase 4: Periodic Delaunay Laplacian smoothing + re-optimization
    if _time_left() > 10.0:
        for _ in range(3):
            if _time_left() < 5.0:
                break
            # Smooth the distribution
            smoothed = _delaunay_smooth(best_points, strength=0.08)
            # Re-optimize from smoothed position
            budget = min(_time_left() * 0.25, 20.0)
            refined = _riemannian_flow(smoothed, budget, initial_lr=0.03)
            e = _energy_fsum(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
            else:
                # Try with different smoothing strength
                smoothed2 = _delaunay_smooth(best_points, strength=0.15)
                refined2 = _riemannian_flow(smoothed2, min(budget, 10.0),
                                            initial_lr=0.03)
                e2 = _energy_fsum(refined2)
                if e2 < best_energy:
                    best_energy = e2
                    best_points = refined2

    # Phase 5: More simulated annealing from the current best
    if _time_left() > 15.0:
        sa_budget = _time_left() * 0.4
        sa_result = _simulated_annealing(best_points, sa_budget,
                                         T_start=0.01, T_end=1e-7,
                                         base_lr=0.01)
        e_sa = _energy_fsum(sa_result)
        if e_sa < best_energy:
            best_energy = e_sa
            best_points = sa_result

    # Phase 6: L-BFGS-B polish (multiple rounds for precision)
    if _time_left() > 5.0:
        polish_budget = min(_time_left() * 0.4, 30.0)
        polished = _polish_lbfgs(best_points, polish_budget)
        e = _energy_fsum(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished

    # Phase 7: Final fine-tuning with small perturbations + polish
    if _time_left() > 5.0:
        for kick_scale in [0.005, 0.01, 0.02]:
            if _time_left() < 3.0:
                break
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            noise = rng.randn(N, 3) * kick_scale
            kicked = _project(best_points + noise)
            budget = min(_time_left() * 0.3, 15.0)
            refined = _riemannian_flow(kicked, budget, initial_lr=0.02)
            polished = _polish_lbfgs(refined, min(budget, 10.0))
            e = _energy_fsum(polished)
            if e < best_energy:
                best_energy = e
                best_points = polished

    # Phase 8: Final high-precision polish
    if _time_left() > 3.0:
        final_opt = _polish_lbfgs(best_points, min(_time_left() * 0.5, 20.0))
        e_final = _energy_fsum(final_opt)
        if e_final < best_energy:
            best_energy = e_final
            best_points = final_opt

    # Final projection and validation
    best_points = _project(best_points)

    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                perturb = rng.randn(3) * 1e-5
                best_points[i] = _project(best_points[i] + perturb)

    # Final energy check
    final_energy = _energy_fsum(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N, 0.5)

    return best_points
