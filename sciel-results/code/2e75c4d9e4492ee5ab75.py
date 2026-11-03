"""Rewritten program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(dists_sq)
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    
    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # For each pair (i,j) with i<j, contribution to grad_i is (x_j - x_i)/d^2
    # and to grad_j is (x_i - x_j)/d^2 = -(x_j - x_i)/d^2
    
    weights = 1.0 / dists_sq  # shape (n_pairs,)
    
    grads = np.zeros((n, 3))
    np.add.at(grads, iu[0], weights[:, np.newaxis] * diff)
    np.add.at(grads, iu[1], -weights[:, np.newaxis] * diff)

    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    energy = -math.fsum(np.log(np.sqrt(dists_sq)).tolist())
    return energy, grads_tangent.flatten()


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        # Try again with a different vector
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _local_minimize(points, deadline, maxiter=500):
    """Run L-BFGS-B to find local minimum with gradient descent fallback."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


def _gradient_descent_sphere(points, deadline, lr=0.1, maxiter=200):
    """Simple gradient descent on the sphere with adaptive step size."""
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy

    for it in range(maxiter):
        if time.time() > deadline:
            break
        # Compute gradient
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)

        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current
        grads = weighted_sum - current * sum_weights[:, np.newaxis]
        dot_products = np.sum(grads * current, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current

        # Update
        new_points = current - lr * grads_tangent
        new_points = _project_to_sphere(new_points)
        new_energy = _log_energy(new_points)

        if new_energy < current_energy:
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
            lr *= 1.05
        else:
            lr *= 0.5
            if lr < 1e-8:
                break

    return best, best_energy


def _basin_hopping_single(init_points, deadline, rng, max_iter=200):
    """Single basin hopping trajectory with adaptive perturbation."""
    current_points, current_energy = _local_minimize(init_points, deadline)
    best_points = current_points.copy()
    best_energy = current_energy

    T = 0.1
    T_min = 1e-7
    cooling = 0.985

    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        if T < T_min:
            break

        # Adaptive perturbation: mix small and large moves
        if rng.rand() < 0.15:
            # Large structural move: reposition 10-20 points significantly
            n_perturb = rng.randint(10, 21)
            idxs = rng.choice(N, n_perturb, replace=False)
            step_scale = rng.uniform(0.3, 0.8)
        else:
            # Normal move: reposition 2-8 points moderately
            n_perturb = rng.randint(2, 9)
            idxs = rng.choice(N, n_perturb, replace=False)
            step_scale = rng.uniform(0.05, 0.3)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * step_scale
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=200)

        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy


def _multi_point_move(points, deadline, rng, n_moves=10, step=0.1):
    """Make multiple small moves and keep improvements."""
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy

    for _ in range(n_moves):
        if time.time() > deadline:
            break
        idx = rng.randint(N)
        tangent = _tangent_vector(current[idx], rng)
        new_point = current[idx] + step * tangent
        new_point = new_point / np.linalg.norm(new_point)
        new_points = current.copy()
        new_points[idx] = new_point
        new_energy = _log_energy(new_points)
        if new_energy < current_energy:
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()

    return best, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.RandomState(12345)

    # Generate diverse initializations
    inits = []

    # Fibonacci sphere
    inits.append(_fibonacci_sphere(N))

    # Random uniform points with different seeds
    for seed in range(10):
        r = np.random.RandomState(seed * 7 + 1)
        inits.append(_random_uniform_sphere(N, r))

    # Perturbed Fibonacci
    for k in range(5):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.2
        inits.append(_project_to_sphere(perturbed))

    # Spherical designs based initializations (approximate)
    # Use different rotation angles
    for angle in [0.1, 0.3, 0.5, 0.7, 1.0]:
        theta = angle
        rot_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        fib = _fibonacci_sphere(N)
        rotated = fib @ rot_matrix.T
        inits.append(_project_to_sphere(rotated))

    # Best-so-far tracking
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)

    # Phase 1: Local minimize all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        p_res, e_res = _local_minimize(init, deadline)
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Phase 2: Basin hopping from best few initializations
    # Collect top initializations
    scored_inits = []
    for idx, init in enumerate(inits):
        e = _log_energy(init)
        scored_inits.append((e, idx, init))
    scored_inits.sort()

    # Run basin hopping from the best 7 initializations with more iterations
    for i in range(min(7, len(scored_inits))):
        if time.time() > deadline - 20.0:
            break
        bh_rng = np.random.RandomState(i * 100 + 500)
        bh_points, bh_energy = _basin_hopping_single(
            scored_inits[i][2], deadline, bh_rng, max_iter=150
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Phase 3: Multi-point refinement from best
    if time.time() < deadline - 12.0:
        mp_rng = np.random.RandomState(999)
        mp_points, mp_energy = _multi_point_move(
            best_points, deadline, mp_rng, n_moves=80, step=0.03
        )
        if mp_energy < best_energy:
            best_energy = mp_energy
            best_points = mp_points.copy()
        
        # Additional local minimize after multi-point moves
        if time.time() < deadline - 5.0:
            mp2_points, mp2_energy = _local_minimize(mp_points, deadline, maxiter=300)
            if mp2_energy < best_energy:
                best_energy = mp2_energy
                best_points = mp2_points.copy()

    # Phase 4: Final high-precision local minimize
    if time.time() < deadline - 3.0:
        # First a medium-precision pass
        final_points, final_energy = _local_minimize(best_points, deadline, maxiter=500)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()
        
        # Then a high-precision polish with tighter tolerances
        if time.time() < deadline - 2.0:
            try:
                x0 = best_points.flatten()
                res = minimize(
                    _energy_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
                )
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
            except Exception:
                pass

    # Gradient descent polish for final refinement
    if time.time() < deadline - 1.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, deadline, lr=0.005, maxiter=200
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()

    best_points = _project_to_sphere(best_points)

    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
