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
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)

    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]

    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points

    grads = weighted_sum - points * sum_weights[:, np.newaxis]

    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    energy = _log_energy(points)
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


def _geodesic_sphere(n):
    """Generate points using subdivided icosahedron (geodesic dome) approach.
    This gives more uniform initial spacing than Fibonacci sphere."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Subdivide edges to get more points
    # For each edge, add midpoint projected to sphere
    vertices = ico.copy()
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist < 1.0:  # Adjacent vertices
                edges.append((i, j))
    
    # Keep subdividing until we have enough points
    max_divs = 4
    for _ in range(max_divs):
        if len(vertices) >= n * 2:
            break
        new_vertices = [vertices]
        for i, j in edges:
            v1, v2 = vertices[i], vertices[j]
            mid = (v1 + v2) / 2.0
            mid = mid / np.linalg.norm(mid)
            new_vertices.append([mid])
            # Add new edges from midpoint to i and j
            edges.append((len(vertices), len(vertices)))  # placeholder, will fix
        vertices = np.vstack(new_vertices)
    
    # Remove duplicates using vectorized approach
    if len(vertices) > 0:
        # Use a simple deduplication: keep first occurrence
        unique_verts = [vertices[0]]
        for v in vertices[1:]:
            dists = np.linalg.norm(unique_verts - v, axis=1)
            if np.min(dists) > 1e-8:
                unique_verts.append(v)
        vertices = np.array(unique_verts)
    
    # Select n points uniformly at random from vertices
    if len(vertices) >= n:
        idxs = np.random.choice(len(vertices), n, replace=False)
        return vertices[idxs]
    else:
        # Fallback to fibonacci if not enough
        return _fibonacci_sphere(n)


def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _local_minimize(points, deadline, maxiter=500, tol=1e-14):
    """Run L-BFGS-B to find local minimum with gradient descent fallback."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': tol, 'gtol': tol, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


def _gradient_descent_sphere(points, deadline, lr=0.05, maxiter=500):
    """Gradient descent on the sphere with adaptive step size and line search."""
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

        # Adaptive step with backtracking line search
        step = lr
        improved = False
        for _ in range(20):
            new_points = current - step * grads_tangent
            new_points = _project_to_sphere(new_points)
            new_energy = _log_energy(new_points)
            if new_energy < current_energy:
                current = new_points
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best = current.copy()
                improved = True
                break
            step *= 0.5
        
        if not improved:
            break
        # Slightly increase step if we made good progress
        if improved:
            lr = min(lr * 1.1, 0.5)

    return best, best_energy


def _basin_hopping(init_points, deadline, rng, max_iter=100):
    """Basin hopping with L-BFGS-B local minimization."""
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.15
    T_min = 1e-7
    cooling = 0.97
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        if T < T_min:
            break
            
        # Perturb: Move a random subset of points with larger steps
        n_perturb = rng.randint(3, 12)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * rng.uniform(0.8, 2.0)
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point
        
        # Local minimize
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


def _simulated_annealing(points, deadline, rng, max_iter=2000, T_start=0.05, T_end=1e-8):
    """Simulated annealing with single-point moves."""
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    n_iter = max_iter
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
        
        # Move one random point
        idx = rng.randint(N)
        tangent = _tangent_vector(current[idx], rng)
        step_size = T * rng.uniform(0.5, 2.0)
        new_point = current[idx] + step_size * tangent
        new_point = new_point / np.linalg.norm(new_point)
        
        new_points = current.copy()
        new_points[idx] = new_point
        new_energy = _log_energy(new_points)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / T):
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
    
    return best, best_energy


def _riemannian_cg(points, deadline, maxiter=500, tol=1e-14):
    """Riemannian conjugate gradient method on the sphere."""
    current = points.copy()
    n = len(current)
    
    # Initial gradient
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
    p_tangent = grads - dot_products * current
    
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    for it in range(maxiter):
        if time.time() > deadline - 0.5:
            break
        
        # Line search with backtracking
        step = 0.1
        improved = False
        for _ in range(30):
            new_points = current - step * p_tangent
            new_points = _project_to_sphere(new_points)
            new_energy = _log_energy(new_points)
            if new_energy < current_energy:
                current = new_points
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best = current.copy()
                improved = True
                break
            step *= 0.5
        
        if not improved:
            break
        
        # Compute new gradient
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
        g_tangent = grads - dot_products * current
        
        # Check convergence
        if np.max(np.linalg.norm(g_tangent, axis=1)) < tol:
            break
        
        # Fletcher-Reeves beta
        prev_g_sq = np.sum(p_tangent * p_tangent)
        if prev_g_sq > 1e-30:
            beta = np.sum(g_tangent * g_tangent) / prev_g_sq
            beta = max(0, min(beta, 1.0))
        else:
            beta = 0.0
        
        p_tangent = g_tangent + beta * p_tangent
        # Project to tangent space
        dot_p = np.sum(p_tangent * current, axis=1, keepdims=True)
        p_tangent = p_tangent - dot_p * current
    
    return best, best_energy


def _final_polish(points, deadline):
    """Final high-precision polishing with very tight tolerances."""
    if time.time() > deadline - 1.0:
        return points
    
    # First pass: L-BFGS-B with tight tolerance
    p1, e1 = _local_minimize(points, deadline, maxiter=2000, tol=1e-16)
    
    # Second pass: Riemannian CG for extra refinement
    if time.time() < deadline - 0.5:
        p2, e2 = _riemannian_cg(p1, deadline, maxiter=500, tol=1e-15)
        if e2 < e1:
            return p2
        return p1
    
    return p1


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Geodesic sphere (subdivided icosahedron) - better uniform spacing
    try:
        geodesic = _geodesic_sphere(N)
        inits.append(geodesic)
    except Exception:
        pass
    
    # 3. Random points with different seeds
    for seed in range(8):
        r = np.random.RandomState(seed * 13 + 7)
        inits.append(_random_uniform_sphere(N, r))
    
    # 4. Perturbed Fibonacci
    for k in range(5):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.15
        inits.append(_project_to_sphere(perturbed))
    
    # 5. Perturbed geodesic
    try:
        geodesic = _geodesic_sphere(N)
        for k in range(3):
            perturbed = geodesic + rng.randn(N, 3) * 0.1
            inits.append(_project_to_sphere(perturbed))
    except Exception:
        pass
    
    # Best so far
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: Local minimize all initializations
    candidates = []
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=400)
        candidates.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from top candidates
    n_bh = min(5, len(candidates))
    for i in range(n_bh):
        if time.time() > deadline - 12.0:
            break
        bh_rng = np.random.RandomState(i * 100 + 123)
        bh_points, bh_energy = _basin_hopping(candidates[i][1], deadline, bh_rng, max_iter=80)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()
    
    # Phase 3: Simulated annealing from best configuration
    if time.time() < deadline - 8.0:
        sa_rng = np.random.RandomState(999)
        sa_points, sa_energy = _simulated_annealing(
            best_points, deadline, sa_rng, 
            max_iter=1500, T_start=0.04, T_end=1e-9
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()
    
    # Phase 4: Final polishing with tight tolerances
    if time.time() < deadline - 1.0:
        polished = _final_polish(best_points, deadline)
        polished_energy = _log_energy(polished)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_points = polished
    
    best_points = _project_to_sphere(best_points)

    # Safety checks
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
