"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = 142.0  # Safety margin before 150s kill

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum for precision."""
    n = len(points)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Extract upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    
    # Energy is sum of log(1/d) = -sum(log(d))
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Energy and gradient for L-BFGS-B. 
    x is flat array of shape (N*3,).
    The function normalizes x to the sphere internally.
    Returns (energy, gradient_flat).
    """
    points = x.reshape(N, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1e-15, norms)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: -sum_{i<j} log(d_ij)
    # d_ij = sqrt(dists_sq_ij)
    # log(d_ij) = 0.5 * log(dists_sq_ij)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists_sq[iu]))
    
    # Gradient calculation
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # Note: This is the gradient in the ambient R^3 space.
    # Since we are projecting to the sphere, the effective gradient on the manifold 
    # is the projection of this gradient onto the tangent space.
    # However, L-BFGS-B works in R^3. If we provide the ambient gradient, 
    # the optimizer will try to move in the normal direction, which is then 
    # projected away by the normalization in the next step.
    # It is more efficient to provide the tangential gradient.
    
    # weights[i,j] = 1/dists_sq[i,j] for i!=j, 0 otherwise
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    # grad_i = sum_j (1/d_ij^2) * (x_j - x_i)
    # grad_i = sum_j (1/d_ij^2) x_j - x_i sum_j (1/d_ij^2)
    sum_weights = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space: g_tan = g - (g . p) p
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    return energy, grads_tangent.flatten()

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _icosahedron_based(n, rng):
    """
    Generate points based on icosahedron vertices, expanded and rotated.
    This provides a more symmetric initial distribution than pure random.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # We need N points. We can use a "spherical Fibonacci" on the faces or just random rotations of the icosahedron.
    # Simpler: Use random rotations of a set of points derived from icosahedron.
    # For N=150, we can just use random points but start with a few icosahedron-based seeds.
    # Let's stick to a mix of Fibonacci and Random for simplicity and robustness.
    return _random_uniform_sphere(n, rng)

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

def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    
    # Check if we have time
    if time.time() > deadline - 1.0:
        return points, _log_energy(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        
        if time.time() > deadline - 1.0:
             return points, _log_energy(points)
             
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        # Use the precise energy function for comparison
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping_single(init_points, deadline, rng, max_iter=100, T_init=0.1):
    """
    Basin hopping:
    1. Local minimize
    2. Perturb
    3. Local minimize
    4. Accept if better (Metropolis)
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-8
    cooling = 0.97  # Slower cooling for more exploration

    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        # Larger perturbations early on, smaller later
        n_perturb = rng.randint(2, 10)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            # Step size proportional to T
            step_size = T
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        # Use fewer iterations for inner loop to save time
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=100)

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

def _polish_high_precision(pts, max_time):
    """High-precision final polish with L-BFGS at tight tolerances."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
        
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 100000, 'maxls': 50}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    rng = np.random.RandomState(42)

    best_x = None
    best_e = float('inf')

    # Phase 1: Diverse Initialization and Local Minimization
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random uniform points (multiple seeds)
    for seed in range(5):
        r = np.random.RandomState(seed * 100 + 1)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci
    for k in range(3):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.1
        inits.append(_project_to_sphere(perturbed))
        
    # 4. "Clustered" random to test escape from local minima
    for k in range(2):
        centers = _random_uniform_sphere(5, rng)
        pts = []
        for c in centers:
            # Generate 30 points around each center
            noise = rng.randn(30, 3) * 0.5
            local_pts = c[None, :] + noise
            local_pts = _project_to_sphere(local_pts)
            pts.append(local_pts)
        inits.append(np.vstack(pts))

    # Run local minimization on each to find good local minima
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Allocate time for this local minimization
        # We want to leave time for basin hopping and polish
        # Let's give each local min ~2-5 seconds depending on how many are left
        time_per_min = min(5.0, remaining / max(1, len(inits) - i))
        
        x_res, e_res = _local_minimize(init, deadline, maxiter=200, ftol=1e-10, gtol=1e-8)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res

    # Phase 2: Basin Hopping from the best local minimum found
    if best_x is not None and time.time() < deadline - 20.0:
        # Run basin hopping from the best point
        # Also try from a few other good points if time permits
        # For now, just use the best one to focus time.
        
        x_bh, e_bh = _basin_hopping_single(best_x, deadline, rng, max_iter=150, T_init=0.05)
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh

    # Phase 3: Final High-Precision Polish
    if best_x is not None and time.time() < deadline - 2.0:
        polish_time = deadline - time.time() - 1.0
        if polish_time > 0.5:
            x_final, e_final = _polish_high_precision(best_x, polish_time)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final

    # Final validation and projection
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    # The problem states points closer than 1e-9 are invalid.
    # Our optimization should naturally avoid this, but let's check.
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    if dist[iu].min() < 1e-9:
        # Fallback to Fibonacci if points collapsed
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
# EVOLVE-BLOCK-END
