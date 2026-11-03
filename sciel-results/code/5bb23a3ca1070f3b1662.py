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
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    log_dists = np.log(dists)
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
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    weighted_points = inv_dists_sq @ points
    
    grad = weighted_points - points * sum_inv_dists_sq[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot_products * points
    
    energy = _log_energy(points)
    
    return energy, grad_tangent.flatten()

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

def _geodesic_sphere(n, rng):
    """Generate points using a subdivided icosahedron."""
    # Simple approximation: Start with icosahedron vertices and subdivide
    # For simplicity, we'll just use random points but ensure minimum distance via rejection?
    # Or just use Fibonacci which is good.
    # Let's stick to Fibonacci and Random for diversity.
    return _fibonacci_sphere(n)

def _local_minimize(points, deadline, maxiter=1000, tol=1e-12):
    """Run L-BFGS-B to find local minimum."""
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

def _simulated_annealing(points, deadline, rng, max_iter=5000):
    """Simulated Annealing to escape local minima."""
    current_points = points.copy()
    current_energy = _log_energy(current_points)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.5
    T_min = 1e-4
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break
            
        # Perturb a random subset of points
        n_perturb = rng.randint(1, 10)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_points[idx]) * new_points[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            else:
                continue
            step_size = T * rng.uniform(0.1, 1.0)
            new_pt = new_points[idx] + step_size * tangent
            new_points[idx] = new_pt / np.linalg.norm(new_pt)
        
        new_energy = _log_energy(new_points)
        
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    
    for seed in range(5):
        r = np.random.RandomState(seed)
        inits.append(_random_uniform_sphere(N, r))
        
    for k in range(5):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.2
        inits.append(_project_to_sphere(perturbed))

    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: Local minimize all inits
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=200)
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Phase 2: Simulated Annealing from best candidates
    # Run SA from the current best and a few others
    candidates = [best_points]
    for init in inits[:3]:
        candidates.append(init)
        
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        sa_rng = np.random.RandomState(i * 100 + 123)
        sa_points, sa_energy = _simulated_annealing(cand, deadline, sa_rng, max_iter=2000)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()

    # Phase 3: Final polishing
    if time.time() < deadline - 2.0:
        final_points, final_energy = _local_minimize(best_points, deadline, maxiter=1000, tol=1e-14)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    best_points = _project_to_sphere(best_points)

    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
