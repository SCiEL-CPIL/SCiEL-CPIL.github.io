"""Improved program for packing 300 circles in a unit circle using physics-based simulation and soft-min optimization."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    Add small random perturbation to break symmetry.
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # If we didn't generate enough, shrink spacing slightly and retry
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _physics_simulation(points, max_iters=1000, dt=0.01, cooling_rate=0.999):
    """
    Physics-based simulation where circles exert repulsive forces on each other
    and the container wall, allowing the system to settle into a locally optimal configuration.
    """
    points = points.copy()
    temperature = 1.0
    
    for it in range(max_iters):
        if temperature < 1e-6:
            break
            
        forces = np.zeros_like(points)
        
        # Pairwise repulsive forces
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random push if coincident
                    diff = np.random.randn(2)
                    dist = np.linalg.norm(diff)
                
                # Repulsive force: stronger when closer
                # Use a soft repulsion that goes to infinity as dist -> 0
                # Force magnitude ~ 1/dist^2 for strong repulsion
                force_mag = 1.0 / (dist * dist + 1e-6)
                force_dir = diff / dist
                
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
        
        # Boundary forces
        for i in range(N):
            norm = np.linalg.norm(points[i])
            if norm > 0.99:
                # Push inward if too close to boundary
                force_mag = (norm - 0.99) * 10.0
                force_dir = -points[i] / norm
                forces[i] += force_mag * force_dir
        
        # Update positions with cooling
        points += dt * forces * temperature
        temperature *= cooling_rate
        
        # Keep points inside the circle
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] *= (0.999 / norms[mask])[:, np.newaxis]
        
        # Check convergence
        if it % 100 == 0 and it > 0:
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < 1e-6:
                break
    
    return points

def _soft_min_objective(x, beta):
    """
    Objective function: minimize -soft_min(constraint distances)
    Constraint distances: pairwise distances / 2 and boundary distances
    """
    p = x.reshape(-1, 2)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Pair distances: upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraint distances
    z_pairs = -beta * pair_dists
    z_bounds = -beta * bound_dists
    
    # Log-sum-exp for stability
    max_z_pair = np.max(z_pairs)
    max_z_bound = np.max(z_bounds)
    max_z = max(max_z_pair, max_z_bound)
    
    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)
    
    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    log_sum_exp = max_z + np.log(sum_exp)
    
    return log_sum_exp / beta

def _soft_min_gradient(x, beta):
    """
    Gradient of the soft-min objective function.
    """
    p = x.reshape(-1, 2)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Pair distances: upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # Compute weights for soft-min
    z_pairs = -beta * pair_dists
    z_bounds = -beta * bound_dists
    
    max_z_pair = np.max(z_pairs)
    max_z_bound = np.max(z_bounds)
    max_z = max(max_z_pair, max_z_bound)
    
    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)
    
    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    
    w_pairs = exp_pairs / sum_exp
    w_bounds = exp_bounds / sum_exp
    
    grad = np.zeros_like(p)
    
    # Pair contributions
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization with hexagonal lattice
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Physics-based simulation to find a good local configuration
    if time.time() < deadline - 60:
        points_physics = _physics_simulation(points, max_iters=500, dt=0.005, cooling_rate=0.998)
        score_physics = _compute_score(points_physics)
        if score_physics > best_score:
            best_points = points_physics.copy()
            best_score = score_physics
    
    # 3. Optimization using L-BFGS-B with smooth soft-min objective
    # Adaptive beta schedule: start low, increase for refinement
    betas = [50, 100, 200, 500, 1000]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        try:
            def objective(x):
                return _soft_min_objective(x, beta)
            
            def gradient(x):
                return _soft_min_gradient(x, beta)
            
            res = minimize(objective, current_points.flatten(), jac=gradient, 
                          method='L-BFGS-B',
                          bounds=[(-1.0, 1.0)] * (2 * N),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < _soft_min_objective(current_points, beta):
                candidate = res.x.reshape(-1, 2)
                score_candidate = _compute_score(candidate)
                if score_candidate > best_score:
                    best_points = candidate.copy()
                    best_score = score_candidate
                    current_points = candidate.copy()
        except Exception:
            pass
    
    # 4. Final high-precision polish with very high beta
    if time.time() < deadline - 10:
        try:
            beta_final = 2000
            def objective_final(x):
                return _soft_min_objective(x, beta_final)
            
            def gradient_final(x):
                return _soft_min_gradient(x, beta_final)
            
            res = minimize(objective_final, best_points.flatten(), jac=gradient_final,
                          method='L-BFGS-B',
                          bounds=[(-1.0, 1.0)] * (2 * N),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            
            candidate = res.x.reshape(-1, 2)
            score_candidate = _compute_score(candidate)
            if score_candidate > best_score:
                best_points = candidate.copy()
                best_score = score_candidate
        except Exception:
            pass
    
    # 5. Final validation and cleanup
    best_points = best_points.reshape(-1, 2)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (0.999999 / norms[mask])[:, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
