"""
Logarithmic energy minimization of 150 points on the unit sphere.
Uses Parallel Tempering (Replica Exchange) Simulated Annealing with periodic L-BFGS polishing.
"""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    The gradient is projected to the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    # We only need the upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient
    # grad_i = sum_j (x_j - x_i) / dist_ij^2
    # Note: 1/dist2 is symmetric
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad = inv_dist2 @ pts - pts * row_sums
    # row_sums[i] = sum_j inv_dist2[i,j]
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts
    grad -= pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space
    # tangent_grad = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad

def _energy_only(pts):
    """Compute only the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())

def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms

def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _lbfgs_polish(pts, deadline, maxiter=200, ftol=1e-12, gtol=1e-10):
    """
    L-BFGS-B polish using flat Cartesian coordinates with projection.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 20,
                                'ftol': ftol, 'gtol': gtol, 'maxls': 30})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        e_orig = _energy_only(pts)
        if e_check < e_orig:
            return p_res, e_check
    except Exception:
        pass
    return pts, _energy_only(pts)

def _perturb_single_point(pts, rng, step_size):
    """
    Move a single random point by a small step in a random tangent direction.
    """
    i = rng.randint(N)
    # Generate random tangent vector
    # Random vector on sphere
    rand_vec = rng.randn(3)
    # Project to tangent space of pts[i]
    tangent = rand_vec - np.dot(rand_vec, pts[i]) * pts[i]
    norm_t = np.linalg.norm(tangent)
    if norm_t < 1e-12:
        return pts
    tangent = tangent / norm_t
    
    # Move
    new_pts = pts.copy()
    new_pts[i] = pts[i] + step_size * tangent
    new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
    return new_pts

def _parallel_tempering(deadline, num_replicas=8, t_high=1.0, t_low=0.01, 
                        perturb_step=0.1, polish_interval=200, polish_time=2.0):
    """
    Parallel Tempering (Replica Exchange) Simulated Annealing.
    """
    # Initialize replicas
    # Use diverse initializations
    inits = [
        _fibonacci_spiral(N),
        _fibonacci_spiral(N, twist=1.5),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
        _random_points(N, seed=4),
        _random_points(N, seed=5),
        _random_points(N, seed=6)
    ]
    
    # If we have more replicas than inits, cycle
    replicas = []
    energies = []
    for i in range(num_replicas):
        p = inits[i % len(inits)].copy()
        # Add slight noise to ensure distinctness if cycling
        if i >= len(inits):
            rng = np.random.RandomState(100+i)
            p = _project_to_sphere(p + rng.randn(N, 3) * 0.01)
        replicas.append(p)
        energies.append(_energy_only(p))
        
    # Temperatures: Geometric progression
    # T[0] is lowest (coldest), T[-1] is highest (hottest)
    temps = np.geomspace(t_low, t_high, num_replicas)
    
    best_energy_global = min(energies)
    best_pts_global = replicas[np.argmin(energies)].copy()
    
    rng = np.random.RandomState(42)
    
    # Annealing parameters
    # We will cool down temperatures slowly
    # T_new = T * cooling_factor
    cooling_factor = 0.999
    min_temp = 0.001
    
    step_count = 0
    last_polish_time = time.time()
    
    while time.time() < deadline - 10.0: # Leave 10s for final polish
        # 1. Select a random replica
        k = rng.randint(num_replicas)
        
        # 2. Perturb it
        new_pts = _perturb_single_point(replicas[k], rng, perturb_step)
        new_energy = _energy_only(new_pts)
        
        # 3. Metropolis Accept/Reject
        delta_e = new_energy - energies[k]
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / temps[k]):
            replicas[k] = new_pts
            energies[k] = new_energy
            
            # Update global best
            if energies[k] < best_energy_global:
                best_energy_global = energies[k]
                best_pts_global = replicas[k].copy()
                
        # 4. Attempt Exchange with a neighbor
        # Choose direction: -1 or 1
        dir = rng.choice([-1, 1])
        j = k + dir
        if 0 <= j < num_replicas:
            # Exchange replicas k and j
            # Probability: exp(- (E_k - E_j) / (1/T_k - 1/T_j))
            # Let T_low be the lower temp (higher beta)
            if temps[k] < temps[j]:
                T_low = temps[k]
                T_high = temps[j]
                E_low = energies[k]
                E_high = energies[j]
                # We are swapping state of k (low T) with j (high T)
                # Delta E for swap = E_j_new - E_k_new ? No.
                # Standard formula: P = exp( - (E_k - E_j) * (1/T_k - 1/T_j) )
                # Let's use: P = exp( - (E_k * 1/T_k - E_j * 1/T_j) - (E_j * 1/T_k - E_k * 1/T_j) )
                # Actually, the change in "H" = E_k/T_k + E_j/T_j
                # H_old = E_k/T_k + E_j/T_j
                # H_new = E_j/T_k + E_k/T_j
                # Delta H = (E_j - E_k)/T_k + (E_k - E_j)/T_j = (E_j - E_k) * (1/T_k - 1/T_j)
                delta_h = (energies[j] - energies[k]) * (1.0/temps[k] - 1.0/temps[j])
                if delta_h < 0 or rng.rand() < math.exp(-delta_h):
                    # Swap
                    replicas[k], replicas[j] = replicas[j], replicas[k]
                    energies[k], energies[j] = energies[j], energies[k]
                    
                    # Update global best
                    if energies[k] < best_energy_global:
                        best_energy_global = energies[k]
                        best_pts_global = replicas[k].copy()
                    if energies[j] < best_energy_global:
                        best_energy_global = energies[j]
                        best_pts_global = replicas[j].copy()
                        
        # 5. Periodic Polishing
        # Polish the coldest replica (index 0) or the current best
        # To save time, only polish if enough time has passed
        if step_count % polish_interval == 0:
            if time.time() > last_polish_time + polish_time:
                # Polish the replica with the lowest energy
                idx_best = np.argmin(energies)
                p_polished, e_polished = _lbfgs_polish(replicas[idx_best], time.time() + 1.0, maxiter=50)
                if e_polished < energies[idx_best]:
                    replicas[idx_best] = p_polished
                    energies[idx_best] = e_polished
                    if e_polished < best_energy_global:
                        best_energy_global = e_polished
                        best_pts_global = p_polished.copy()
                last_polish_time = time.time()
                
        # 6. Cooling
        # Cool down temperatures
        temps = temps * cooling_factor
        # Ensure temps don't go below min_temp
        temps = np.maximum(temps, min_temp)
        
        step_count += 1

    return best_pts_global, best_energy_global

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Run Parallel Tempering
    # We have ~145s. Let's use 120s for PT and 25s for final polish.
    pt_deadline = start_time + 120.0
    
    best_pts, best_energy = _parallel_tempering(pt_deadline)
    
    # Final High-Precision Polish
    # Use remaining time for a very tight L-BFGS
    final_deadline = deadline - 2.0
    if time.time() < final_deadline:
        # First, a standard polish
        p1, e1 = _lbfgs_polish(best_pts, final_deadline - 10.0, maxiter=500, ftol=1e-14, gtol=1e-12)
        if e1 < best_energy:
            best_pts, best_energy = p1, e1
            
        # Second, an even tighter polish if time allows
        if time.time() < final_deadline - 1.0:
            p2, e2 = _lbfgs_polish(best_pts, final_deadline - 1.0, maxiter=1000, ftol=1e-16, gtol=1e-14)
            if e2 < best_energy:
                best_pts, best_energy = p2, e2

    best_pts = _project_to_sphere(best_pts)

    # Validation
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N)

    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_pts

# EVOLVE-BLOCK-END
