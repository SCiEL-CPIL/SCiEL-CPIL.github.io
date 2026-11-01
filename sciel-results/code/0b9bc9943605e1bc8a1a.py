"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

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
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # Hex lattice coordinates:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * j * sqrt(3) / 2
    max_k = int(1.0 / (s * 0.5)) + 3
    
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
        max_k = int(1.0 / (s * 0.5)) + 4
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
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    points = []
    # Estimate ring structure
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    ring_counts = []
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    diff = n - sum(ring_counts)
    if diff > 0:
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff == 0:
                break
    elif diff < 0:
        for i in range(len(ring_counts)-1, -1, -1):
            rem = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= rem
            diff += rem
            if diff == 0:
                break
    
    ring_counts = [c for c in ring_counts if c > 0]
    num_rings = len(ring_counts)
    
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    for k, count in enumerate(ring_counts):
        if k == 0:
            points.append([0.0, 0.0])
        else:
            r_k = k * s * 0.95
            angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.pi/count
            for angle in angles:
                points.append([r_k * np.cos(angle), r_k * np.sin(angle)])
    
    points = np.array(points[:n])
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _force_directed_relax(points, n_steps=50, dt=0.01):
    """
    Simple force-directed relaxation to improve initial configuration.
    Pushes overlapping circles apart and pulls boundary-violating circles inward.
    """
    p = points.copy()
    n = len(p)
    
    for step in range(n_steps):
        forces = np.zeros_like(p)
        
        # Pairwise repulsion for overlaps
        for i in range(n):
            for j in range(i+1, n):
                d = p[i] - p[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    d = np.array([1e-6, 0.0])
                    dist = np.linalg.norm(d)
                # If too close, push apart
                if dist < 0.1:
                    force_mag = (0.1 - dist) / dist
                    forces[i] += force_mag * d
                    forces[j] -= force_mag * d
        
        # Boundary attraction for points too close to edge
        for i in range(n):
            norm = np.linalg.norm(p[i])
            if norm > 0.95:
                # Pull inward
                forces[i] -= (norm - 0.95) * p[i] / norm
        
        # Update positions with damping
        p += dt * forces
        # Ensure points stay in reasonable bounds
        norms = np.linalg.norm(p, axis=1)
        mask = norms > 0.99
        p[mask] *= 0.99 / norms[mask, np.newaxis]
    
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization - try multiple initializations and keep the best
    np.random.seed(42)
    
    hex_init = _hex_lattice_init(N)
    ring_init = _ring_init(N)
    
    # Pre-condition with force-directed relaxation
    hex_relaxed = _force_directed_relax(hex_init, n_steps=30, dt=0.005)
    ring_relaxed = _force_directed_relax(ring_init, n_steps=30, dt=0.005)
    
    inits = [hex_init, hex_relaxed, ring_init, ring_relaxed]
    
    points = inits[0]
    best_points = points.copy()
    best_score = _compute_score(points)
    
    for init in inits[1:]:
        score = _compute_score(init)
        if score > best_score:
            best_score = score
            best_points = init
            points = init
    
    # 2. Optimization using L-BFGS-B with smooth soft-min objective
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        log_sum_exp = max_z + np.log(sum_exp)
        
        val = log_sum_exp / beta
        
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        i_idx, j_idx = iu
        
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs * 0.25 / safe_pair_dists
        vecs = p[i_idx] - p[j_idx]
        
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, -grad

    # Optimization loop with smoother beta schedule
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    current_points = points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-10,
                    'ftol': 1e-12,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 8:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                current_points = best_points
                
        except Exception:
            pass

    # Final polish passes with very high beta if time permits
    polish_betas = [10000.0, 20000.0, 50000.0]
    for pbeta in polish_betas:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x: objective_and_grad(x, pbeta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, pbeta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 150,
                    'gtol': 1e-12,
                    'ftol': 1e-14,
                    'maxls': 15,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
