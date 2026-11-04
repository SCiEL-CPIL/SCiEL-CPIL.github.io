"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Minimum distance to boundary
    min_boundary = HALF - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)

def _build_contact_graph(centers, r_target, expansion_factor=1.2):
    """
    Build contact graph using Delaunay triangulation.
    Returns list of (i, j) pairs that are in contact or near-contact.
    """
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
        
        # Also add edges from KDTree for robustness
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        pairs = tree.query_pairs(2.0 * r_target * expansion_factor)
        for i, j in pairs:
            edges.add((min(i, j), max(i, j)))
        
        return list(edges)
    except Exception:
        # Fallback: use KDTree only
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        pairs = tree.query_pairs(2.0 * r_target * expansion_factor)
        return [(min(i, j), max(i, j)) for i, j in pairs]

def _force_directed_relaxation(centers, r_target, iterations=500, dt=0.001, time_limit=30.0):
    """
    Force-directed graph relaxation treating the packing as a rigid framework.
    Springs enforce target edge lengths, walls provide repulsive forces.
    """
    t_start = time.time()
    c = centers.copy()
    n = len(c)
    
    # Build initial contact graph
    edges = _build_contact_graph(c, r_target)
    
    # Spring constants
    k_spring = 100.0  # For maintaining target distances
    k_wall = 200.0    # For wall repulsion
    
    for it in range(iterations):
        if time.time() - t_start > time_limit:
            break
        
        # Rebuild contact graph every 50 iterations to capture topology changes
        if it % 50 == 0 and it > 0:
            edges = _build_contact_graph(c, r_target)
        
        forces = np.zeros_like(c)
        
        # Spring forces: try to maintain distance = 2*r_target for contact pairs
        for i, j in edges:
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            target_dist = 2.0 * r_target
            # Hooke's law: F = -k * (dist - target) * unit_vec
            # Force on i is in direction of j if dist > target (pulling together)
            # Force on i is in direction away from j if dist < target (pushing apart)
            force_mag = k_spring * (target_dist - dist)
            unit_vec = diff / dist
            forces[i] += force_mag * unit_vec
            forces[j] -= force_mag * unit_vec
        
        # Wall forces: repulsive when too close to walls
        for axis in range(3):
            # Positive wall (x = +0.5)
            wall_dist_pos = HALF - c[:, axis]
            wall_viol_pos = np.maximum(0, r_target - wall_dist_pos)
            forces[:, axis] -= k_wall * wall_viol_pos
            
            # Negative wall (x = -0.5)
            wall_dist_neg = HALF + c[:, axis]
            wall_viol_neg = np.maximum(0, r_target - wall_dist_neg)
            forces[:, axis] += k_wall * wall_viol_neg
        
        # Update positions with damping
        c += dt * forces
        # Apply light damping by scaling velocity implicitly
        # Keep within bounds
        c = np.clip(c, -0.499, 0.499)
    
    return c

def _hybrid_optimize(centers, r_target, time_limit=40.0):
    """
    Hybrid approach: force-directed relaxation followed by BFGS polishing.
    """
    t_start = time.time()
    
    # Phase 1: Force-directed relaxation
    remaining1 = min(20.0, time_limit * 0.6)
    if remaining1 > 5.0:
        c_relaxed = _force_directed_relaxation(centers, r_target, 
                                                iterations=300, dt=0.002,
                                                time_limit=remaining1)
        if np.all(np.isfinite(c_relaxed)):
            c_relaxed = np.clip(c_relaxed, -0.499, 0.499)
            centers = c_relaxed
    
    # Phase 2: BFGS polishing
    remaining2 = time_limit - (time.time() - t_start) - 2.0
    if remaining2 > 5.0:
        try:
            from scipy.optimize import minimize as sp_minimize
            result = sp_minimize(
                _soft_min_obj_grad,
                centers.flatten(),
                args=(50.0,),
                method='BFGS',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
            )
            if np.all(np.isfinite(result.x)):
                centers = result.x.reshape(N, 3)
                centers = np.clip(centers, -0.499, 0.499)
        except Exception:
            pass
    
    return centers

def _soft_min_obj_grad(x, beta):
    """
    Objective function: minimize -soft_min(effective_radii).
    effective_radii = [pair_dists/2, wall_dists]
    soft_min(z) = -1/beta * log(sum(exp(-beta * z)))
    We want to maximize soft_min, so we minimize -soft_min.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.minimum(HALF - c, HALF + c)  # min distance to each wall
    
    # All effective radii
    pair_radii = dists / 2.0
    # Flatten pair radii (upper triangle only)
    iu = np.triu_indices(N, k=1)
    pair_r_flat = pair_radii[iu]
    wall_r_flat = wall_dists.flatten()
    
    all_r = np.concatenate([pair_r_flat, wall_r_flat])
    
    # Soft min
    # To avoid overflow, use log-sum-exp trick
    min_r = all_r.min()
    shifted = all_r - min_r
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = -min_r - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    # Gradient of -soft_min wrt each r_k
    # d(-soft_min)/dr_k = -exp(-beta * r_k) / sum(exp(-beta * r_k))
    exp_terms = np.exp(-beta * shifted)
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient wrt positions
    grad = np.zeros_like(c)
    
    # Pairwise contribution
    # For pair (i,j): r_ij = ||c_i - c_j||/2
    # d(-soft_min)/d c_i = -w_ij * d r_ij / d c_i = -w_ij * (c_i - c_j) / (2 * ||c_i - c_j||)
    # But we need to sum over all pairs involving i
    
    # Reconstruct pairwise weights
    pair_weights = np.zeros((N, N))
    pair_flat_idx = 0
    for i in range(N):
        for j in range(i+1, N):
            pair_weights[i, j] = weights[pair_flat_idx]
            pair_weights[j, i] = weights[pair_flat_idx]
            pair_flat_idx += 1
    
    # Compute gradient from pairs
    for i in range(N):
        for j in range(N):
            if i == j or pair_weights[i, j] == 0:
                continue
            diff_ij = c[i] - c[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-12:
                continue
            grad[i] -= pair_weights[i, j] * diff_ij / (2.0 * dist_ij)
    
    # Wall contribution
    wall_flat_idx = 0
    for i in range(N):
        for axis in range(3):
            w = weights[wall_flat_idx]
            wall_flat_idx += 1
            # r_wall = min(HALF - c_ik, HALF + c_ik)
            # If c_ik >= 0: r = HALF - c_ik, d r/d c_ik = -1
            # If c_ik < 0: r = HALF + c_ik, d r/d c_ik = +1
            if c[i, axis] >= 0:
                grad[i, axis] += w
            else:
                grad[i, axis] -= w
    
    obj_val = -soft_min_val
    return obj_val, grad.flatten()

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin under 150s

    # ---- Seed: Simple cubic 5x5x5 grid ----
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    rng = np.random.RandomState(42)

    # Strategy: Force-directed relaxation with hybrid BFGS polishing
    # Try a sequence of target radii
    
    targets = [0.105, 0.108, 0.110, 0.112, 0.114, 0.116]
    
    current_pts = best_pts.copy()
    
    for target_r in targets:
        if time.time() - t0 > DEADLINE - 15.0:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.003, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining = DEADLINE - (time.time() - t0) - 10.0
        if remaining < 10:
            break
            
        opt_pts = _hybrid_optimize(
            init_pts, 
            target_r, 
            time_limit=remaining
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()

    # If time permits, try additional restarts with larger noise
    if time.time() - t0 < DEADLINE - 20.0:
        for i in range(3):
            if time.time() - t0 > DEADLINE - 10.0:
                break
            
            noise = rng.normal(0, 0.015, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            target_r = best_r * 1.005
            
            remaining = DEADLINE - (time.time() - t0) - 5.0
            if remaining < 10:
                break
                
            opt_pts = _hybrid_optimize(
                init_pts, 
                target_r, 
                time_limit=remaining
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    # Final Polish: BFGS on the best configuration
    if time.time() - t0 < DEADLINE - 5.0 and best_r > 0.1:
        noise = rng.normal(0, 0.0005, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        remaining = DEADLINE - (time.time() - t0) - 2.0
        if remaining > 3:
            try:
                result = minimize(
                    _soft_min_obj_grad,
                    init_pts.flatten(),
                    args=(100.0,),
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 300, 'gtol': 1e-10, 'disp': False}
                )
                if np.all(np.isfinite(result.x)):
                    opt_pts = result.x.reshape(N, 3)
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _compute_radius(opt_pts)
                    
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()
            except Exception:
                pass

    return best_pts
# EVOLVE-BLOCK-END
