"""Improved packing of 50 equal spheres in a unit cube using Hilbert init + gradient optimization."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score using KDTree for speed."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    return min(min_pair / 2.0, min_wall)

def _hilbert_curve_3d(n, order=4):
    """Generate Hilbert curve points in 3D unit cube.
    
    Args:
        n: number of points to generate
        order: Hilbert curve order (2^order points per axis)
    
    Returns:
        Array of shape (n, 3) with points in [0, 1]^3
    """
    # Generate full Hilbert curve in 3D
    size = 2 ** order
    total_points = size ** 3
    
    # For 3D Hilbert curve, we use a recursive approach
    # This generates points in order along the Hilbert curve
    def hilbert_3d(order):
        if order == 0:
            return np.array([[0, 0, 0]], dtype=np.float64)
        
        # 8 sub-cubes in 3D Hilbert curve
        # The order of visiting sub-cubes in 3D Hilbert curve
        # This is a simplified version - for better results, use proper 3D Hilbert
        sub_order = [
            [0,0,0], [0,0,1], [0,1,1], [0,1,0],
            [1,1,0], [1,1,1], [1,0,1], [1,0,0]
        ]
        
        results = []
        for sx, sy, sz in sub_order:
            sub = hilbert_3d(order - 1)
            # Scale and offset
            sub_scaled = sub / (2 ** order) + np.array([sx, sy, sz]) / (2 ** order)
            # Some sub-cubes need to be rotated/reflected for proper Hilbert connectivity
            # Simplified: just use the basic mapping
            results.append(sub_scaled)
        
        return np.vstack(results)
    
    # Generate full curve
    full_curve = hilbert_3d(order)
    
    # Take first n points (or sample evenly if n > total_points)
    if n <= total_points:
        return full_curve[:n]
    else:
        # Sample evenly from the full curve
        indices = np.linspace(0, total_points - 1, n, dtype=int)
        return full_curve[indices]

def _soft_min_score(centers, beta=50.0):
    """Differentiable soft-minimum score for gradient-based optimization."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    a = min_pair / 2.0
    b = min_wall
    # Soft min: -1/beta * log(exp(-beta*a) + exp(-beta*b))
    m = min(a, b)
    return m - math.log(math.exp(-beta * (a - m)) + math.exp(-beta * (b - m))) / beta

def _objective_flat(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft-min score."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    return -_soft_min_score(centers, beta=beta)

def _initial_grid_444():
    """4x4x4 grid, first 50 points."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _initial_hilbert():
    """Initialize using Hilbert curve mapping to 3D cube [-0.5, 0.5]^3."""
    # Generate Hilbert curve points in [0, 1]^3
    hilbert_pts = _hilbert_curve_3d(N, order=4)
    
    # Map to [-0.5, 0.5]^3
    centers = hilbert_pts - 0.5
    
    # Add small jitter to break symmetries
    centers += np.random.uniform(-0.001, 0.001, centers.shape)
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _repulsive_dynamics(centers, time_limit=5.0, steps=2000):
    """
    Repulsive dynamics to expand the packing.
    Uses pairwise repulsion and wall repulsion with adaptive step size.
    """
    current = centers.copy()
    start_time = time.time()
    
    # Parameters
    k_repel = 0.02
    k_wall = 0.02
    dt = 0.005
    dt_decay = 0.999
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        forces = np.zeros_like(current)
        
        # Pairwise repulsion - vectorized for speed
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d < 1e-8:
                    # Jitter if coincident
                    diff = np.random.uniform(-0.01, 0.01, 3)
                    d = np.linalg.norm(diff)
                # Force magnitude - stronger repulsion for close pairs
                f_mag = k_repel / (d * d + 1e-8)
                f_vec = (diff / d) * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec
                
        # Wall repulsion
        for i in range(N):
            for axis in range(3):
                # Distance to positive wall
                d_pos = 0.5 - current[i, axis]
                if d_pos < 0.15:
                    f_mag = k_wall / (d_pos * d_pos + 1e-8)
                    forces[i, axis] -= f_mag
                
                # Distance to negative wall
                d_neg = 0.5 + current[i, axis]
                if d_neg < 0.15:
                    f_mag = k_wall / (d_neg * d_neg + 1e-8)
                    forces[i, axis] += f_mag
        
        # Update positions with adaptive step
        current += forces * dt
        current = np.clip(current, -0.5, 0.5)
        
        # Decay step size
        dt *= dt_decay
        
    return current

def _l_bfgs_refine(centers, time_limit=10.0):
    """Refine using L-BFGS-B with soft-minimum objective."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Coarse pass
    if time.time() - start_time < time_limit * 0.4:
        result1 = minimize(
            _objective_flat, x0, args=(15.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        x0 = result1.x
    
    # Medium pass
    if time.time() - start_time < time_limit * 0.7:
        result2 = minimize(
            _objective_flat, x0, args=(50.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9}
        )
        x0 = result2.x
    
    # Fine pass
    if time.time() - start_time < time_limit * 0.95:
        result3 = minimize(
            _objective_flat, x0, args=(150.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-10}
        )
        x0 = result3.x
    
    return np.clip(x0.reshape(N, 3), -0.5, 0.5)

def _kick_local_search(centers, time_limit=5.0, max_iter=20000):
    """
    Local search that resolves overlaps by kicking spheres apart.
    Uses adaptive step size and multiple perturbation strategies.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()
    
    # Adaptive parameters
    kick_strength = 1.0
    perturbation_scale = 0.005
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find minimum distance pair
        tree = cKDTree(current)
        dists, idxs = tree.query(current, k=2)
        min_d = np.min(dists[:, 1])
        min_idx_i = np.argmin(dists[:, 1])
        min_idx_j = idxs[min_idx_i, 1]
        
        # Check wall distance
        wall_dists = 0.5 - np.abs(current)
        min_wall = np.min(wall_dists)
        
        current_radius = min(min_d / 2.0, min_wall)
        
        if current_radius > best_score:
            best = current.copy()
            best_score = current_radius
            # Reset kick strength when we improve
            kick_strength = 1.0
        
        # If there is an overlap, kick them apart
        if min_d < 2.0 * best_score * 1.005:
            diff = current[min_idx_i] - current[min_idx_j]
            d = np.linalg.norm(diff)
            if d > 1e-8:
                # Push apart with adaptive strength
                target_d = 2.0 * best_score * 1.005
                push_mag = (target_d - min_d) * 0.5 * kick_strength
                push = (diff / d) * push_mag
                current[min_idx_i] += push
                current[min_idx_j] -= push
                kick_strength *= 0.95  # Decay to avoid oscillation
        
        # Also handle wall collisions
        for i in range(N):
            for axis in range(3):
                margin = best_score * 1.005
                if current[i, axis] > 0.5 - margin:
                    current[i, axis] = 0.5 - margin
                elif current[i, axis] < -0.5 + margin:
                    current[i, axis] = -0.5 + margin
        
        # Multiple perturbation strategies
        if it % 50 == 0:
            # Single point perturbation
            idx = np.random.randint(N)
            current[idx] += np.random.uniform(-perturbation_scale, perturbation_scale, 3)
            current = np.clip(current, -0.5, 0.5)
        
        if it % 200 == 0:
            # Multi-point perturbation
            n_perturb = np.random.randint(2, 5)
            indices = np.random.choice(N, n_perturb, replace=False)
            for idx in indices:
                current[idx] += np.random.uniform(-perturbation_scale * 0.5, perturbation_scale * 0.5, 3)
            current = np.clip(current, -0.5, 0.5)
        
        if it % 500 == 0:
            # Larger perturbation for escaping deep local minima
            idx = np.random.randint(N)
            current[idx] += np.random.uniform(-perturbation_scale * 3, perturbation_scale * 3, 3)
            current = np.clip(current, -0.5, 0.5)
        
        # Gradually increase perturbation scale
        perturbation_scale = min(0.02, perturbation_scale * 1.001)
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Try multiple initializations and keep the best
    initializations = [
        _initial_grid_444(),
        _initial_hilbert()
    ]
    
    best_centers = None
    best_score = -1.0
    
    for init_centers in initializations:
        score = _exact_score(init_centers)
        if score > best_score:
            best_score = score
            best_centers = init_centers.copy()
    
    # 1. Repulsive Dynamics to expand
    if time.time() < deadline - 40.0:
        dynamic_centers = _repulsive_dynamics(best_centers, time_limit=15.0)
        dyn_score = _exact_score(dynamic_centers)
        if dyn_score > best_score:
            best_score = dyn_score
            best_centers = dynamic_centers
    
    # 2. Kick Local Search
    if time.time() < deadline - 30.0:
        ls_centers, ls_score = _kick_local_search(best_centers, time_limit=15.0)
        if ls_score > best_score:
            best_score = ls_score
            best_centers = ls_centers
    
    # 3. L-BFGS-B Refinement
    if time.time() < deadline - 20.0:
        refined = _l_bfgs_refine(best_centers, time_limit=20.0)
        ref_score = _exact_score(refined)
        if ref_score > best_score:
            best_score = ref_score
            best_centers = refined
    
    # 4. Final Polish with Kick LS
    if time.time() < deadline - 5.0:
        ls_centers, ls_score = _kick_local_search(best_centers, time_limit=10.0)
        if ls_score > best_score:
            best_score = ls_score
            best_centers = ls_centers
    
    # 5. Final L-BFGS polish if time remains
    if time.time() < deadline - 2.0:
        refined = _l_bfgs_refine(best_centers, time_limit=5.0)
        ref_score = _exact_score(refined)
        if ref_score > best_score:
            best_score = ref_score
            best_centers = refined
            
    return best_centers

# EVOLVE-BLOCK-END
