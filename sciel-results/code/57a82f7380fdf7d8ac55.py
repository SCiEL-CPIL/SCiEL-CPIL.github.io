import numpy as np
from scipy.optimize import minimize
import time

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # 150s limit, keep 5s margin
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Computes the exact radius for a given set of centers.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    # Pairwise distances
    # Using broadcasting for vectorization, but N=100 is small enough for loops if needed.
    # Let's use a vectorized approach for speed.
    # dists shape (N, N)
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # centers shape (N, 3)
    # dist to walls: 0.5 - abs(coord)
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_objective(params, beta):
    """
    Objective function to MINIMIZE.
    We want to MAXIMIZE r.
    We define the objective as: -r + penalty_violations.
    
    Actually, a better formulation for L-BFGS-B:
    Minimize: -R + lambda * (soft_violations)
    
    Constraints:
    1. Pairwise: |C_i - C_j| >= 2R  =>  2R - |C_i - C_j| <= 0
    2. Wall: |C_ik| <= 0.5 - R     =>  |C_ik| + R - 0.5 <= 0
    
    Instead of hard constraints, we use a soft penalty on violations.
    Let V_pair = max(0, 2R - dist_ij). Soft version: -1/beta * log(sum(exp(-beta * dist_ij / (2R))) )? No, that's soft min of distances.
    
    Let's use the following smooth surrogate for the "minimum distance" constraint:
    We want to ensure min_dist >= 2R.
    Equivalent to: -min_dist + 2R <= 0.
    
    Let D_min_soft = -1/beta * log( sum_{i<j} exp(-beta * d_ij) ).
    This approximates min(d_ij) as beta -> inf.
    
    Objective: J = -R + mu * max(0, 2R - D_min_soft)^2 + nu * max(0, R + max_coord_abs - 0.5)^2
    
    To make it fully differentiable and smooth, we can just minimize:
    J = -R + mu * (2R - D_min_soft)^2 + nu * (R + C_max_soft - 0.5)^2
    
    Where C_max_soft approximates max(|C_ik|).
    """
    R = params[0]
    centers = params[1:].reshape(N, 3)
    
    # Clip R to a reasonable range to prevent explosion
    if R < 1e-6: R = 1e-6
    if R > 0.15: R = 0.15
    
    # 1. Soft Min of Pairwise Distances
    # d_ij matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid underflow/overflow, shift by min dist
    d_min_actual = np.min(dists)
    if d_min_actual == 0:
        d_min_actual = 1e-9
    exp_terms = np.exp(-beta * (dists - d_min_actual))
    # sum of exp terms
    s = np.sum(exp_terms)
    D_min_soft = d_min_actual - (1.0 / beta) * np.log(s)
    
    # Violation 1: We want D_min_soft >= 2R
    viol_pair = 2 * R - D_min_soft
    # Smooth square of positive part: max(0, x)^2 approx by x^2 if x>0, 0 else 0.
    # For differentiability, just use x^2. The optimizer will push R down if viol is large.
    penalty_pair = viol_pair ** 2
    
    # 2. Soft Max of Coordinate Abs Values
    # We want max(|C_ik|) <= 0.5 - R
    # Let M_soft approx max(|C_ik|)
    abs_centers = np.abs(centers)
    m_max_actual = np.max(abs_centers)
    exp_terms_wall = np.exp(-beta * (abs_centers - m_max_actual))
    s_wall = np.sum(exp_terms_wall)
    M_max_soft = m_max_actual - (1.0 / beta) * np.log(s_wall)
    
    # Violation 2: We want M_max_soft <= 0.5 - R  =>  M_max_soft + R - 0.5 <= 0
    viol_wall = M_max_soft + R - 0.5
    penalty_wall = viol_wall ** 2
    
    # Objective: Minimize -R + penalties
    # We tune mu, nu dynamically or fix them.
    mu = 10.0
    nu = 10.0
    
    return -R + mu * penalty_pair + nu * penalty_wall

def _build_bcc_init():
    """Body Centered Cubic lattice initialization."""
    # BCC unit cell has 2 atoms. 
    # In a cube of side 1, we can fit a grid of BCC cells.
    # Let's try a 4x4x4 grid of BCC cells? That would be 4^3 * 2 = 128 spheres. Too many.
    # 3x3x4 = 36 cells * 2 = 72 spheres. Too few.
    # Let's just use a dense grid and perturb.
    # Or generate BCC points and take the first 100.
    points = []
    # Grid of 4x4x4 BCC cells
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                x = -0.5 + (i + 0.5) * (1.0/4.0)
                y = -0.5 + (j + 0.5) * (1.0/4.0)
                z = -0.5 + (k + 0.5) * (1.0/4.0)
                points.append([x, y, z])
                # Center
                points.append([x + 1.0/8.0, y + 1.0/8.0, z + 1.0/8.0])
                points.append([x - 1.0/8.0, y - 1.0/8.0, z - 1.0/8.0])
                # Actually, BCC is corner + center.
                # My above loop adds 3 points per cell? No, corner is shared.
                # Let's just generate a simple BCC array.
    
    # Simpler BCC generation:
    # Lattice constant a = 1/4.
    # Points at (0,0,0) and (0.5, 0.5, 0.5) in unit cell.
    # Total points in 4x4x4 grid: 4^3 * 2 = 128.
    # We need 100. We can drop 28.
    # Let's generate all 128, compute radius, and drop the 28 that are most "redundant" or just take a subset?
    # Better: Generate 128, then use a greedy removal or just take the first 100 (which might be biased).
    # Let's generate 100 points from a 4x4x4 BCC lattice by skipping some.
    
    pts = []
    # We want roughly equal spacing.
    # Let's use a 5x5x4 grid of simple cubic points, which is 100 points.
    # This is the "Grid" init.
    # For BCC, let's use a 3x3x3 grid of BCC cells (54 points) + extra?
    # Let's stick to the 5x5x4 grid for one init, and a Halton sequence for another.
    
    # Re-implementing BCC init properly:
    # Use a 4x4x4 grid of cells.
    # Each cell has a center and corners.
    # To get exactly 100, we can use a 5x5x4 grid of Simple Cubic, which is a valid init.
    # Let's create a distinct BCC-like init by taking a 4x4x4 BCC (128 pts) and removing 28 pts with largest "redundancy" (closest neighbors).
    
    all_pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                base_x = -0.5 + i * 0.25
                base_y = -0.5 + j * 0.25
                base_z = -0.5 + k * 0.25
                # Corner
                all_pts.append([base_x, base_y, base_z])
                # Center
                all_pts.append([base_x + 0.125, base_y + 0.125, base_z + 0.125])
    
    all_pts = np.array(all_pts)
    # Add small jitter
    all_pts += np.random.uniform(-0.01, 0.01, size=all_pts.shape)
    
    # We have 128 points. We need 100.
    # Compute pairwise distances for all 128.
    dists = np.linalg.norm(all_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Sort by min_dist descending? No, we want to keep points that are well-separated.
    # If we remove a point, its neighbors might get closer? No, removing points only increases min distance for remaining.
    # We want to keep the 100 points that form the "best" packing.
    # Greedy removal: Remove the point whose removal increases the global min radius the least?
    # Or simply: Keep the 100 points that have the largest "local" clearance?
    # Let's just keep the 100 points with the largest min_dist to their nearest neighbor.
    indices = np.argsort(min_dists)[::-1][:100]
    return all_pts[indices]

def _build_grid_init():
    """Simple 5x5x4 Grid."""
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 4)
    centers = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    centers += np.random.uniform(-0.02, 0.02, size=(N, 3))
    return centers

def _build_halton_init():
    """Halton sequence for low discrepancy."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        p = [halton(i, 2), halton(i, 3), halton(i, 5)]
        # Map [0,1] to [-0.4, 0.4] to stay safely inside
        p = [-0.4 + 0.8 * x for x in p]
        points.append(p)
    return np.array(points)

def _optimize_from_initial(init_centers, start_beta=10.0, end_beta=1000.0, num_steps=50):
    """
    Runs the annealing optimization starting from init_centers.
    Returns best centers and radius found.
    """
    best_centers = init_centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Initial R estimate
    current_R = best_radius * 0.9
    
    # Parameters: [R, x1, y1, z1, ..., xN, yN, zN]
    x0 = np.concatenate(([current_R], init_centers.flatten()))
    
    # We will increase beta over time
    betas = np.linspace(start_beta, end_beta, num_steps)
    
    for beta in betas:
        if _time_left() < 1.0:
            break
            
        # Define objective and gradient (numerical or analytical? Numerical is safer for complex surrogates, but slow.
        # Let's use analytical gradient if possible, or finite differences with small step.
        # Given the complexity, let's try L-BFGS-B with numerical gradient first. If too slow, switch to analytical.
        # N=100, 301 params. 100 evals per gradient step. 200 steps * 100 evals * 301 ops = 6M ops. Fast enough.
        
        def objective(params):
            return _soft_min_objective(params, beta)
            
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        x_opt = res.x
        R_opt = x_opt[0]
        centers_opt = x_opt[1:].reshape(N, 3)
        
        # Clip to valid range
        centers_opt = np.clip(centers_opt, -0.5, 0.5)
        
        # Calculate actual radius
        r_actual = _compute_radius(centers_opt)
        
        if r_actual > best_radius:
            best_radius = r_actual
            best_centers = centers_opt
            current_R = r_actual # Update R for next step
        else:
            # If we didn't improve, keep previous best R for stability
            current_R = max(current_R, r_actual)
            
        x0 = np.concatenate(([current_R], best_centers.flatten()))
        
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    inits = [
        _build_grid_init(),
        _build_bcc_init(),
        _build_halton_init()
    ]
    
    best_centers = None
    best_radius = -1.0
    
    # Allocate time to each init
    # We have ~145s. 3 inits. ~45s each.
    time_per_init = 40.0
    
    for i, init in enumerate(inits):
        if _time_left() < time_per_init:
            break
            
        # Set a local deadline for this init
        local_deadline = time.time() + time_per_init
        
        # Monkey patch _time_left to respect local deadline? 
        # Better: pass deadline to _optimize_from_initial?
        # For now, _optimize_from_initial uses global _time_left.
        # We can just rely on the global check inside the loop.
        
        centers, radius = _optimize_from_initial(init)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
            
        # If we have time, try a high-beta polish on the best result so far
        if _time_left() > 5.0 and best_centers is not None:
            # Run a short, high-beta polish
            centers_polished, radius_polished = _optimize_from_initial(
                best_centers, 
                start_beta=500.0, 
                end_beta=2000.0, 
                num_steps=10
            )
            if radius_polished > best_radius:
                best_radius = radius_polished
                best_centers = centers_polished

    if best_centers is None:
        # Fallback
        best_centers = inits[0]
        best_radius = _compute_radius(best_centers)
        
    return best_centers
