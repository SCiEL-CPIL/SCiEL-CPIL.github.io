"""Model-and-optimize approach for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve, lsqr

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding itself)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _initial_grid():
    """Create initial hexagonal packing layout with slight jitter."""
    # Hexagonal arrangement: alternate rows offset by half-spacing
    # For 50 circles, use 7 rows with varying counts
    # Row pattern for hex packing: 8,7,8,7,8,7,5 = 50
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    
    points = []
    total_rows = len(row_counts)
    # Vertical spacing for hex packing
    vert_spacing = 0.9 / (total_rows - 1) if total_rows > 1 else 0
    
    for row_idx, count in enumerate(row_counts):
        y = -0.45 + row_idx * vert_spacing
        # Horizontal spacing
        horiz_spacing = 0.9 / count if count > 1 else 0
        x_start = -0.45 + (0.9 - horiz_spacing * (count - 1)) / 2
        # Offset alternating rows
        if row_idx % 2 == 1:
            x_offset = horiz_spacing / 2
        else:
            x_offset = 0
        
        for i in range(count):
            x = x_start + i * horiz_spacing + x_offset
            points.append([x, y])
    
    points = np.array(points[:N])
    
    rng = np.random.default_rng(12345)
    points = points + rng.normal(0, 0.008, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _contact_graph_optimize(centers, time_budget=120.0):
    """
    Structural change: Rigid-body assembly optimization via contact graph.
    Iteratively identifies active constraints (contacts) and solves a sparse
    linear system to find a displacement that increases the minimum radius.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Parameters
    max_iters = 200
    tolerance = 1e-9
    step_scale = 1.0
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        c = best_centers.copy()
        r_curr = best_r
        
        # 1. Identify active constraints
        # We consider a constraint active if its slack is very small.
        # We want to find a direction d such that for all active constraints,
        # the derivative of the distance in direction d is positive (pushing out).
        
        # Wall constraints:
        # dist_i_wall = 0.5 - |c_i|
        # We want to increase this.
        # Gradient of (0.5 - |x|) wrt x is -sign(x).
        # We want to move in direction -grad = sign(x).
        
        # Pair constraints:
        # dist_ij = ||c_i - c_j||
        # We want to increase this.
        # Gradient wrt c_i is (c_i - c_j) / ||c_i - c_j||.
        # We want to move c_i in this direction and c_j in opposite.
        
        # Let's collect all "near-min" constraints.
        # Threshold: constraints within 1% of the current minimum radius.
        threshold = r_curr * 1.01
        
        # Build sparse linear system A * delta = b
        # We want to maximize the minimum increase in slack.
        # Let s_k be the slack of constraint k.
        # s_k_new = s_k + grad_k . delta
        # We want s_k_new >= r_curr + dr
        # To maximize dr, we can solve:
        # Maximize dr
        # Subject to: grad_k . delta >= dr - s_k  for all active k
        #              ||delta|| <= 1 (normalize step)
        
        # This is a linear program, but we can approximate with a least-squares
        # approach where we try to equalize the slacks of the active constraints.
        # Or simpler: Find delta that maximizes min_k (grad_k . delta).
        
        # Let's use a simpler iterative scheme:
        # 1. Find all constraints with slack < threshold.
        # 2. Compute the "force" vector for each.
        # 3. Sum the forces to get a net displacement direction.
        # 4. Normalize and take a step.
        
        tree = cKDTree(c)
        # Get neighbors to find pair constraints
        dists, idxs = tree.query(c, k=10) # Check up to 10 neighbors
        
        wall_slacks = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
        
        # Active indices for walls
        active_walls = np.where(wall_slacks < threshold)[0]
        
        # Active pairs
        active_pairs = []
        for i in range(N):
            for j_idx in range(1, 10):
                j = idxs[i, j_idx]
                if j < i:
                    continue
                d = dists[i, j_idx]
                if d / 2.0 < threshold:
                    active_pairs.append((i, j, d))
        
        # If no active constraints, we are likely at a local max or isolated.
        if len(active_walls) == 0 and len(active_pairs) == 0:
            break
            
        # Construct the force vector
        force = np.zeros((N, 2))
        
        # Wall forces
        for i in active_walls:
            # Push away from the closest wall
            dx = 0.5 - c[i, 0]
            dy = 0.5 - c[i, 1]
            if dx < dy:
                # Closest to right or left wall?
                if c[i, 0] > 0:
                    force[i, 0] -= 1.0 # Push left
                else:
                    force[i, 0] += 1.0 # Push right
            else:
                if c[i, 1] > 0:
                    force[i, 1] -= 1.0 # Push down
                else:
                    force[i, 1] += 1.0 # Push up
        
        # Pair forces
        for i, j, d in active_pairs:
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
                # Push i away from j, j away from i
                force[i] += unit
                force[j] -= unit
        
        # Normalize force
        norm_f = np.linalg.norm(force)
        if norm_f < 1e-9:
            break
        force /= norm_f
        
        # Determine step size
        # We want to move such that the minimum slack increases.
        # A safe step is proportional to the minimum slack deficit.
        # Let's try a fixed small step and adjust if radius decreases.
        step_size = 0.005 * step_scale
        
        new_centers = c + force * step_size
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers
            step_scale = min(step_scale * 1.05, 2.0) # Increase step if improving
        else:
            step_scale = max(step_scale * 0.8, 0.1) # Decrease step if failing
            
            # If step is too small, try a random perturbation to escape
            if step_scale < 0.15:
                rng = np.random.default_rng(it)
                perturbation = rng.normal(0, 0.001, (N, 2))
                new_centers = c + perturbation
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers
                    step_scale = 0.5
                else:
                    # Stuck, break to avoid wasting time
                    if it > 50:
                        break

    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration: hexagonal packing
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    # Structural Change: Use Contact Graph Optimization
    # This replaces the previous SA + L-BFGS + Binary Search pipeline.
    # It directly manipulates the contact graph to satisfy constraints.
    
    if time.time() < deadline - 5:
        best_centers, best_radius = _contact_graph_optimize(
            best_centers, time_budget=min(130.0, deadline - time.time() - 5)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
