"""Packing 50 equal circles in a unit square using contact-graph constraint solving."""
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

N = 50

def _radius(pts):
    """Compute the radius for a set of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    r_wall = np.min(wall)
    # Pairwise distances
    # Compute all pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_wall, r_pair)

def _init_hexagonal():
    """Initialize with a hexagonal-like pattern."""
    # Use a grid with slight perturbation
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    # Add small random perturbation
    np.random.seed(42)
    pts += np.random.normal(0, 0.01, size=pts.shape)
    return pts

def _init_random():
    """Random initialization."""
    np.random.seed(0)
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def _contact_graph(pts, r, k=4):
    """Build a contact graph: for each point, connect to its k nearest neighbors."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    graph = []
    for i in range(n):
        # Find k nearest neighbors
        nn_idx = np.argsort(dist[i])[:k]
        for j in nn_idx:
            if i < j:
                graph.append((i, j))
    # Remove duplicates
    graph = list(set(graph))
    return graph

def _solve_constraints(pts, r, max_iter=500, tol=1e-10):
    """
    Solve for positions satisfying:
    - ||p_i - p_j|| >= 2r for all contacts (i,j)
    - p_i within [-0.5+r, 0.5-r]^2
    
    Use iterative projection / gradient descent on the constraint violations.
    """
    n = len(pts)
    pts = pts.copy()
    r_target = r
    
    # Build contact graph based on current positions
    graph = _contact_graph(pts, r_target, k=5)
    
    for iteration in range(max_iter):
        # Compute violations and forces
        forces = np.zeros_like(pts)
        max_violation = 0.0
        
        # Pairwise constraints
        for (i, j) in graph:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                # Random perturbation
                d = np.random.normal(0, 1e-6, size=2)
                dist = np.linalg.norm(d)
            
            min_dist = 2 * r_target
            if dist < min_dist:
                violation = min_dist - dist
                max_violation = max(max_violation, violation)
                # Force to push apart
                force = (d / dist) * violation
                forces[i] += force
                forces[j] -= force
        
        # Wall constraints
        for i in range(n):
            for axis in range(2):
                val = pts[i, axis]
                min_val = -0.5 + r_target
                max_val = 0.5 - r_target
                
                if val < min_val:
                    violation = min_val - val
                    max_violation = max(max_violation, violation)
                    forces[i, axis] += violation
                elif val > max_val:
                    violation = val - max_val
                    max_violation = max(max_violation, violation)
                    forces[i, axis] -= violation
        
        if max_violation < tol:
            break
        
        # Update positions
        step_size = 0.5
        pts += step_size * forces
    
    return pts

def _local_optimize(pts, r, max_iter=200):
    """Local optimization: try to increase radius slightly and solve constraints."""
    best_pts = pts.copy()
    best_r = r
    
    for _ in range(max_iter):
        # Try to increase radius
        new_r = best_r * 1.001
        # Solve constraints
        new_pts = _solve_constraints(pts, new_r, max_iter=100)
        actual_r = _radius(new_pts)
        
        if actual_r > best_r:
            best_r = actual_r
            best_pts = new_pts
        else:
            # Try smaller step
            new_r = best_r * 1.0001
            new_pts = _solve_constraints(pts, new_r, max_iter=50)
            actual_r = _radius(new_pts)
            if actual_r > best_r:
                best_r = actual_r
                best_pts = new_pts
            else:
                break
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Multiple initialization strategies
    inits = []
    
    # Hexagonal-like init
    inits.append(_init_hexagonal())
    
    # Random inits
    for seed in range(5):
        np.random.seed(seed)
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Grid init
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    best_pts = None
    best_r = 0.0
    
    for init_idx, pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Get initial radius
        r = _radius(pts)
        
        # Local optimization
        if time.time() > deadline:
            break
        
        pts_opt, r_opt = _local_optimize(pts, r, max_iter=50)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt
    
    # If we have time, do more iterations on the best solution
    if best_pts is not None and time.time() < deadline - 10:
        pts_final, r_final = _local_optimize(best_pts, best_r, max_iter=100)
        if r_final > best_r:
            best_r = r_final
            best_pts = pts_final
    
    # Final check
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Ensure valid output
    best_pts = np.asarray(best_pts, dtype=np.float64)
    if best_pts.shape != (N, 2) or not np.all(np.isfinite(best_pts)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_pts
