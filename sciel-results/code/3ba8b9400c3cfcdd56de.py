"""Program for packing 200 equal circles in a unit square using QP-like optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time


def _hexagonal_lattice(n, side=1.0):
    """Generate a hexagonal lattice of approximately n points in a unit square centered at origin."""
    points = []
    # Hexagonal lattice: spacing dx, dy = dx * sqrt(3)/2
    # We'll try different spacings to fit ~n points
    best_points = None
    best_count = 0
    
    # Try various spacings
    for trial in range(50):
        dx = 0.01 + trial * 0.005
        dy = dx * np.sqrt(3) / 2
        
        # Generate lattice points
        xs = np.arange(-0.5 + dx/2, 0.5, dx)
        ys = np.arange(-0.5 + dy/2, 0.5, dy)
        
        pts = []
        for i, y in enumerate(ys):
            x_offset = (dx / 2) if i % 2 == 0 else 0
            for x in xs:
                px = x + x_offset
                py = y
                if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                    pts.append((px, py))
        
        pts = np.array(pts)
        if len(pts) >= n:
            # Take first n points
            pts = pts[:n]
            if len(pts) > best_count:
                best_count = len(pts)
                best_points = pts.copy()
        
        if best_points is not None and len(best_points) >= n:
            break
    
    if best_points is None or len(best_points) < n:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:n]
    
    return best_points


def _objective_and_constraints(pos, n):
    """
    Compute the minimum distance and check constraints.
    Returns (min_dist, is_valid)
    """
    pos = pos.reshape(-1, 2)
    
    # Check if all points are inside the square
    if np.any(pos < -0.5 + 1e-10) or np.any(pos > 0.5 - 1e-10):
        return 0.0, False
    
    # Compute all pairwise distances
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                min_dist = d
    
    # Distance to walls
    wall_dists = []
    for i in range(n):
        wall_dists.append(0.5 - pos[i, 0])
        wall_dists.append(0.5 + pos[i, 0])
        wall_dists.append(0.5 - pos[i, 1])
        wall_dists.append(0.5 + pos[i, 1])
    
    min_wall_dist = min(wall_dists)
    
    # The radius is half the minimum of pairwise distances and twice the wall distance
    # Actually: r = min(min_pairwise/2, min_wall_dist)
    # Wait, let me re-read: r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    # So r = min(min_pairwise_dist / 2, min_wall_dist)
    
    r = min(min_dist / 2.0, min_wall_dist)
    
    return r, True


def _optimize_from_start(start_pos, max_iter=10000, tol=1e-10):
    """
    Optimize the packing using gradient-based method.
    We maximize the minimum radius by iteratively pushing points apart.
    """
    n = len(start_pos)
    pos = start_pos.copy().astype(float)
    
    # We'll use a simple iterative method:
    # 1. Find the constraint that is most violated (smallest distance)
    # 2. Move points to increase that distance
    
    best_r = _objective_and_constraints(pos, n)[0]
    
    for iteration in range(max_iter):
        # Compute all pairwise distances
        min_dist = np.inf
        min_pair = (-1, -1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        
        # Compute wall distances
        min_wall_dist = np.inf
        min_wall_point = -1
        min_wall_axis = -1
        for i in range(n):
            for axis in range(2):
                if axis == 0:
                    d_wall = 0.5 - pos[i, 0]
                    if d_wall < min_wall_dist:
                        min_wall_dist = d_wall
                        min_wall_point = i
                        min_wall_axis = 0
                else:
                    d_wall = 0.5 - pos[i, 1]
                    if d_wall < min_wall_dist:
                        min_wall_dist = d_wall
                        min_wall_point = i
                        min_wall_axis = 1
        
        # Current radius
        current_r = min(min_dist / 2.0, min_wall_dist)
        
        # Check for improvement
        if current_r > best_r + 1e-15:
            best_r = current_r
        
        # If we can't improve further, break
        if min_dist / 2.0 < min_wall_dist:
            # Pairwise constraint is active
            i, j = min_pair
            # Move points i and j apart
            diff = pos[i] - pos[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                # Random perturbation
                pos[i] += np.array([0.001, 0])
                pos[j] -= np.array([0.001, 0])
            else:
                direction = diff / norm
                step = 0.01
                pos[i] += direction * step
                pos[j] -= direction * step
        else:
            # Wall constraint is active
            i = min_wall_point
            axis = min_wall_axis
            if axis == 0:
                pos[i, 0] -= 0.01
            else:
                pos[i, 1] -= 0.01
        
        # Clamp to square
        pos = np.clip(pos, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Check convergence
        if iteration % 100 == 0 and current_r < best_r - 1e-8:
            # Rollback? For simplicity, just continue
            pass
    
    return pos, best_r


def construct():
    """Construct the optimal packing of 200 circles in a unit square."""
    # Start with a hexagonal lattice
    start_pos = _hexagonal_lattice(N)
    
    # Ensure we have exactly N points
    if len(start_pos) < N:
        # Pad with grid points
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        grid = grid[:N]
        # Combine: use hexagonal for first len(start_pos), grid for rest
        remaining = N - len(start_pos)
        extra = grid[len(start_pos):len(start_pos)+remaining] if len(grid) > len(start_pos) else grid[:remaining]
        start_pos = np.vstack([start_pos, extra]) if len(start_pos) > 0 else extra
    
    start_pos = start_pos[:N]
    
    # Optimize
    optimized_pos, best_r = _optimize_from_start(start_pos, max_iter=5000)
    
    # Final check and return
    optimized_pos = np.clip(optimized_pos, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return optimized_pos
# EVOLVE-BLOCK-END
