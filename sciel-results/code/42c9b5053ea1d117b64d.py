"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 21
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_objective(x, N_circles):
    """
    x: (2N, ) array of [x_1, y_1, ..., x_N, y_N]
    We pack circles in a rectangle of width w and height h with w + h = 2 (perimeter 4).
    We use a fixed aspect ratio optimization approach or optimize w as well?
    The problem says "variable rectangle of perimeter 4".
    Let w be the width. Then h = 2 - w.
    We need to maximize sum of radii.
    Radii are determined by the closest constraints:
    r_i = min( x_i, w - x_i, y_i, h - y_i, min_{j != i} (dist(i,j) - r_j) ... )
    Actually, for a given configuration of centers and a given w, the maximum radius for circle i is:
    r_i = min( x_i, w - x_i, y_i, 2 - w - y_i, min_{j != i} (dist(i,j) - r_j) ) -- this is coupled.
    
    Better approach:
    Optimize over centers (x_i, y_i) and a single width w (0 < w < 2).
    The radius r_i is the minimum distance to the boundary and to other circles (accounting for their radii).
    This is a non-smooth problem.
    
    Alternative:
    Use a "push" method or local search.
    Or, use a grid of aspect ratios w, and for each w, run a local optimization of centers to maximize sum of radii.
    
    Let's try a local search / gradient-free optimization for a fixed w, then scan w.
    Or, treat w as part of the optimization vector.
    
    Let's define the problem as:
    Maximize sum(r_i)
    Subject to:
    x_i >= r_i
    x_i <= w - r_i
    y_i >= r_i
    y_i <= h - r_i  (h = 2 - w)
    dist(i, j) >= r_i + r_j
    
    This is a non-convex problem.
    
    Strategy:
    1. Initialize with a grid packing for a specific w.
    2. Use a local search to improve the sum of radii.
       - At each step, compute the "bottleneck" for each circle.
       - Move circles to increase the minimum radius.
    
    A simpler heuristic:
    For a given w, place circles in a grid, then iterate:
    - Compute r_i for each circle as the min distance to boundary and to other circles (assuming others are fixed).
    - Move each circle slightly towards the center of its available space, or away from the closest constraint.
    
    Let's try a randomized local search with multiple restarts.
    
    Objective function for a fixed w:
    Given centers, compute the maximum possible radius for each circle assuming others are fixed? No, that's not right.
    
    Let's use a different formulation:
    We want to maximize the minimum radius R such that all circles have radius >= R?
    No, we want to maximize the SUM of radii.
    
    Let's use a local search that maximizes the sum of radii.
    For a given configuration, the radius of circle i is:
    r_i = min( x_i, w - x_i, y_i, h - y_i, min_{j != i} (dist(i,j) - r_j) )
    This is implicit.
    
    Approximation:
    Iterate:
    1. Set r_i = min distance to boundary.
    2. For each pair, if dist(i,j) < r_i + r_j, reduce the larger one (or both) to make dist(i,j) = r_i + r_j.
    3. Repeat until stable.
    
    This gives a valid packing. We then move centers to increase the sum.
    
    Let's implement a function `compute_radii(centers, w)` that returns the maximal radii for a given center configuration.
    """
    h = 2.0 - w
    n = len(centers)
    # Initial radii from boundary
    r = np.minimum(np.minimum(centers[:, 0], w - centers[:, 0]), 
                   np.minimum(centers[:, 1], h - centers[:, 1]))
    # Iteratively enforce circle-circle constraints
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < r[i] + r[j]:
                    # Reduce radii to fit
                    # Simple approach: reduce both proportionally or set sum to d
                    # Better: reduce the one with more "room" or just split the deficit
                    deficit = r[i] + r[j] - d
                    if deficit > 1e-9:
                        # Reduce r[i] and r[j]
                        # Keep it simple: reduce both by deficit/2, but not below 0
                        # Actually, we want to maximize sum, so we should reduce the one that allows more growth later?
                        # For now, just make them tangent.
                        # Let's reduce the larger one first? Or split?
                        # Splitting is symmetric.
                        reduce_i = deficit / 2.0
                        reduce_j = deficit / 2.0
                        # Ensure non-negative
                        if r[i] < reduce_i:
                            reduce_j += (reduce_i - r[i])
                            reduce_i = r[i]
                        if r[j] < reduce_j:
                            reduce_i += (reduce_j - r[j])
                            reduce_j = r[j]
                        r[i] -= reduce_i
                        r[j] -= reduce_j
                        changed = True
        if not changed:
            break
    return r

def _local_search(centers, w, iterations=2000, step_size=0.01, cooling=0.99):
    """
    Local search to maximize sum of radii.
    """
    h = 2.0 - w
    n = len(centers)
    current_centers = centers.copy()
    current_r = _compute_radii(current_centers, w)
    current_score = np.sum(current_r)
    
    best_centers = current_centers.copy()
    best_score = current_score
    
    for it in range(iterations):
        if time.time() > DEADLINE - 5:
            break
            
        # Pick a random circle and a random direction
        i = np.random.randint(n)
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_centers = current_centers.copy()
        new_centers[i, 0] += dx
        new_centers[i, 1] += dy
        
        # Check bounds
        if new_centers[i, 0] < 0 or new_centers[i, 0] > w or new_centers[i, 1] < 0 or new_centers[i, 1] > h:
            continue
            
        new_r = _compute_radii(new_centers, w)
        new_score = np.sum(new_r)
        
        if new_score > current_score:
            current_centers = new_centers
            current_r = new_r
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        else:
            # Simulated annealing style acceptance? Or just hill climbing?
            # Hill climbing is safer.
            pass
            
        # Decrease step size
        step_size *= cooling
        
    return best_centers, best_score

def construct():
    start_time = time.time()
    best_score = -1
    best_result = None
    
    # Scan aspect ratios
    # w is width, h = 2 - w. w in (0, 2).
    # Symmetric around w=1.
    for w in np.linspace(0.5, 1.5, 20):
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize with grid
        side = int(np.ceil(np.sqrt(N)))
        # Scale grid to fit in w x h
        # Grid spacing
        dx = w / side
        dy = (2.0 - w) / side
        
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) >= N:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        
        # Local search
        res_centers, score = _local_search(centers, w, iterations=500, step_size=0.05, cooling=0.995)
        
        if score > best_score:
            best_score = score
            best_result = (res_centers, w)
            
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    centers, w = best_result
    radii = _compute_radii(centers, w)
    
    # Format output: (x, y, r)
    result = np.column_stack((centers, radii))
    return result
# EVOLVE-BLOCK-END
