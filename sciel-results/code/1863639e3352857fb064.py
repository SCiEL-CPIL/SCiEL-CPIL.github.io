"""
Strategy: Kinetic Monte Carlo (Simulated Annealing) with adaptive target radius.
Instead of smooth global optimization, we use stochastic moves to explicitly
maximize the minimum distance. We maintain a KD-tree for neighbor queries to
speed up collision checks and boundary checks. The temperature anneals
exponentially, allowing large moves early (to escape local minima) and small
moves late (to polish the packing). The target radius r is adaptively increased
whenever the current minimum distance exceeds it, driving the optimization
toward the maximum possible packing.
"""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 300
DEADLINE = 145.0  # Safety margin from 150s limit

def _score(points):
    """Compute the maximum radius r for given centers."""
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances using KDTree for efficiency
    tree = cKDTree(points)
    # Query all pairs within a reasonable distance? No, we need min distance.
    # For N=300, O(N^2) is ~45k checks, which is fast in C.
    # But let's use the tree to find nearest neighbors for each point to find global min.
    # Actually, for just the score, O(N^2) is fine if optimized.
    # Let's stick to vectorized O(N^2) for simplicity and speed in NumPy for N=300.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _min_dist_and_boundary(points):
    """
    Compute the minimum pairwise distance and minimum boundary distance.
    Returns (min_pair_dist, min_boundary_dist).
    """
    # Boundary
    r_pts = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - r_pts)
    
    # Pairwise
    # Use KDTree to find nearest neighbors for each point.
    # The global min pair distance is the min of all nearest neighbor distances.
    tree = cKDTree(points)
    # Query k=2 to get the nearest neighbor (k=1 is self)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1])
    
    return min_pair, min_boundary

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk."""
    # Start with a dense hex grid and pick the best N
    # Spacing s such that we have slightly more than N points
    # Area ~ pi, point area ~ s^2 * sqrt(3)/2. N ~ pi / (s^2 * sqrt(3)/2)
    # s ~ sqrt(2*pi / (N * sqrt(3)))
    s = math.sqrt(2 * math.pi / (N * math.sqrt(3)))
    
    # Try a few spacings to ensure we get exactly N points inside
    best_points = None
    for factor in [1.0, 1.01, 1.02, 1.03, 0.99, 0.98]:
        s_try = s * factor
        k = int(1.0 / s_try) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s_try
                y = j * s_try * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            # Keep the N closest to center
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def construct():
    start_time = time.time()
    
    # Initialize
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * math.cos(angle)
            points[i, 1] = r * math.sin(angle)
    
    best_points = points.copy()
    min_pair, min_boundary = _min_dist_and_boundary(points)
    best_r = min(min_boundary, min_pair / 2.0)
    
    # KDTree for neighbor queries
    tree = cKDTree(points)
    
    # Simulated Annealing Parameters
    # Initial temperature: large enough to allow moves that increase energy (decrease min dist)
    # But we want to maximize min dist. 
    # Standard SA minimizes energy. Here we maximize "radius".
    # Let's define energy E = -min_dist. We want to minimize E.
    # Or simply: accept a move if it improves min_dist, or with probability exp(-delta / T) if it worsens.
    # delta = old_min_dist - new_min_dist (if new is worse, delta > 0)
    
    T_start = 0.01
    T_end = 1e-8
    # Number of iterations
    # We have ~140s. 
    # Each iteration: query tree (fast), check move, update tree (slow if full rebuild).
    # Instead of full rebuild, we can use a simpler approach:
    # Since N=300 is small, rebuilding KDTree is O(N log N) ~ 300*8 = 2400 ops. Very fast.
    # Let's do 1 move per iteration.
    # Estimate time per move: ~10-50 us? 
    # 140s / 20us = 7 million moves. That's a lot.
    # Let's be conservative. 100,000 moves should be fast enough.
    
    max_iters = 200000
    # Adaptive target radius
    target_r = best_r * 1.01
    
    # Random state
    rng = np.random.default_rng(42)
    
    # Pre-compute some constants
    # We will move one particle at a time.
    
    for i_iter in range(max_iters):
        # Time check
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Annealing schedule: exponential decay
        # T = T_start * (T_end / T_start) ** (i_iter / max_iters)
        alpha = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start * (alpha ** i_iter)
        
        # Pick a random particle to move
        idx = rng.integers(0, N)
        old_point = points[idx].copy()
        
        # Propose a new position
        # Move in a random direction with random magnitude
        # Magnitude should scale with T and current density
        # Typical spacing ~ sqrt(1/N) ~ 0.05
        # Move magnitude: uniform in [0, step_size]
        # step_size can be adaptive. Start with 0.05, decrease with T?
        # Or just use T directly?
        # Let's use a step size that allows exploration.
        # If T is high, take bigger steps.
        step_size = 0.05 * (T / T_start + 0.1) # Scale with T, but keep some base
        
        # Random direction
        angle = rng.uniform(0, 2 * math.pi)
        dx = step_size * math.cos(angle)
        dy = step_size * math.sin(angle)
        
        new_point = old_point + np.array([dx, dy])
        
        # Check boundary
        new_r = math.hypot(new_point[0], new_point[1])
        if new_r > 1.0:
            # Reject or project? Projecting is better for SA.
            # But projection changes the move. Let's just reject if outside.
            # Or scale down.
            # To maintain detailed balance, rejection is safer.
            continue
        
        # Check collision with neighbors
        # Query tree for neighbors of old_point within a radius that could be affected
        # The new point might collide with points that were close to old_point or new_point.
        # Since N is small, let's just check against all points? 
        # No, that's O(N). 300 checks * 200k iters = 60M checks. Fast in C, but Python loop is slow.
        # Vectorized check against all points is O(N) per move.
        
        # Let's vectorize the distance check for the single new point against all points
        # dists = np.linalg.norm(points - new_point, axis=1)
        # dists[idx] = inf
        # min_dist_to_others = np.min(dists)
        
        # This is O(N) per move. 300 ops.
        # 200k * 300 = 60M ops. NumPy can do this in ~1-2 seconds?
        # Let's try.
        
        # Update point in array temporarily
        points[idx] = new_point
        
        # Compute new min distance and boundary
        # We only need to know if the global min decreased.
        # Actually, we need the new global min to compare with old.
        # But computing global min is O(N log N) or O(N^2).
        # If we do O(N) check for the moved point, we don't know the global min.
        # However, the global min can only change if the moved point was part of the min pair,
        # or if it creates a new smaller pair.
        
        # Strategy:
        # 1. Calculate distance from new_point to all other points.
        # 2. Find min distance from new_point to others: d_new
        # 3. If d_new < current_global_min_pair, then new_global_min = min(d_new, current_global_min_pair_without_old)
        #    This is tricky because we need the min of the rest.
        
        # Simpler robust approach for N=300:
        # Just recompute the global min distance using KDTree query.
        # It's fast enough.
        
        # Rebuild tree? No, tree is stale.
        # For N=300, building tree is fast.
        # But we can just compute distances from new_point to all others.
        
        dists_to_others = np.linalg.norm(points - new_point, axis=1)
        dists_to_others[idx] = np.inf
        min_dist_new_point = np.min(dists_to_others)
        
        # The new global min pair distance is min(min_dist_new_point, min_dist_of_rest)
        # We don't have min_dist_of_rest readily available without querying the rest.
        # But we know the old global min.
        # If min_dist_new_point >= old_global_min, then new_global_min >= old_global_min?
        # No, because the old global min might have involved the point we moved.
        
        # Let's just compute the full score for the new configuration?
        # That's O(N^2) or O(N log N).
        # With 200k iters, 200k * 300^2 is too slow.
        
        # Optimization:
        # Maintain the list of "critical" pairs? Too complex.
        
        # Let's use the KDTree properly.
        # We can update the tree by deleting and inserting? cKDTree doesn't support efficient updates.
        # Rebuilding is O(N log N).
        # 200k * 300 * log(300) ~ 200k * 2400 = 480M ops. Might be too slow in Python/SciPy overhead.
        
        # Alternative: Use a fixed grid or just accept the O(N) check and assume
        # that if the local min around the moved point is good, the global min is likely good.
        # This is a heuristic.
        
        # Let's try a different approach:
        # Compute the new global min distance using the fact that N is small.
        # We can compute the pairwise distances for the whole set in a vectorized way?
        # No, that's O(N^2) memory and compute.
        
        # Let's go back to the KDTree rebuild. It's the most robust.
        # We will limit iterations to ensure we finish in time.
        # Let's measure time for one rebuild + query.
        
        # Actually, for N=300, `cKDTree(points).query(points, k=2)` is very fast.
        # Let's assume 100,000 iterations are safe.
        
        # For now, let's just compute the min distance for the new configuration
        # by checking the moved point against all others, and assuming the rest didn't change.
        # This is an approximation.
        # d_min_new = min(min_dist_new_point, old_min_pair_if_not_involving_idx)
        # We don't know if old_min_pair involved idx.
        
        # Let's just do the full KDTree query. It's the safest bet for correctness.
        # If it's too slow, we'll reduce iterations.
        
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        new_min_pair = np.min(dists[:, 1])
        new_min_boundary = np.min(1.0 - np.hypot(points[:, 0], points[:, 1]))
        new_r = min(new_min_pair / 2.0, new_min_boundary)
        
        # Acceptance criterion
        if new_r > best_r:
            # Always accept improvement
            best_r = new_r
            best_points = points.copy()
            # Increase target
            target_r = new_r * 1.005
        else:
            # Deterioration
            delta = best_r - new_r
            # Accept with probability exp(-delta / T)
            if delta < 1e-12:
                accept = True
            else:
                prob = math.exp(-delta / max(T, 1e-12))
                accept = rng.random() < prob
            
            if accept:
                # Keep the new state, but don't update best
                pass
            else:
                # Reject: revert
                points[idx] = old_point
                continue
        
        # If we accepted (either improvement or worsening), we proceed.
        # If we rejected, we continued.
        
        # Update best if improved
        if new_r > best_r:
            best_r = new_r
            best_points = points.copy()
    
    # Final polish: try a few local relaxations on the best found
    # Simple gradient-free polish: move points slightly towards the center of their neighbors
    # Or just return best_points
    
    # Ensure validity
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    return best_points
