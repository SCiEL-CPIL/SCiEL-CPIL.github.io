"""
Strategy: Directed Local Search (DLS) with Simulated Annealing Restarts.
Instead of global gradient-based optimization (L-BFGS-B), we use a local search
algorithm that:
1. Identifies the most violated constraint (minimum distance or boundary).
2. Moves the involved points directly to satisfy that constraint (greedy repair).
3. Uses random perturbations (simulated annealing) to escape local minima.
This is a different design family: discrete/local geometric repair vs. smooth global optimization.
"""
import numpy as np
import time
import math
import random

N = 300
DEADLINE = 118.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk."""
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None:
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

def _compute_radius(points):
    """Compute the maximum valid radius for the given points."""
    # Boundary constraint
    dists_origin = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - dists_origin)
    
    # Pairwise constraint
    # For efficiency, we don't compute all pairs every time in the tight loop,
    # but for the final score we do.
    # Here we just return a quick estimate or the min boundary if we assume pairs are okay?
    # No, we need the true score.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _local_search(points, max_iter=2000, time_limit=10.0):
    """
    Perform a local search to improve the packing.
    Strategy:
    1. Find the current radius r.
    2. Find the most violated constraint (pair or boundary) that limits r.
    3. Move the involved points to alleviate the violation.
    4. If stuck, apply a random perturbation (annealing).
    """
    start_time = time.time()
    pts = points.copy()
    n = N
    
    # Pre-compute initial radius
    r_current = _compute_radius(pts)
    
    # We will try to increase r_current slightly and fix violations
    # Or simply minimize the "violation" for a fixed target r?
    # Let's use a target r that is slightly higher than current.
    
    # To be robust, let's do a "push-apart" loop.
    # If dist(i,j) < 2*r, push them apart.
    # If dist(i,0) > 1-r, push i inward.
    
    # We iterate on a target radius.
    # Start with r_target = r_current
    # If we can satisfy constraints, increase r_target.
    
    # Actually, a simpler heuristic:
    # Just run a force-directed relaxation with very strong repulsion
    # but using a "greedy" update rule.
    
    # Let's implement a specific local search:
    # 1. Calculate all pair distances and boundary distances.
    # 2. Find min distance d_min.
    # 3. r = d_min / 2 or boundary limit.
    # 4. If pair limit, take the pair (i,j) with min dist. Move i and j apart by epsilon.
    # 5. If boundary limit, take point i with max dist. Move i inward by epsilon.
    # 6. Repeat.
    
    # To escape local minima, we add random jitter.
    
    best_r = r_current
    best_pts = pts.copy()
    
    r_target = r_current
    stuck_count = 0
    last_r = r_current
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Check for coincident points or invalid states
        # Compute current constraints
        dists_origin = np.hypot(pts[:, 0], pts[:, 1])
        boundary_viol = np.max(dists_origin - (1.0 - r_target))
        
        # Pairwise
        # For speed, we only check if we need to. But we need to find the worst pair.
        # N=300, 300^2 = 90k ops. 2000 iters = 180M ops. Might be slow in Python loops.
        # Let's use NumPy vectorization for the search of the min pair.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        min_dist = math.sqrt(min_dist_sq)
        
        # Determine limiting constraint
        limit_by_boundary = (dists_origin.max() > 1.0 - r_target)
        limit_by_pair = (min_dist < 2.0 * r_target)
        
        if not limit_by_boundary and not limit_by_pair:
            # Feasible for r_target. Try to increase r_target.
            r_target += 0.0001
            stuck_count = 0
            # Update best if improved
            actual_r = min(np.min(1.0 - dists_origin), min_dist / 2.0)
            if actual_r > best_r:
                best_r = actual_r
                best_pts = pts.copy()
            continue
            
        # If violated, we need to move points.
        
        if limit_by_boundary:
            # Move the point furthest out inward
            idx = np.argmax(dists_origin)
            p = pts[idx]
            norm = np.linalg.norm(p)
            if norm > 1e-9:
                # Move inward
                step = 0.01
                pts[idx] = p * (1.0 - step / norm)
            else:
                pts[idx] = np.array([0.0, 0.0])
                
        elif limit_by_pair:
            # Find the pair
            min_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
            i, j = min_idx
            if i == j: continue
            pi = pts[i]
            pj = pts[j]
            d_vec = pi - pj
            d = np.linalg.norm(d_vec)
            if d < 1e-9:
                # Random push
                pts[i] += np.array([0.01, 0.0])
                pts[j] -= np.array([0.01, 0.0])
            else:
                # Push apart
                unit_vec = d_vec / d
                step = 0.01
                pts[i] = pi + 0.5 * step * unit_vec
                pts[j] = pj - 0.5 * step * unit_vec
        
        # Check progress
        # We don't recompute full score every step for speed, just check if we are still violating
        # But we need to know if we are stuck to trigger annealing.
        
        # Simple stagnation detection: if we haven't increased r_target in a while?
        # Or just add random noise every X iterations.
        if it % 100 == 99:
            # Random perturbation (Annealing)
            # Pick a random point and move it randomly
            idx = random.randint(0, n-1)
            noise = np.random.normal(0, 0.05, 2)
            pts[idx] += noise
            # Ensure it stays in disk roughly
            norm = np.linalg.norm(pts[idx])
            if norm > 1.0:
                pts[idx] = pts[idx] / norm * 0.99

    # Final polish: compute exact radius
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
    # Ensure best_pts is the one that gives best_r? 
    # Actually, best_pts is updated only when feasible for r_target.
    # But the local search might have moved points to satisfy constraints for a higher r_target.
    # So best_pts should be good.
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    
    # 1. Initialize
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * math.cos(angle)
            points[i, 1] = r * math.sin(angle)
            
    best_points = points.copy()
    best_score = _compute_radius(points)
    
    # 2. Local Search with Restarts
    # We run multiple rounds of local search with increasing perturbation
    # or just one long run with annealing.
    
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 10.0:
        # Run local search
        # Use a time budget for the local search
        time_budget = min(remaining * 0.9, 100.0)
        
        # We can try multiple restarts if time permits
        # But local search with annealing is usually good enough for 1 run if long enough.
        # Let's do a single long run with annealing.
        
        optimized, score = _local_search(points, max_iter=50000, time_limit=time_budget)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
    # 3. Final Polish / Validation
    # The local search might leave points slightly overlapping due to the greedy step.
    # We do a final "relaxation" pass to ensure validity.
    
    # Check validity and fix
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-9)
        
    # Check coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    if min_d < 1e-6:
        # Perturb coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                    best_points[j, 0] += 1e-5
                    best_points[j, 1] += 1e-5 * random.uniform(-1, 1)
                    # Re-check boundary
                    nr = np.linalg.norm(best_points[j])
                    if nr > 1.0:
                        best_points[j] = best_points[j] / nr * 0.99

    # Final score check
    final_score = _compute_radius(best_points)
    if final_score < best_score:
        # This shouldn't happen if we track best, but just in case
        pass
        
    return best_points
