"""
Strategy: Local pairwise push-apart relaxation with multi-restart perturbation.
Instead of global Nelder-Mead on 600 variables, we use a local search that only
moves the two circles involved in each constraint violation. This is much more
efficient for the min-objective and can escape local minima via restarts.
"""
import numpy as np
import time
import math

N = 300
DEADLINE = 118.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
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
            # Keep the N closest to center for a compact packing
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
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

def _score(points):
    """Compute the maximum radius r for given centers."""
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _local_relax(points, max_iters=5000, time_limit=60.0):
    """
    Local pairwise push-apart relaxation.
    Only moves the two circles involved in each constraint violation.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    # Track best score found
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Build adjacency: for each point, find neighbors within 2.5 * current_radius
    # This avoids checking all 44850 pairs every iteration
    current_r = best_score
    
    for it in range(max_iters):
        # Time check
        if time.time() - start_time > time_limit:
            break
        
        # Estimate current radius
        dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(dist_boundary)
        
        # Only check pairs that could be violating
        # If current radius is r, then pairs with dist < 2*r are problematic
        # But we don't know r exactly, so use min_boundary as estimate
        r_est = min_boundary
        
        # Find all violating pairs
        # For efficiency, only check pairs within 2.5 * r_est
        threshold = 2.5 * r_est if r_est > 0 else 0.3
        
        # Compute pairwise distances (vectorized)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find violating pairs
        violating = dists < 2.0 * r_est
        violating_idx = np.argwhere(violating)
        
        # Also check boundary violations
        boundary_violating = np.where(dist_boundary < 0)[0]
        
        # If no violations, we're at a local optimum
        if len(violating_idx) == 0 and len(boundary_violating) == 0:
            # Try to increase radius slightly
            # Actually, we're done with this relaxation phase
            break
        
        # Process a random subset of violations to avoid bias
        if len(violating_idx) > 0:
            # Shuffle and take a subset
            np.random.shuffle(violating_idx)
            subset = violating_idx[:min(50, len(violating_idx))]
            
            for i, j in subset:
                if i >= j:
                    continue
                d = dists[i, j]
                if d < 1e-12:
                    # Coincident points - push apart randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    pts[i, 0] -= 0.01 * np.cos(angle)
                    pts[i, 1] -= 0.01 * np.sin(angle)
                    pts[j, 0] += 0.01 * np.cos(angle)
                    pts[j, 1] += 0.01 * np.sin(angle)
                    continue
                
                # Push apart along the line connecting them
                direction = (pts[j] - pts[i]) / d
                push = (2.0 * r_est - d) / 2.0 + 1e-6  # Slight extra push
                pts[i] -= push * direction
                pts[j] += push * direction
        
        # Handle boundary violations
        for idx in boundary_violating:
            r = np.hypot(pts[idx, 0], pts[idx, 1])
            if r > 0:
                # Push inward
                pts[idx] = pts[idx] / r * (1.0 - 1e-6)
            else:
                pts[idx, 0] = 0.01
                pts[idx, 1] = 0.0
        
        # Periodically check score
        if it % 100 == 0:
            current_score = _score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
    
    return best_pts, best_score

def _perturb(points, magnitude=0.02):
    """Perturb points to escape local minima."""
    pts = points.copy()
    noise = np.random.normal(0, magnitude, size=pts.shape)
    pts += noise
    # Ensure points stay inside the unit disk
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r > 0.95
    pts[mask] = pts[mask] / r[mask, np.newaxis] * 0.95
    return pts

def construct():
    start_time = time.time()
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Multi-restart local relaxation
    max_restarts = 10
    for restart in range(max_restarts):
        # Time check
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5.0  # 5s safety margin
        if remaining < 5.0:
            break
        
        # For first restart, use original; for subsequent, perturb
        if restart == 0:
            current = points.copy()
        else:
            # Perturb from best found so far
            magnitude = 0.01 + 0.01 * restart
            current = _perturb(best_points, magnitude)
        
        # Run local relaxation
        time_limit_for_this = min(remaining * 0.8, 30.0)
        relaxed, score = _local_relax(current, max_iters=2000, time_limit=time_limit_for_this)
        
        if score > best_score:
            best_score = score
            best_points = relaxed.copy()
        
        # If we've hit a good score, we can stop early
        if best_score > 0.053:
            break
    
    # Final check: ensure validity
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6
    
    return best_points
