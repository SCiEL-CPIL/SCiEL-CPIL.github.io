"""Pack 300 equal circles in the unit circle using a ring-based construction + local optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300


def _compute_radius(pts):
    """Compute the feasible common radius for a set of centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Container constraint
    r_container = 1.0 - np.max(np.linalg.norm(pts, axis=1))
    if r_container <= 0:
        return 0.0
    # Pairwise constraint: minimum distance / 2
    # Use chunked computation to manage memory
    min_dist = np.inf
    for i in range(N):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        if d.size > 0:
            min_dist = min(min_dist, np.min(d))
    r_pair = min_dist / 2.0
    return min(r_container, r_pair)


def _objective(x, pts_init):
    """Objective: maximize minimum distance (negative for minimization).
    We minimize -min_dist + penalty for going outside."""
    pts = x.reshape(N, 2)
    # Penalty for going outside unit circle
    dists = np.linalg.norm(pts, axis=1)
    outside_penalty = np.sum(np.maximum(0.0, dists - 1.0 + 0.01)) ** 2 * 100.0
    
    # Compute minimum pairwise distance
    min_dist = np.inf
    for i in range(N):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        if d.size > 0:
            min_dist = min(min_dist, np.min(d))
    
    return -min_dist + outside_penalty


def _build_ring_initial():
    """Build a ring-based initial configuration."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    # Build rings from center outward
    # For 300 circles, we need roughly sqrt(300) ~ 17 rings
    # Each ring at radius r can hold approximately 2*pi*r / (2*r_circle) circles
    # But let's just place them on concentric rings with angular spacing
    
    remaining = N - 1  # one is at center
    ring_idx = 1
    while remaining > 0:
        # Radius of this ring
        r = ring_idx * 0.055  # approximate spacing
        if r >= 0.98:
            # Last ring, place all remaining
            n_on_ring = remaining
        else:
            # Number of circles that fit on this ring
            # Circumference / (2 * r_circle) where r_circle ~ 0.053
            n_on_ring = max(1, int(2 * np.pi * r / (2 * 0.053)))
            n_on_ring = min(n_on_ring, remaining)
        
        angles = 2 * np.pi * np.arange(n_on_ring) / n_on_ring
        for angle in angles:
            points.append([r * np.cos(angle), r * np.sin(angle)])
        
        remaining -= n_on_ring
        ring_idx += 1
    
    points = np.array(points[:N])
    return points


def _local_optimize(pts, time_limit=100.0):
    """Perform local optimization using gradient-based methods."""
    start_time = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Try a few random restarts with small perturbations
    for restart in range(5):
        if time.time() - start_time > time_limit - 10:
            break
        
        if restart == 0:
            current = pts.copy()
        else:
            # Add small random perturbation
            current = pts + 0.01 * np.random.randn(N, 2)
        
        # Use Nelder-Mead or BFGS for optimization
        # We'll use a simplified approach: iterative pairwise repulsion
        
        # Simple iterative optimization: push apart overlapping circles
        for iteration in range(500):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute forces
            forces = np.zeros_like(current)
            
            # Container force: pull inward if outside
            dists = np.linalg.norm(current, axis=1)
            for i in range(N):
                if dists[i] > 0.99:
                    direction = current[i] / (dists[i] + 1e-10)
                    forces[i] -= direction * 0.1 * (dists[i] - 0.99)
            
            # Pairwise repulsion
            for i in range(N):
                d = current[i + 1:] - current[i]
                dist = np.linalg.norm(d, axis=1)
                # Only consider close pairs
                close = dist < 0.2
                if np.any(close):
                    for j in np.where(close)[0]:
                        actual_j = i + 1 + j
                        if dist[j] > 1e-10:
                            force_mag = 0.05 / (dist[j] ** 2 + 1e-6)
                            forces[i] += d[j] / dist[j] * force_mag
                            forces[actual_j] -= d[j] / dist[j] * force_mag
            
            # Update positions
            current += forces * 0.5
            
            # Clamp to unit circle
            new_dists = np.linalg.norm(current, axis=1)
            mask = new_dists > 0.99
            if np.any(mask):
                current[mask] *= 0.99 / new_dists[mask]
            
            # Check improvement
            r = _compute_radius(current)
            if r > best_r + 1e-8:
                best_r = r
                best_pts = current.copy()
        
        # If no improvement, break
        if _compute_radius(current) <= best_r + 1e-6:
            if restart > 0:
                break
    
    return best_pts, best_r


def construct():
    """Construct 300 circle centers in the unit circle maximizing common radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build initial configuration
    pts = _build_ring_initial()
    
    # Ensure valid initial state
    r_init = _compute_radius(pts)
    if r_init <= 0:
        # Fallback: uniform random
        np.random.seed(42)
        while True:
            pts = np.random.randn(N, 2)
            dists = np.linalg.norm(pts, axis=1)
            mask = dists > 1.0
            if not np.any(mask):
                break
            pts[mask] *= 1.0 / dists[mask]
        r_init = _compute_radius(pts)
    
    best_pts = pts.copy()
    best_r = r_init
    
    # Local optimization
    if time.time() < deadline - 5:
        best_pts, best_r = _local_optimize(pts, time_limit=deadline - time.time() - 5)
    
    # Final validation
    final_r = _compute_radius(best_pts)
    if final_r > 0:
        return best_pts
    else:
        # Return the initial if optimization failed
        return pts
