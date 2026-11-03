"""
Strategy: Physically-motivated initialization followed by rigorous local search.
1. Initialize points using a "repulsion" simulation (charges repel, walls attract to center).
2. Local search: For each point, try moving it in random directions. Accept if the minimum
   separation (radius) increases.
3. Repair: If a move causes overlap, push the overlapping points apart.
This approach directly optimizes the exact metric required by the evaluator.
"""

import numpy as np
import time
import sys

N = 50

def _compute_radius(pts):
    """
    Compute the maximum radius r such that N circles of radius r 
    centered at pts fit inside the unit circle.
    r = min( min_{i<j} |p_i - p_j| / 2, min_i (1 - |p_i|) )
    """
    if len(pts) == 0:
        return 0.0
    
    # Wall constraint: distance from origin to point must be <= 1 - r
    # So r <= 1 - |p_i|
    dist_to_center = np.linalg.norm(pts, axis=1)
    wall_min = 1.0 - np.max(dist_to_center)
    
    if wall_min <= 0:
        return 0.0
        
    # Pairwise constraint: distance between points must be >= 2r
    # So 2r <= |p_i - p_j| => r <= |p_i - p_j| / 2
    # Compute all pairwise distances efficiently
    # Use broadcasting: pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    
    # Exclude diagonal (distance 0)
    np.fill_diagonal(dist_sq, np.inf)
    
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq == np.inf:
        pair_min = np.inf
    else:
        pair_min = np.sqrt(min_dist_sq) / 2.0
        
    return min(wall_min, pair_min)

def _push_apart(pts, min_dist, max_iter=1000):
    """
    Repair overlaps by pushing points apart.
    Returns the repaired points.
    """
    pts = pts.copy()
    n = len(pts)
    
    for _ in range(max_iter):
        moved = False
        # Find all pairs that are too close
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Find pairs with dist < min_dist
        # We only need to fix the most violating pairs
        min_dists = np.min(dist_sq, axis=1)
        violators = np.where(min_dists < min_dist * min_dist)[0]
        
        if len(violators) == 0:
            break
            
        for i in violators:
            # Find all j such that dist(i, j) < min_dist
            d_sq_i = dist_sq[i]
            j_candidates = np.where(d_sq_i < min_dist * min_dist)[0]
            
            for j in j_candidates:
                if i == j:
                    continue
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    # Random kick
                    d_vec = np.random.normal(size=2)
                    d = np.linalg.norm(d_vec)
                d_hat = d_vec / d
                # Push i away from j and j away from i
                # Required separation is min_dist
                # Current separation is d
                # Deficit is min_dist - d
                # Push each by deficit/2
                push = (min_dist - d) / 2.0
                pts[i] += push * d_hat
                pts[j] -= push * d_hat
                moved = True
                
        # Also ensure points stay inside the container? 
        # The push might push them out. 
        # We will handle wall constraints in the local search acceptance step.
        # But for stability, let's project them back if they go far out.
        # Actually, the objective function penalizes going out.
        # Let's just return. The local search will fix it.
        
        if not moved:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Initialization
    # Start with a hexagonal grid or random points?
    # Random points with repulsion simulation is often better for escaping bad local minima.
    # Let's try a few initializations and keep the best.
    
    best_pts = None
    best_r = 0.0
    
    # Initialization 1: Random
    np.random.seed(42)
    pts = np.random.uniform(-0.9, 0.9, size=(N, 2))
    
    # Simple repulsion simulation to spread them out
    # This is not a rigorous optimizer, just a good initial guess
    for _ in range(500):
        if time.time() > deadline - 10:
            break
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i+1, N):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-6:
                    d_vec = np.random.normal(size=2)
                    d = np.linalg.norm(d_vec)
                # Repulsion force ~ 1/d^2
                f = 1.0 / (d * d)
                f_hat = d_vec / d
                forces[i] += f * f_hat
                forces[j] -= f * f_hat
            # Attraction to center to keep them in
            # Soft wall: if |p| > 0.8, pull back
            r = np.linalg.norm(pts[i])
            if r > 0.8:
                forces[i] -= 5.0 * (r - 0.8) * pts[i] / r
        
        # Move
        pts += 0.01 * forces
        # Project back to unit circle if needed
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            pts[mask] /= norms[mask, np.newaxis] * 0.95
            
    r_init = _compute_radius(pts)
    if r_init > best_r:
        best_r = r_init
        best_pts = pts.copy()
        
    # 2. Local Search
    # We will iteratively improve the best configuration found.
    # For each point, try random perturbations.
    # If the new radius is larger, accept.
    # If the new configuration has overlaps, try to repair it.
    
    current_pts = best_pts.copy()
    current_r = best_r
    
    # Precompute neighbor lists or just use all pairs? N=50 is small, all pairs is fine.
    # Use a "best improvement" or "first improvement" strategy?
    # First improvement is faster per iteration.
    
    # To speed up, we can use a "kick" strategy:
    # Pick a random point, move it slightly, check if r increases.
    
    # We'll use a more robust local search:
    # 1. Calculate current r.
    # 2. Identify the "bottleneck" constraints (pairs that define min dist, points that define min wall).
    # 3. Try to move points involved in bottlenecks to relieve them.
    
    # However, a simpler random hill climbing with occasional "jumps" (restarts) often works well.
    # Let's do a hybrid:
    # - Local search for 80% of time.
    # - If stuck (no improvement for X iterations), do a "kick" (random perturbation of a few points).
    
    iterations_without_improvement = 0
    max_no_improve = 2000
    kick_count = 0
    
    while time.time() < deadline - 5.0:
        # Pick a random point
        idx = np.random.randint(N)
        
        # Generate a random direction and magnitude
        # Magnitude should be proportional to current r
        # Try several magnitudes
        improved = False
        for _ in range(50):
            angle = np.random.uniform(0, 2 * np.pi)
            # Try small moves first
            magnitude = current_r * np.random.uniform(0.01, 0.2)
            new_pt = current_pts[idx].copy()
            new_pt[0] += magnitude * np.cos(angle)
            new_pt[1] += magnitude * np.sin(angle)
            
            # Check if new point is inside container (loosely)
            if np.linalg.norm(new_pt) > 1.0:
                continue
                
            trial_pts = current_pts.copy()
            trial_pts[idx] = new_pt
            
            # Check for overlaps. If overlaps, try to repair.
            # But repair is expensive. 
            # Alternative: only accept if no overlaps AND r increases?
            # That's too strict. 
            # Better: calculate r_trial. If r_trial > current_r, accept.
            # If r_trial <= current_r but overlaps are resolved, maybe accept?
            # The objective function _compute_radius handles overlaps by returning a small r.
            # So if r_trial > current_r, it implies overlaps are reduced or resolved.
            
            r_trial = _compute_radius(trial_pts)
            
            if r_trial > current_r:
                current_pts = trial_pts
                current_r = r_trial
                best_pts = current_pts.copy()
                best_r = current_r
                iterations_without_improvement = 0
                improved = True
                break
            else:
                # If we made it worse or equal, revert.
                pass
        
        if not improved:
            iterations_without_improvement += 1
            
        if iterations_without_improvement > max_no_improve:
            # Kick: randomly perturb a few points
            # This helps escape local minima
            num_kicks = np.random.randint(2, 5)
            kick_indices = np.random.choice(N, size=num_kicks, replace=False)
            for idx in kick_indices:
                angle = np.random.uniform(0, 2 * np.pi)
                magnitude = current_r * np.random.uniform(0.5, 1.5)
                current_pts[idx][0] += magnitude * np.cos(angle)
                current_pts[idx][1] += magnitude * np.sin(angle)
                
            # Repair the kicked configuration
            # Use push_apart with current_r as min_dist? 
            # No, use a slightly smaller min_dist to allow flexibility
            # Actually, just let the local search fix it.
            # But if it's too broken, r will be very small.
            # Let's try to repair it to be valid first.
            current_pts = _push_apart(current_pts, 2 * current_r * 0.9, max_iter=200)
            
            # Recalculate r
            current_r = _compute_radius(current_pts)
            
            # If kick made it worse than best, revert to best?
            # Or keep going? 
            # If we revert, we might get stuck in the same local minimum.
            # Let's keep the kicked state but if it's much worse, maybe revert.
            if current_r < best_r * 0.5:
                # Too bad, revert to best
                current_pts = best_pts.copy()
                current_r = best_r
            else:
                # Accept the kick state as new current
                pass
                
            iterations_without_improvement = 0
            kick_count += 1
            
            # If we've done too many kicks, maybe restart from scratch?
            if kick_count > 10:
                # Restart with a different random seed
                np.random.seed(np.random.randint(10000))
                pts = np.random.uniform(-0.9, 0.9, size=(N, 2))
                for _ in range(200):
                    forces = np.zeros_like(pts)
                    for i in range(N):
                        for j in range(i+1, N):
                            d_vec = pts[i] - pts[j]
                            d = np.linalg.norm(d_vec)
                            if d < 1e-6:
                                d_vec = np.random.normal(size=2)
                                d = np.linalg.norm(d_vec)
                            f = 1.0 / (d * d)
                            f_hat = d_vec / d
                            forces[i] += f * f_hat
                            forces[j] -= f * f_hat
                        r = np.linalg.norm(pts[i])
                        if r > 0.8:
                            forces[i] -= 5.0 * (r - 0.8) * pts[i] / r
                    pts += 0.01 * forces
                    norms = np.linalg.norm(pts, axis=1)
                    mask = norms > 0.95
                    if np.any(mask):
                        pts[mask] /= norms[mask, np.newaxis] * 0.95
                r_init = _compute_radius(pts)
                if r_init > best_r:
                    best_r = r_init
                    best_pts = pts.copy()
                    current_pts = best_pts.copy()
                    current_r = best_r
                kick_count = 0

    # Final check
    # Ensure the best_pts are valid
    final_r = _compute_radius(best_pts)
    if final_r > 0:
        return best_pts
    else:
        # Fallback to hexagonal if something went wrong
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

if __name__ == "__main__":
    pts = construct()
    print(_compute_radius(pts))
