"""
Pack 200 equal circles in a unit square using Simulated Annealing 
with an incremental radius strategy.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 140  # Leave margin for 150s limit

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    Returns the min of (wall distances) and (pairwise distances / 2).
    """
    if len(pts) == 0:
        return 0.0
    
    # Wall distances
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _generate_hex_initial(s=0.0732):
    """
    Generate a centered hexagonal lattice packing as initial guess.
    """
    points = []
    row_h = s * np.sqrt(3) / 2
    y = -0.5 + row_h/2
    row_idx = 0
    while y < 0.5:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + s/2 + shift
        while x < 0.5:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    if len(pts) == 0:
        return np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Filter points inside the square
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        return np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts

def _calc_cost(pts, r):
    """
    Calculate the cost function for SA.
    Cost = sum of squared overlaps + sum of squared wall violations.
    """
    # Pairwise overlaps
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    
    # Overlaps where D < 2r
    overlaps = np.where(D < 2*r, (2*r - D), 0.0)
    pair_cost = np.sum(overlaps**2)
    
    # Wall violations
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    
    wall_viol_left = np.where(d_left < r, (r - d_left), 0.0)
    wall_viol_right = np.where(d_right < r, (r - d_right), 0.0)
    wall_viol_bottom = np.where(d_bottom < r, (r - d_bottom), 0.0)
    wall_viol_top = np.where(d_top < r, (r - d_top), 0.0)
    
    wall_cost = np.sum(wall_viol_left**2) + np.sum(wall_viol_right**2) + \
                np.sum(wall_viol_bottom**2) + np.sum(wall_viol_top**2)
    
    return pair_cost + wall_cost

def _sa_pack(pts_init, r_target, time_limit):
    """
    Run Simulated Annealing to pack circles of radius r_target.
    Returns (success, best_pts).
    """
    t_start = time.time()
    pts = pts_init.copy()
    current_cost = _calc_cost(pts, r_target)
    best_pts = pts.copy()
    best_cost = current_cost
    
    # SA Parameters
    T = 0.005
    T_min = 1e-6
    alpha = 0.9995
    max_steps = 100000 # High limit, but time-limited
    
    for step in range(max_steps):
        if time.time() - t_start > time_limit:
            break
            
        # Pick a random particle
        i = np.random.randint(N)
        
        # Propose a move: random displacement
        scale = 0.01 * np.sqrt(T)
        delta = np.random.normal(0, scale, 2)
        
        new_pts = pts.copy()
        new_pts[i] += delta
        
        # Keep within bounds
        new_pts[i] = np.clip(new_pts[i], -0.5, 0.5)
        
        new_cost = _calc_cost(new_pts, r_target)
        
        # Acceptance criterion
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            current_cost = new_cost
            pts = new_pts
            if current_cost < best_cost:
                best_cost = current_cost
                best_pts = pts.copy()
        
        T *= alpha
        if T < T_min:
            break
            
    # Check if we achieved a valid packing (cost < epsilon)
    success = best_cost < 1e-4
    return success, best_pts

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    # Initial guess
    pts_init = _generate_hex_initial()
    r_current = _get_radius(pts_init)
    
    best_r = r_current
    best_pts = pts_init.copy()
    
    # Strategy: Incremental Radius with SA
    delta_r = 0.0001
    min_delta_r = 0.000005
    failures = 0
    
    while True:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 10 - elapsed
        if remaining < 5:
            break
            
        # Try to increase radius
        r_target = best_r + delta_r
        
        # Run SA for a portion of the remaining time
        sa_time = min(remaining / 2, 10) # Cap SA time per step to allow multiple attempts
        
        success, pts_sa = _sa_pack(best_pts, r_target, sa_time)
        
        if success:
            # Verify radius
            r_new = _get_radius(pts_sa)
            if r_new > best_r:
                best_r = r_new
                best_pts = pts_sa
                failures = 0
                # Increase delta slightly
                delta_r = min(delta_r * 1.05, 0.0005)
            else:
                # SA found a config but radius didn't improve significantly
                best_pts = pts_sa
                failures += 1
                if failures > 3:
                    delta_r *= 0.5
                    if delta_r < min_delta_r:
                        break
        else:
            # Failed to pack at r_target
            failures += 1
            delta_r *= 0.5
            if delta_r < min_delta_r:
                # Try random restarts if time permits
                if remaining > 10:
                    for _ in range(3):
                        if time.time() - t0 > _TIME_LIMIT - 10:
                            break
                        pts_rand = np.random.uniform(-0.4, 0.4, (N, 2))
                        r_rand = _get_radius(pts_rand)
                        if r_rand > best_r:
                            best_r = r_rand
                            best_pts = pts_rand
                            delta_r = 0.0001
                            failures = 0
                            break
                else:
                    break
            continue
        
        # Check time
        if time.time() - t0 > _TIME_LIMIT - 5:
            break

    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
