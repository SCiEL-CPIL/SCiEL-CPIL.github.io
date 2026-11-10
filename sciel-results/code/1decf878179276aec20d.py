"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = start_time + 115.0

    # 1. Initialization: Hexagonal Lattice
    # Start with a spacing that ensures we have at least N points, but not too many.
    # Area argument: N circles of radius r in unit circle.
    # r ~ sqrt(1/N). Spacing s ~ 2r.
    # Let's start with s such that we get roughly N points in the disk.
    # Number of points in hex lattice in disk of radius R with spacing s is approx pi R^2 / (s^2 * sqrt(3)/2)
    # We want count ~ N.
    # N ~ pi / (s^2 * 0.866) => s^2 ~ pi / (N * 0.866) => s ~ sqrt(pi / (N * 0.866))
    
    s_init = np.sqrt(np.pi / (N * 0.866)) * 1.1 # 10% looser to ensure enough points and validity
    
    def generate_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                # Keep points well inside to start
                if r <= 1.0 - spacing * 0.6:
                    pts.append((x, y))
        if not pts:
            return np.array([])
        return np.array(pts)

    points = generate_hex_lattice(s_init)
    
    # Ensure we have at least N points
    while len(points) < N:
        s_init *= 0.95
        points = generate_hex_lattice(s_init)
        if time.time() > deadline:
            break

    if len(points) < N:
        # Fallback: random points
        points = np.random.uniform(-0.9, 0.9, (N, 2))
    else:
        # If we have more than N, select N.
        # To maximize min distance, we should probably keep the ones that are most "spread out" or just random.
        # Random selection from a valid lattice is a good stochastic start.
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            points = points[indices]
        else:
            # Pad if necessary (should not happen)
            if len(points) < N:
                pad = np.random.uniform(-0.9, 0.9, (N - len(points), 2))
                points = np.vstack([points, pad])

    # 2. Optimization: Force-Directed Relaxation (Minimize Violation)
    # We try to increase a target radius r_target.
    # The objective is to minimize V = sum of squared violations.
    # Violation for pair (i,j): max(0, 2*r_target - dist_ij)^2
    # Violation for boundary i: max(0, dist_i - (1 - r_target))^2
    
    # Initial r_target based on current configuration
    def calc_min_radius(pts):
        # Fast approx for initial target
        # Use cdist for min pair distance
        if len(pts) < 2:
            return 0.0
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
        return min(min_pair / 2.0, min_bound)

    r_current = calc_min_radius(points)
    # Start target slightly higher than current to drive expansion
    r_target = r_current * 1.01
    
    # Parameters
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(points)
    best_score = r_current
    best_points = points.copy()
    
    # Main Loop
    while time.time() < deadline - 5.0: # Leave 5s for final cleanup/checks
        
        # Run a batch of gradient steps for current r_target
        steps_in_batch = 20
        for _ in range(steps_in_batch):
            if time.time() > deadline:
                break
            
            # Compute pairwise distances
            # For N=300, cdist is fast enough
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            
            # Pairwise Forces
            # We want to increase distances for pairs where dist < 2*r_target
            # Force on i from j is proportional to (2*r_target - dist_ij) * (p_i - p_j) / dist_ij
            # Only if dist < 2*r_target
            
            # To avoid division by zero and only apply force to close pairs
            mask = dists < 2.0 * r_target
            # Set diagonal to 1 to avoid issues, though masked out
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            
            # Violation magnitude
            viol = np.where(mask, 2.0 * r_target - dists, 0.0)
            weights = viol / dists_safe
            weights[~mask] = 0.0
            np.fill_diagonal(weights, 0.0)
            
            # diff[i,j] = points[i] - points[j]
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            
            # Force on i is sum_j weights[i,j] * diff[i,j]
            forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary Forces
            radii = np.linalg.norm(points, axis=1)
            b_mask = radii > 1.0 - r_target
            b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
            # Direction: inward (-p_i / |p_i|)
            b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
            
            total_forces = forces + b_forces
            
            # Update velocity and position
            v = momentum * v + lr * total_forces
            points += v
            
            # Hard constraint: Keep inside unit circle
            new_radii = np.linalg.norm(points, axis=1)
            max_r = 1.0 - r_target
            outside = new_radii > max_r
            if np.any(outside):
                # Project back to boundary
                points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
                v[outside] = 0.0 # Stop moving if hitting wall

        # Evaluate score after batch
        current_score = calc_min_radius(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            # Success: Increase target radius
            r_current = current_score
            r_target = r_current * 1.005 # Increase target slightly
            # Decay learning rate to settle
            lr = max(0.001, lr * 0.98)
        else:
            # Failure: Could not achieve r_target
            # Reduce target and try again, or reset to best
            r_target = r_current * 0.995
            # If stuck, reset to best and try a different perturbation or smaller step
            if r_target < best_score * 0.95:
                 # We are far below best, reset
                 points = best_points.copy()
                 v = np.zeros_like(points)
                 r_target = best_score * 1.001
                 lr = 0.005 # Smaller step
            else:
                 # Just keep trying with slightly lower target
                 pass
                 
    # Final Polish: Use a very small step size to tighten the configuration
    # This helps if the main loop stopped with some slack
    if time.time() < deadline - 1.0:
        r_target = best_score * 1.0001
        lr = 0.0005
        momentum = 0.9
        v = np.zeros_like(points)
        
        # Run a few more steps
        for _ in range(50):
            if time.time() > deadline:
                break
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            mask = dists < 2.0 * r_target
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            viol = np.where(mask, 2.0 * r_target - dists, 0.0)
            weights = viol / dists_safe
            weights[~mask] = 0.0
            np.fill_diagonal(weights, 0.0)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            radii = np.linalg.norm(points, axis=1)
            b_mask = radii > 1.0 - r_target
            b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
            b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
            
            total_forces = forces + b_forces
            v = momentum * v + lr * total_forces
            points += v
            
            new_radii = np.linalg.norm(points, axis=1)
            max_r = 1.0 - r_target
            outside = new_radii > max_r
            if np.any(outside):
                points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
                v[outside] = 0.0
                
        final_score = calc_min_radius(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()

    return best_points
# EVOLVE-BLOCK-END
