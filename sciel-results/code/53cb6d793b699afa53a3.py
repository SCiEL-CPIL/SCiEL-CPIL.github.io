"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Strategy: Force-based relaxation
    # 1. Initialize centers using a Fibonacci disk distribution for uniform coverage.
    # 2. Iteratively adjust positions based on forces:
    #    - Repulsion between circles if distance < 2r (push apart)
    #    - Attraction to the boundary for all circles (push out to maximize r)
    #    - Constraint enforcement: keep inside unit circle (radius 1 - r)
    # 3. The "radius" r is effectively determined by the minimum distance constraint.
    #    We don't fix r initially; we let the positions settle, then compute the max r.
    #    However, to maximize r, we need to push circles as far apart as possible.
    #    A better formulation: 
    #    Let positions P. 
    #    We want to maximize min( min_{i<j} |P_i - P_j|/2, min_i (1 - |P_i|) ).
    #    This is equivalent to finding a configuration where the "minimum clearance" is maximized.
    #    
    #    We can simulate this by treating the circles as having a variable radius r.
    #    Start with a small r, expand it slowly while relaxing positions.
    #    Or, simpler: 
    #    Initialize P. 
    #    Loop:
    #      Compute current min_dist_pair and min_dist_boundary.
    #      r_current = min(min_dist_pair/2, min_dist_boundary).
    #      Apply forces to increase these minima.
    #      Force for pair (i,j) if dist < 2*r_target: push apart.
    #      Force for boundary if |P_i| > 1 - r_target: push in.
    #      But we want to MAXIMIZE r. So we should push pairs apart always?
    #      No, that might push them out of bounds.
    #      
    #    Alternative: "Jostle" method.
    #    1. Start with a valid configuration (e.g., hex lattice or Fibonacci) with radius r0.
    #    2. Increase r slightly.
    #    3. Relax positions to satisfy non-overlap and boundary constraints for the new r.
    #    4. If relaxation fails, decrease r and try again? Or use soft constraints.
    #    
    #    Let's use the soft constraint approach with a target radius r_target.
    #    We will perform a binary search on r_target? No, that's slow.
    #    Instead, we will simulate an annealing process where we try to push circles apart.
    #    
    #    Let's stick to a direct force simulation aiming to maximize the minimum distance.
    #    Define a "score" function for a configuration: S = min(d_min_pair/2, d_min_boundary).
    #    We want to maximize S.
    #    
    #    Gradient ascent on S is hard due to min().
    #    Instead, use a penalty function:
    #    Penalty = sum( max(0, 2*r - dist_ij)^2 ) + sum( max(0, dist_i - (1-r))^2 )
    #    Here r is a parameter we are trying to maximize.
    #    
    #    Better approach: 
    #    1. Initialize P with Fibonacci disk.
    #    2. Set r to a small value, e.g., 0.01.
    #    3. Relax P to minimize overlaps for this r.
    #    4. Increase r by a small step.
    #    5. Relax P again.
    #    6. Repeat until time runs out or convergence.
    #    7. The final r for which relaxation succeeded (or converged well) is our answer.
    #    8. Compute the actual max radius for the final P.
    
    # Initialization: Fibonacci Disk
    def fibonacci_disk(n):
        points = np.zeros((n, 2))
        golden_angle = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            r = np.sqrt(i / n)
            theta = i * golden_angle
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
        return points
    
    P = fibonacci_disk(N)
    
    # Initial radius estimate
    # Compute min distances
    def compute_min_dist(P, r):
        # Pairwise distances
        # For N=300, 300x300 matrix is 90k entries. Fast enough.
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distances
        radii = np.sqrt(np.sum(P**2, axis=1))
        min_bound = np.min(1.0 - radii)
        
        return min(min_pair, min_bound)
    
    # Relaxation function for a fixed r
    # Returns updated P and success flag
    def relax(P, r, steps=2000, lr=0.05):
        P = P.copy()
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Compute forces
            forces = np.zeros_like(P)
            
            # Pairwise repulsion if dist < 2r
            # To be efficient, we can use a grid or just compute all pairs since N=300 is small
            diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            
            # Mask for overlapping pairs
            mask = dists < 2.0 * r
            mask.fill_diagonal(False)
            
            if np.any(mask):
                # Force magnitude proportional to overlap
                overlaps = (2.0 * r - dists) * mask
                # Normalize direction
                # Avoid division by zero
                dists_safe = np.where(dists > 1e-9, dists, 1.0)
                dirs = diff / dists_safe[:, :, np.newaxis]
                
                # Sum forces
                # Force on i from j is -overlap * dir_ij (push i away from j)
                # Actually, if dist < 2r, we want to push them apart.
                # Vector from i to j is P_j - P_i = -diff[i,j] if diff is P_i - P_j.
                # Let's define diff = P_i - P_j. Then vector i->j is -diff.
                # Force on i should be in direction away from j, i.e., diff direction.
                # So force_i += overlap * (diff / dist)
                
                # Using broadcasting:
                # forces[i] = sum_j overlap_ij * (diff_ij / dist_ij)
                # diff is (N, N, 2). diff[i,j] = P_i - P_j.
                # We want to add to P_i a vector in direction P_i - P_j.
                
                unit_dirs = diff / dists_safe[:, :, np.newaxis]
                # Multiply by overlap magnitude
                force_contribs = unit_dirs * (overlaps[:, :, np.newaxis])
                
                # Sum over j (axis 1)
                forces += np.sum(force_contribs, axis=1)
            
            # Boundary constraint: if |P_i| > 1 - r, push inward
            radii = np.sqrt(np.sum(P**2, axis=1))
            boundary_violation = np.maximum(0, radii - (1.0 - r))
            
            if np.any(boundary_violation > 0):
                # Direction is inward: -P / |P|
                P_safe = np.where(radii[:, np.newaxis] > 1e-9, P, 1e-9)
                inward_dirs = -P / radii[:, np.newaxis]
                forces += boundary_violation[:, np.newaxis] * inward_dirs
            
            # Update positions
            # Simple gradient descent on penalty (minimizing overlap)
            # Penalty is convex locally, so simple descent works.
            P += lr * forces
            
            # Hard clamp to container to prevent instability
            radii = np.sqrt(np.sum(P**2, axis=1))
            max_r = 1.0 - r
            mask_out = radii > max_r
            if np.any(mask_out):
                P[mask_out] *= (max_r / radii[mask_out])[:, np.newaxis]
        
        return P
    
    # Main loop: Incremental radius increase
    r_current = 0.01
    # Estimate max possible r ~ 0.05
    # Step size for r
    r_step = 0.0005
    
    # First, relax for initial r to settle the Fibonacci points
    P = relax(P, r_current, steps=1000, lr=0.05)
    
    while time.time() < deadline - 5.0:
        # Try to increase r
        r_new = r_current + r_step
        
        # Relax for new r
        P_candidate = relax(P, r_new, steps=500, lr=0.05)
        
        # Check if the configuration is valid for r_new
        # Compute actual min radius
        actual_r = compute_min_dist(P_candidate, r_new)
        
        if actual_r >= r_new * 0.99: # Accept if close to target
            P = P_candidate
            r_current = r_new
            # Decrease step size as we get closer to optimum?
            # r_step *= 0.99
        else:
            # Failed to pack for r_new. 
            # Try a smaller step? Or stop?
            # Let's try to recover by relaxing a bit more at r_current
            P = relax(P, r_current, steps=200, lr=0.02)
            # Break if we can't make progress
            # Simple heuristic: if we fail 5 times in a row, stop
            break
            
    # Final relaxation to maximize the radius for the current configuration
    # We can try to "push" the radius up slightly more with a very fine search
    # Or just compute the final score.
    
    # One last check: ensure all points are strictly inside
    radii = np.sqrt(np.sum(P**2, axis=1))
    # The computed r is the bottleneck. 
    # If any point is outside 1-r, the score is lower.
    # Our relaxation enforced 1-r_current.
    # The actual score is min(min_pair/2, min_bound).
    
    return P

# EVOLVE-BLOCK-END
