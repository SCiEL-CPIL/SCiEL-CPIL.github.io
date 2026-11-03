import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Gradient Descent (Force-Directed Layout) with iterative radius increase
    to find a near-optimal packing of N circles in a unit circle.
    """
    
    def compute_forces(points, r):
        """
        Compute the force on each point to minimize overlap and boundary violation.
        Returns an array of forces (N, 2).
        """
        forces = np.zeros((N, 2))
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    # Random perturbation if coincident
                    angle = np.random.rand() * 2 * np.pi
                    diff = np.array([np.cos(angle), np.sin(angle)])
                    dist = 1.0
                elif dist < 2 * r:
                    # Repulsive force proportional to overlap
                    overlap = 2 * r - dist
                    force_mag = overlap
                    forces[i] += force_mag * (diff / dist)
                    forces[j] -= force_mag * (diff / dist)
        
        # Boundary repulsion
        for i in range(N):
            dist_to_origin = np.linalg.norm(points[i])
            if dist_to_origin > 1.0 - r:
                # Push towards center
                # Overlap with boundary
                overlap = dist_to_origin - (1.0 - r)
                force_mag = overlap
                if dist_to_origin > 1e-9:
                    forces[i] -= force_mag * (points[i] / dist_to_origin)
                else:
                    # If at center, force is zero from boundary term, but repulsion handles it
                    pass
                    
        return forces

    def minimize_energy(points, r, max_iters=1000, tol=1e-6):
        """
        Run gradient descent to minimize energy for a fixed radius r.
        Returns the minimized points and the final energy (approximated by max violation).
        """
        lr = 0.01
        best_points = points.copy()
        best_violation = np.inf
        
        for it in range(max_iters):
            forces = compute_forces(points, r)
            
            # Check violation
            min_dist = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(points[i] - points[j])
                    if d < min_dist:
                        min_dist = d
            max_r = np.max(np.linalg.norm(points, axis=1))
            boundary_viol = max(0, max_r + r - 1.0)
            pair_viol = max(0, 2 * r - min_dist)
            violation = max(pair_viol, boundary_viol)
            
            if violation < best_violation:
                best_violation = violation
                best_points = points.copy()
                
            if violation < tol:
                break
            
            # Update positions
            # Adaptive learning rate? Simple decay.
            points += lr * forces
            lr *= 0.999 # Slow decay
            
            # Optional: Project back to unit circle if way out? 
            # The force should handle it, but let's ensure stability.
            norms = np.linalg.norm(points, axis=1)
            mask = norms > 1.5
            if np.any(mask):
                points[mask] = points[mask] / norms[mask][:, np.newaxis] * 1.5

        return best_points, best_violation

    # Initialization: Grid or Spiral
    points = np.zeros((N, 2))
    # Use a simple grid initialization for variety, then let forces arrange them
    # Or use the Fermat spiral again as it's a good starting point for any method
    for i in range(N):
        r_init = np.sqrt(i / N) * 0.9
        theta = i * np.pi * (3 - np.sqrt(5))
        points[i] = [r_init * np.cos(theta), r_init * np.sin(theta)]
        
    # Start with a small radius
    r = 0.05
    min_step = 0.001
    max_r = 0.13 # Upper bound known to be close
    
    start_time = time.time()
    time_limit = 115.0
    
    current_points = points.copy()
    
    # We will try to increase r
    # We keep a history of best points for each r
    best_overall_points = points.copy()
    best_overall_r = r
    
    while time.time() - start_time < time_limit:
        # Try to increase r
        r_new = r + min_step
        
        # Run gradient descent for r_new
        new_points, violation = minimize_energy(current_points, r_new, max_iters=200, tol=1e-5)
        
        if violation < 1e-4:
            # Success
            r = r_new
            current_points = new_points
            best_overall_r = r
            best_overall_points = current_points.copy()
            
            # Maybe increase step size slightly?
            if r > 0.12:
                min_step = 0.0005
            elif r > 0.1:
                min_step = 0.001
            else:
                min_step = 0.002
        else:
            # Failure to pack at r_new
            # Try smaller step? Or stop?
            # Let's try to reduce the step size and retry a few times
            min_step *= 0.5
            if min_step < 1e-5:
                break
            
    # Final polish: Run a few more gradient steps on the best found radius
    # to ensure it's as tight as possible
    for _ in range(500):
        forces = compute_forces(best_overall_points, best_overall_r)
        best_overall_points += 0.005 * forces
        # Ensure validity
        norms = np.linalg.norm(best_overall_points, axis=1)
        mask = norms > 1.0 - best_overall_r
        if np.any(mask):
             # This is tricky, gradient descent should handle it, but if it overshoots...
             # Let's just trust the forces for now, or clip.
             # Clipping might create overlaps. Better to let forces fix it.
             pass

    # Final check and cleanup
    # Calculate actual score to return
    def get_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        return min(min_pairwise / 2.0, min_boundary)

    # Ensure the returned points are valid
    final_score = get_score(best_overall_points)
    
    # If the gradient descent ended in a slightly invalid state, try to fix it
    # by scaling towards center if boundary is violated
    radii = np.linalg.norm(best_overall_points, axis=1)
    if np.any(radii > 1.0 - final_score):
        # Scale down
        scale = (1.0 - final_score) / np.max(radii)
        best_overall_points *= scale
        # Recalculate score
        final_score = get_score(best_overall_points)

    return best_overall_points

# EVOLVE-BLOCK-END
