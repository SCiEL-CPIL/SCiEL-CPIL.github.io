import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
    # We can improve the initial guess.
    # For N=50, a hexagonal lattice is good.
    # Let's use a slightly tighter spacing initially.
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

def _gradient_step(centers, r, step_size):
    """
    Perform a directed gradient ascent step to increase r.
    """
    new_centers = centers.copy()
    
    # Compute pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    
    # Find the minimum distance
    min_dist = np.min(dists)
    
    # Determine which constraints are active (bottlenecks)
    # We use a threshold to include near-min distances to handle degeneracies
    # This creates a "soft" gradient
    threshold = min_dist * 1.05 # 5% above min
    active_pairs = dists < threshold
    
    # Compute repulsive forces from active pairs
    # Force on i from j is (c_i - c_j) / dist
    # We sum these for all active j
    forces = np.zeros_like(centers)
    
    # Efficiently sum forces
    # dists is NxN
    # We want to sum (c_i - c_j) / d_ij for all j where d_ij < threshold
    # Note: d_ij = d_ji
    
    # To avoid division by zero (diagonal is inf, so 1/inf = 0, which is fine)
    # But we only want to sum for active pairs.
    
    # Create a weight matrix
    weights = np.zeros_like(dists)
    weights[active_pairs] = 1.0 / dists[active_pairs]
    
    # Force on i is sum_j weights[i,j] * (c_i - c_j)
    # = sum_j weights[i,j] * c_i - sum_j weights[i,j] * c_j
    # = c_i * sum_j weights[i,j] - sum_j weights[i,j] * c_j
    
    sum_weights = np.sum(weights, axis=1)
    # sum_j weights[i,j] * c_j
    # This is matrix multiplication of weights (NxN) and centers (Nx2)
    weighted_centers = weights @ centers
    
    forces = centers * sum_weights[:, np.newaxis] - weighted_centers
    
    # Add boundary forces
    # If 1 - |c_i| is close to r (and r is determined by boundary or close), push in.
    # Actually, if the current radius r is limited by boundary, we should push in.
    # If r is limited by pair, pushing in might help if it allows spreading out.
    # Generally, keeping circles inside is good.
    # Let's add a force towards center proportional to how close to boundary they are.
    dists_center = np.hypot(centers[:, 0], centers[:, 1])
    # Force towards center: -c_i / |c_i| * strength
    # Strength increases if dists_center is close to 1.
    # Let's use a simple penalty if they are outside or close.
    # Since we want to maximize min(1-|c_i|), we want to decrease |c_i|.
    # Gradient of (1-|c_i|) is -c_i/|c_i|.
    # So we add -c_i/|c_i| to the force.
    
    # Only apply if dists_center > 0
    mask = dists_center > 1e-9
    boundary_forces = np.zeros_like(centers)
    boundary_forces[mask] = -centers[mask] / dists_center[mask, np.newaxis]
    
    # Scale boundary force relative to pair force?
    # The pair force magnitude is roughly 1/d * count.
    # Boundary force magnitude is 1.
    # Let's just add it.
    forces += boundary_forces
    
    # Normalize forces?
    # The step size should be small.
    # Update centers
    new_centers += step_size * forces
    
    # Ensure they don't fly out too far?
    # The boundary force should keep them in.
    # But to be safe, clip to unit circle?
    # If we clip, we might break the "gradient" nature.
    # However, if a circle moves outside, r becomes 0 (invalid).
    # So we must prevent invalid states.
    # If new_centers has any |c| > 1, it's invalid.
    # Let's check and revert or project.
    new_dists = np.hypot(new_centers[:, 0], new_centers[:, 1])
    if np.any(new_dists > 1.0):
        # Project back to unit circle? Or just reject this step?
        # Rejection is safer for gradient ascent on constrained domain.
        # Or scale down the step.
        # Let's just return the original centers if invalid.
        return centers, 0.0
        
    return new_centers, 1.0 # 1.0 indicates success

def _optimize(centers, deadline):
    """
    Directed repulsion-based local search.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 500
    min_step_size = 1e-6
    
    while time.time() < deadline:
        # Perform a batch of gradient steps
        for _ in range(10):
            new_centers, success = _gradient_step(current, current_r, step_size)
            if not success:
                # Step was invalid, reduce step size and try again?
                # Or just break and reheat?
                break
            
            new_r = _compute_radius(new_centers)
            
            if new_r > current_r + 1e-9:
                current = new_centers
                current_r = new_r
                no_improve_count = 0
                if current_r > best_r:
                    best = current.copy()
                    best_r = current_r
            else:
                # No improvement
                no_improve_count += 1
                # We can still move if it's "equivalent" to explore?
                # For now, if no improvement, we might be stuck.
                # Let's accept small moves to explore?
                # Or just break to reheat.
                if no_improve_count > 10:
                    break
                
                # Accept if new_r is close to current_r (plateau)
                if new_r > current_r - 1e-6:
                    current = new_centers
                    current_r = new_r
                else:
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        # Check for stall
        if no_improve_count > max_no_improve:
            # Reheat: random perturbation
            perturbation = np.random.uniform(-step_size, step_size, size=(N, 2))
            perturbed = current + perturbation
            # Ensure perturbed is valid?
            # If not, scale down
            p_dists = np.hypot(perturbed[:, 0], perturbed[:, 1])
            if np.any(p_dists > 1.0):
                # Scale to fit
                scale = 1.0 / np.max(p_dists)
                perturbed *= scale
            
            p_r = _compute_radius(perturbed)
            if p_r > 0:
                current = perturbed
                current_r = p_r
            no_improve_count = 0
            step_size = max(min_step_size, step_size * 0.5)
            
        # Slowly decrease step size?
        # step_size *= 0.999 
        # Or keep it constant until stall?
        # Let's keep it constant to allow exploration, but reduce on stall.

    return best, best_r

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    
    # Run optimization
    best_centers, best_r = _optimize(centers, deadline)
    
    # If time remains, do a few more restarts from best?
    # Or just refine?
    # The gradient step is fast. We can run _optimize again with a smaller step size?
    # Or just let it run until deadline.
    
    return best_centers
