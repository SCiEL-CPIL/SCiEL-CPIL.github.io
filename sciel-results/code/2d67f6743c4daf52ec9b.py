"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


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
    # Start with a reasonable spacing
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Ensure points are strictly inside with some margin
        margin = spacing * 0.5
        mask = radius <= 1.0 - margin
        points = points[mask]
        
        if len(points) >= N:
            # Sort by distance from center to pick the most central N points
            # This tends to produce a more symmetric and stable core
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx].copy()
        spacing *= 0.95
        if spacing < 0.1:
             # Fallback to random if lattice fails
             return np.random.uniform(-0.8, 0.8, (N, 2))


def _local_search(centers, time_limit=10.0):
    """
    Perform local search by pushing apart close circles and pulling boundary circles inward.
    This is a greedy heuristic that is faster than pure annealing for local refinement.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Compute distances
        dx = current[:, 0:1] - current[:, 0:1].T
        dy = current[:, 1:2] - current[:, 1:2].T
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.hypot(current[:, 0], current[:, 1])
        dists_boundary = 1.0 - norms
        
        # Current radius
        r_pair = np.min(dists) / 2.0
        r_boundary = np.min(dists_boundary)
        current_r = min(r_pair, r_boundary)
        
        if current_r > best_r:
            best_r = current_r
            best = current.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 200:
                # Reset step size or break if stuck
                if step_size < 0.001:
                    break
                step_size *= 0.9

        # Calculate forces (gradients of the minimum distance approximation)
        # We want to maximize the minimum distance.
        # Approximation: Sum of repulsive forces from neighbors closer than 2*current_r + margin
        # And attractive force to center for points close to boundary.
        
        forces = np.zeros_like(current)
        
        # 1. Pairwise repulsion
        # Only consider pairs closer than 2 * (current_r + epsilon)
        # Using a soft threshold to avoid discontinuities
        threshold = 2.0 * (current_r + 0.05)
        mask = (dists < threshold) & (dists > 1e-6)
        
        # Force from i to j is (c_i - c_j) / dist_ij^2 (inverse square law for stronger push when close)
        # We sum these for each i.
        # Vectorized approach:
        # F_ij = (c_i - c_j) / dist_ij^2
        # We can compute this for all pairs and sum over j for each i.
        
        # To save memory/time, we only compute for the mask
        # But creating a full N x N x 2 array is 50x50x2 = 5000 floats, which is tiny.
        
        diff_x = current[:, 0:1] - current[:, 0:1].T
        diff_y = current[:, 1:2] - current[:, 1:2].T
        
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-6, dists, 1.0)
        
        # Inverse square force magnitude
        # We want to push apart.
        # Force on i from j is proportional to (c_i - c_j) / dist^2
        # Let's use 1/dist^3 to make the direction unit vector scaled by 1/dist^2?
        # Actually, gradient of 1/d is -1/d^2 * dir. So force is dir / d^2.
        
        inv_dist_sq = np.where(mask, 1.0 / (dists_safe**2), 0.0)
        
        fx = diff_x * inv_dist_sq
        fy = diff_y * inv_dist_sq
        
        # Sum forces over neighbors (axis 1)
        forces[:, 0] += np.sum(fx, axis=1)
        forces[:, 1] += np.sum(fy, axis=1)
        
        # 2. Boundary attraction
        # If a point is close to the boundary (dist_boundary < current_r + margin), pull it to center.
        bnd_mask = dists_boundary < (current_r + 0.05)
        # Force direction is towards origin: -c / |c|
        # Magnitude scaled by how close to boundary it is.
        norms_safe = np.where(norms > 1e-6, norms, 1.0)
        bnd_force_mag = np.where(bnd_mask, 1.0 / (dists_boundary + 0.01)**2, 0.0)
        
        forces[:, 0] -= bnd_force_mag * (current[:, 0] / norms_safe)
        forces[:, 1] -= bnd_force_mag * (current[:, 1] / norms_safe)
        
        # Normalize forces to prevent large jumps
        force_norms = np.hypot(forces[:, 0], forces[:, 1])
        max_force = np.max(force_norms)
        if max_force > 0:
            forces /= max_force
        
        # Update positions
        current += step_size * forces
        
        # Ensure points stay inside the unit circle (hard constraint)
        # If a point goes outside, project it back onto the circle
        new_norms = np.hypot(current[:, 0], current[:, 1])
        outside_mask = new_norms > 1.0
        if np.any(outside_mask):
            current[outside_mask] *= (1.0 / new_norms[outside_mask])
            
        # Decay step size slightly
        step_size *= 0.995
        
    return best, best_r


def _anneal_and_refine(centers, deadline):
    """
    Alternate between local refinement and random perturbations (annealing) to escape local minima.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    temp = 0.05
    start_temp = 0.05
    end_temp = 0.001
    n_anneal_steps = 1000
    
    i = 0
    while time.time() < deadline:
        # Phase 1: Local Refinement
        # Run local search for a short duration or fixed number of iterations
        # We use a time slice for local search
        local_deadline = time.time() + 1.0
        current, current_r = _local_search(current, time_limit=min(1.0, deadline - time.time()))
        
        if current_r > best_r:
            best_r = current_r
            best = current.copy()
        
        # Phase 2: Annealing Step (Random Perturbation)
        # Pick a random point and move it
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = temp * np.random.exponential(1.0)
        
        old_x, old_y = current[idx, 0], current[idx, 1]
        new_x = old_x + magnitude * np.cos(angle)
        new_y = old_y + magnitude * np.sin(angle)
        
        # Check boundary
        if np.hypot(new_x, new_y) > 1.0:
            # Reject or project? Projecting might help escape boundary traps
            norm = np.hypot(new_x, new_y)
            new_x /= norm
            new_y /= norm
            
        current[idx, 0] = new_x
        current[idx, 1] = new_y
        
        new_r = _compute_radius(current)
        
        if new_r > current_r:
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = current_r - new_r
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / max(temp, 1e-10))
            if np.random.random() < accept_prob:
                current_r = new_r
            else:
                # Revert
                current[idx, 0] = old_x
                current[idx, 1] = old_y
        
        # Cool down
        i += 1
        temp = start_temp * (end_temp / start_temp) ** (i / n_anneal_steps)
        
        # If temperature is very low, restart with higher temp to explore more
        if temp < 0.005:
            temp = 0.02
            i = 0

    return best, best_r


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    
    # Initial refinement
    centers, r = _local_search(centers, time_limit=5.0)
    
    # Main optimization loop: Annealing + Local Refinement
    centers, r = _anneal_and_refine(centers, deadline)
    
    return centers
# EVOLVE-BLOCK-END
