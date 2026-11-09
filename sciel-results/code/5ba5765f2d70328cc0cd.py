"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_radius(points):
    """Compute the packing radius for a set of points."""
    if len(points) < 2:
        return np.inf
    # Minimum distance between any two points
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = np.min(dist_matrix)
    # Minimum distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    return min(min_dist / 2.0, min_boundary)

def _hexagonal_lattice(n_target):
    """Generate a hexagonal lattice packing inside the unit disk as initial guess."""
    # Estimate spacing based on area density
    # Area of unit disk = pi. N circles of radius r -> N*pi*r^2 <= pi -> r <= 1/sqrt(N)
    # Distance d = 2r. Hex lattice density is pi/sqrt(12) ~ 0.9069
    # So effective area per circle in hex lattice is ~ 1/0.9069 * (pi*r^2) ? 
    # Simpler: Just start with a reasonable d and refine.
    d = 2.0 / (np.sqrt(n_target) * 0.95) # Slightly tighter than max theoretical
    
    points = []
    # Generate grid large enough to cover unit disk
    max_range = int(2.0 / d) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * d + (j % 2) * d / 2.0
            y = j * d * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # Sort by distance from origin to keep the most central points
    if len(points) >= n_target:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:n_target]
        return points[indices]
    
    # Fallback if too few points (should not happen with this range)
    # If we don't have enough, we just return what we have or random points
    # But for N=300, d~0.11, range~20, points~1200, so it's fine.
    return points[:n_target] if len(points) == n_target else None

def _langevin_dynamics(points, start_time, deadline):
    """
    Optimize points using Langevin Dynamics with cooling.
    Objective: Maximize min distance.
    Proxy Objective: Minimize -sum(exp(-k * dist^2)) or similar soft min.
    Here we use a force-based approach:
    Force on i from j: Repulsive if too close, but we want to maximize the MIN distance.
    
    A better formulation for Langevin:
    We want to push all points away from each other and towards the center (or keep them in).
    Actually, we just want to maximize the minimum distance.
    Let D be the current minimum distance.
    We can use a potential energy U = -sum_{i<j} f(d_ij) where f is increasing.
    Or simply: Force = gradient of soft-min.
    
    Let's use a standard "repulsive" force model where the strength depends on how "critical" the pair is.
    If d_ij < D_current * margin, push them apart.
    
    Simpler and robust:
    Use a potential V = -1/d_ij for d_ij > 0. This maximizes all distances, which tends to maximize the minimum.
    However, 1/d is singular. Use 1/(d^2 + eps).
    
    Let's try:
    Force on i from j: (p_i - p_j) / |p_i - p_j|^3 * strength
    Boundary Force: -p_i / |p_i|^2 * strength_boundary (pushes inward if outside, or just keeps inside?)
    Actually, we need to keep them INSIDE.
    If |p_i| > 1, force = -p_i * strength_boundary.
    
    To specifically target the MINIMUM distance (which determines the score), we should bias the forces
    towards the pairs that are currently closest.
    
    Algorithm:
    1. Compute all pairwise distances.
    2. Identify the "critical" pairs (those with distance < threshold).
    3. Apply strong repulsive force to critical pairs.
    4. Apply weak repulsive force to all pairs to maintain general spacing.
    5. Apply boundary constraint.
    6. Update velocities and positions with damping (Langevin).
    """
    points = points.astype(np.float64).copy()
    n = len(points)
    
    # Initial temperature/step size
    dt = 0.01
    damping = 0.9
    # Cooling schedule
    temp_start = 0.05
    temp_end = 0.001
    total_time = deadline - start_time
    if total_time <= 0:
        return points
    
    # Pre-compute indices for pairs? No, N=300, N^2=90k, fine to compute each step.
    
    # We will run for as many steps as possible within time limit
    steps = 0
    last_improve = start_time
    best_radius = _compute_radius(points)
    best_points = points.copy()
    
    # Use a "soft" force to avoid singularities
    eps = 1e-4
    
    while time.time() < deadline - 2.0: # 2s safety margin
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise interactions
        # Vectorized distance calculation
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We want to maximize the minimum distance.
        # Current min distance
        current_min_dist = np.min(dists)
        
        # Define a threshold for "active" repulsion
        # Only push apart pairs that are close to the current minimum or below a global target
        # To be efficient, we can just use a potential that is always repulsive but stronger when close.
        # Potential V = 1 / (d^2 + eps)
        # Force = -grad V = 2 * diff / (d^2 + eps)^2  (direction: away from each other)
        # Wait, V = 1/d^2. dV/dd = -2/d^3. Force = -dV/dd * unit_vec = 2/d^3 * unit_vec.
        # unit_vec = diff / d. So Force = 2 * diff / d^4.
        
        # Let's use a softer potential to allow larger steps.
        # F = diff / (d^2 + eps)^1.5  (like Coulomb but softened)
        
        # To focus on the bottleneck (minimum distance), we can weight forces by how "small" the distance is relative to the max distance.
        # Or simply, use the standard repulsion.
        
        # Calculate forces
        # Avoid division by zero
        dists_sq = dists * dists
        # Force magnitude factor
        # We want to push apart.
        # F_ij = diff_ij * factor
        factor = 1.0 / (dists_sq + eps) ** 1.5
        np.fill_diagonal(factor, 0.0)
        
        # Force on i is sum over j of (p_i - p_j) * factor_ij
        # diff is p_i - p_j
        forces += np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Boundary force: Keep points inside unit circle
        # If |p| > 1, push in. If |p| < 1, maybe pull in slightly to prevent clustering at edge?
        # Actually, we just need to satisfy the constraint.
        # Hard constraint: if |p| > 1, project back and kill velocity? 
        # Soft constraint: Force = -p * (|p| - 1) * k_b if |p| > 1.
        
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        outside_mask = radii > 1.0
        if np.any(outside_mask):
            # Strong push back
            # Force direction: -p
            # Magnitude: proportional to how far outside
            boundary_force[outside_mask] = -points[outside_mask] * (radii[outside_mask] - 1.0) * 10.0
            
        # Also, to maximize the radius, we want points to be as far from center as possible?
        # No, we want the MIN distance to boundary to be large.
        # So we want to push points AWAY from the boundary?
        # The constraint is min_i (1 - |c_i|). To maximize this, we want |c_i| to be small?
        # No, wait.
        # Score is min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) ).
        # If a point is at the boundary, 1 - |c_i| = 0. Score is 0.
        # So we MUST keep points strictly inside.
        # To maximize the score, we need BOTH the inter-point distances AND the distance to boundary to be large.
        # This means we want to "inflate" the packing.
        # So, we should apply a force pushing points AWAY from the boundary (inwards) and AWAY from each other.
        
        # Revised Boundary Force:
        # We want to maximize min(1 - |p_i|).
        # This is equivalent to minimizing max(|p_i|).
        # So we should push all points towards the center?
        # If we push all to center, they collide.
        # The optimal configuration balances the inter-point repulsion and the "centripetal" force to keep them away from the edge.
        
        # Let's add a weak force pushing all points towards the origin.
        # F_center = -points * k_c
        k_c = 0.1 # Tuning parameter
        forces += -points * k_c
        
        # Update velocities (implicit in position update for simple Langevin)
        # v = (v + F*dt) * damping
        # p = p + v*dt
        
        # We don't store velocity explicitly, we can use a simple gradient ascent with momentum.
        # Or just: p_new = p + forces * dt + noise
        
        # Add noise for exploration (Langevin)
        # Noise magnitude should decrease over time (cooling)
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / total_time)
        current_temp = temp_start * (1.0 - progress) + temp_end * progress
        
        noise = np.random.normal(0, current_temp, size=points.shape)
        
        # Update
        points = points + forces * dt + noise
        
        # Hard constraint: Clip points to unit disk
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            points[mask] = points[mask] / radii[mask, np.newaxis]
        
        # Check improvement
        if steps % 10 == 0:
            r = _compute_radius(points)
            if r > best_radius + 1e-7:
                best_radius = r
                best_points = points.copy()
                last_improve = time.time()
            
            # Adaptive step size?
            # If radius decreased significantly, reduce dt?
            # Or just keep it simple.
            
        steps += 1
        
        # Check for stagnation?
        # If no improvement for a long time, maybe restart with different initial guess?
        # For now, rely on noise to escape.
        
        # Time check
        if time.time() > deadline - 2.0:
            break
            
    return best_points

def construct():
    """Construct centers of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # 1. Generate initial guess
    init_points = _hexagonal_lattice(N)
    
    if init_points is None or len(init_points) != N:
        # Fallback to random points if lattice fails
        # Random points in unit disk
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        init_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # Ensure initial points are valid (inside disk)
    radii = np.linalg.norm(init_points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        init_points[mask] = init_points[mask] / radii[mask, np.newaxis]
        
    best_points = init_points.copy()
    best_radius = _compute_radius(best_points)
    
    # 2. Run Langevin Dynamics
    # We might want to run multiple restarts if time permits
    # But Langevin is slow. Let's run one long run.
    
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        optimized_points = _langevin_dynamics(best_points, start_time, deadline)
        r_opt = _compute_radius(optimized_points)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_points = optimized_points
    
    # Final check: ensure all points are strictly inside
    radii = np.linalg.norm(best_points, axis=1)
    # If any point is on the boundary (radius >= 1), push it in slightly
    # The evaluator rejects points on or outside.
    # "A centre on or outside the container ... make the candidate invalid."
    # So we need |c_i| < 1.
    # Let's clip to 0.999999 to be safe.
    mask = radii >= 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / radii[mask, np.newaxis] * 0.999999
        
    # Also check for coincident centers?
    # "coincident centers ... make the candidate invalid"
    # Langevin should prevent this, but let's verify.
    # If two points are identical, distance is 0, radius is 0.
    # If score is 0, it's valid but bad.
    # The problem says "coincident centers ... invalid".
    # So we must ensure |c_i - c_j| > 0.
    # With floating point, exact coincidence is rare unless initialized that way.
    # Hex lattice has distinct points. Langevin moves them.
    
    return best_points
# EVOLVE-BLOCK-END
