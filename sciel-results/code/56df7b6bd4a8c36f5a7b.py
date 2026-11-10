"""Optimized 50-circle packing using physics-inspired relaxation with fixed target radius."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def _relax_fixed_r(points, r_target, max_iters=1000, seed=42):
    """
    Relax points to satisfy packing constraints for a fixed target radius r_target.
    Uses force-directed dynamics.
    """
    rng = np.random.RandomState(seed)
    points = points.copy()
    velocities = np.zeros_like(points)
    
    # Parameters
    dt = 0.02
    damping = 0.95
    force_scale = 1.0
    
    for it in range(max_iters):
        forces = np.zeros_like(points)
        max_violation = 0.0
        
        # 1. Boundary Forces
        radii = np.linalg.norm(points, axis=1)
        margin = 1.0 - r_target
        # Violation if radius > margin
        viol = radii - margin
        mask = viol > 0
        if np.any(mask):
            max_violation = max(max_violation, np.max(viol[mask]))
            # Push inward
            # Normalized vector towards center
            # Avoid division by zero
            norms = radii[mask]
            norms[norms < 1e-8] = 1e-8
            # Force direction: -position / norm
            # Magnitude: proportional to violation
            f_mag = viol[mask] * force_scale
            f_vec = -points[mask] / norms[:, None] * f_mag[:, None]
            forces[mask] += f_vec
            
        # 2. Pairwise Forces
        # N=50, N^2/2 = 1225. Vectorized calculation is fast.
        # dist[i,j]
        dist = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        
        min_dist_required = 2.0 * r_target
        
        # Find pairs where dist < min_dist_required
        # We need to apply force to i and j.
        # To do this efficiently in vectorized form is tricky for the force accumulation.
        # Loop over i, but vectorize over j?
        
        for i in range(N):
            # Distances from i to all j
            d_i = dist[i]
            # Find j > i where d_i[j] < min_dist_required
            # Also handle j < i? 
            # If we loop i from 0 to N-1 and check all j, we double count.
            # Let's just check all j != i.
            
            # Mask for valid neighbors
            mask_j = (d_i < min_dist_required) & (np.arange(N) != i)
            
            if np.any(mask_j):
                indices_j = np.where(mask_j)[0]
                d_vals = d_i[indices_j]
                
                # Violations
                v = min_dist_required - d_vals
                max_violation = max(max_violation, np.max(v))
                
                # Force directions
                # points[i] - points[j]
                diff = points[i] - points[indices_j]
                # Normalize
                d_vals_safe = d_vals.copy()
                d_vals_safe[d_vals_safe < 1e-8] = 1e-8
                diff_norm = diff / d_vals_safe[:, None]
                
                # Force magnitudes
                f_mag = v * force_scale
                
                # Total force on i from these neighbors
                f_on_i = np.sum(diff_norm * f_mag[:, None], axis=0)
                
                # Apply to i
                forces[i] += f_on_i
                
                # Apply to j's (Newton's third law)
                # We need to subtract f_on_i from the respective j's
                # This is hard to vectorize back into the forces array easily without a loop over indices_j
                # But indices_j is small on average.
                for idx_j in indices_j:
                    # The force on j is -f_on_i_contrib
                    # We computed diff_norm for each j.
                    # We need to identify which diff_norm corresponds to which idx_j.
                    # Since we summed them, we lost the individual components.
                    # Let's re-calculate or store them.
                    # Better: Just loop over the small number of neighbors.
                    pass # Placeholder
                
                # Let's just do the inner loop for the few neighbors.
                # Since N is small, this is fine.
                for idx_j in indices_j:
                    d_val = d_i[idx_j]
                    if d_val < 1e-8:
                        continue
                    diff_j = points[i] - points[idx_j]
                    diff_norm_j = diff_j / d_val
                    v_j = min_dist_required - d_val
                    f_j = diff_norm_j * (v_j * force_scale)
                    forces[idx_j] -= f_j
                    
        # Update velocities and positions
        velocities = (velocities + forces * dt) * damping
        points += velocities * dt
        
        # Clamp to container
        radii_new = np.linalg.norm(points, axis=1)
        mask_out = radii_new > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii_new[mask_out, None]
            
        # Check convergence
        if max_violation < 1e-5:
            break
            
    return points

def _hex_init(spacing=0.22):
    """Generate hexagonal initialization."""
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    points = np.array(points)
    if len(points) > N:
        idx = np.random.RandomState(42).choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        rng = np.random.RandomState(42)
        while len(points) < N:
            r = rng.uniform(0, 0.9)
            a = rng.uniform(0, 2*np.pi)
            points = np.vstack([points, [r*np.cos(a), r*np.sin(a)]])
    return points[:N]

def construct():
    """
    Construct the optimal packing of 50 equal circles in the unit disk.
    """
    best_points = None
    best_radius = 0.0
    
    # Target radii to try
    # Best known is ~0.1258
    # We try a range of targets
    targets = [0.125, 0.124, 0.123, 0.120, 0.110]
    
    inits = [
        _hex_init(0.22),
        _hex_init(0.24),
        np.random.RandomState(1).uniform(-0.8, 0.8, (N, 2)),
        np.random.RandomState(2).uniform(-0.8, 0.8, (N, 2)),
    ]
    
    for init in inits:
        for target_r in targets:
            # Try to relax to this target
            pts = _relax_fixed_r(init, target_r, max_iters=500, seed=42)
            
            # Calculate actual radius
            dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist)
            
            radii = np.linalg.norm(pts, axis=1)
            min_margin = np.min(1.0 - radii)
            
            actual_r = min(min_dist / 2.0, min_margin)
            
            if actual_r > best_radius:
                best_radius = actual_r
                best_points = pts.copy()
                
    if best_points is None:
        best_points = np.zeros((N, 2))
        
    return best_points
# EVOLVE-BLOCK-END
