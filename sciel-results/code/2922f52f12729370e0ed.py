"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
# Safety margin: Evaluator kills at 150s. We aim to finish well before 120s.
# Let's target 110s for computation to be safe.
_DEADLINE = _START_TIME + 110.0


def _score(positions):
    """
    Compute the valid packing radius (min of pairwise dist/2 and dist to boundary).
    Returns 0 if invalid (non-finite, outside container).
    """
    n = len(positions)
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Check container constraint
    radii = np.hypot(positions[:, 0], positions[:, 1])
    if np.any(radii > 1.0):
        return 0.0
    
    min_boundary = 1.0 - np.max(radii)
    if min_boundary < 0:
        return 0.0
        
    if n < 2:
        return min_boundary

    # Pairwise distances
    # Using broadcasting for vectorization
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_boundary)


def _hex_lattice_init(n):
    """
    Generate initial positions on a hexagonal lattice, then select n closest to center.
    This provides a good, non-overlapping starting point.
    """
    # Generate a hexagonal grid of candidate points
    # We want a spacing that allows at least N points within radius 1.
    # Area of unit circle is ~3.14. Area of hex cell ~ s^2 * sqrt(3)/2.
    # For N=50, s ~ sqrt(3.14 / (50 * 0.866)) ~ sqrt(0.072) ~ 0.26
    # Let's try a few spacings and pick the one that yields enough points.
    
    best_points = None
    for spacing in [0.25, 0.22, 0.20, 0.18]:
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                # Keep points well inside the boundary to allow for optimization
                if r < 1.0 - spacing / 2.0:
                    points.append((x, y))
        
        if len(points) >= n:
            points = np.array(points)
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            if best_points is None or _score(points[idx]) > _score(best_points):
                best_points = points[idx].copy()
    
    if best_points is None:
        # Fallback to random
        theta = np.random.uniform(0, 2 * np.pi, n)
        rho = np.sqrt(np.random.uniform(0, 0.8, n))
        best_points = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
        
    return best_points


def _optimize(pts, max_time):
    """
    Optimize positions using vectorized force-directed layout with hard constraints.
    """
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    # Pre-compute indices for pairwise operations
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Adaptive parameters
    step = 0.01
    lr_decay = 0.9995
    stagnation_counter = 0
    patience = 2000 # Iterations without improvement before restart
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        if time.time() > _DEADLINE - 2.0:
            break
            
        # 1. Compute Forces
        # Vectorized pairwise differences
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)
        
        # Avoid division by zero
        d_safe = np.where(d < 1e-10, 1e-10, d)
        
        # Repulsive force: 1/d^4 (strong short-range repulsion)
        # We scale it to keep magnitudes manageable
        force_mag = 1.0 / (d2 * d2)
        
        fx = force_mag * dx / d_safe
        fy = force_mag * dy / d_safe
        
        # Accumulate forces on each particle
        Fx = np.zeros(n)
        Fy = np.zeros(n)
        
        # Force on i is -F (pushed away from j), Force on j is +F
        np.add.at(Fx, i_idx, -fx)
        np.add.at(Fy, i_idx, -fy)
        np.add.at(Fx, j_idx, fx)
        np.add.at(Fy, j_idx, fy)
        
        # 2. Update Positions
        new_pts = pts + step * np.column_stack([Fx, Fy])
        
        # 3. Hard Constraint Projection
        # Keep circles inside the unit disk.
        # The effective radius of the container for the center is (1 - r_current).
        # However, r_current changes. A robust way is to project based on the *current* score or a fixed target.
        # Using the current best score as the constraint radius is safer.
        r_constraint = 1.0 - best_score
        
        radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = radii > r_constraint
        if np.any(mask):
            # Project back to the boundary
            scale = r_constraint / radii[mask]
            new_pts[mask] *= scale[:, None]
            
        # 4. Evaluate Score
        current_score = _score(new_pts)
        
        if current_score > best_score + 1e-12:
            best_score = current_score
            best_pts = new_pts.copy()
            pts = new_pts
            stagnation_counter = 0
        else:
            # If no improvement, we still might want to keep moving if forces are strong,
            # but usually in these problems, we only accept improvements or do simulated annealing.
            # For simplicity and speed, we accept moves that don't decrease score too much,
            # but strictly speaking, hill climbing is safer for "maximize".
            # Let's do strict hill climbing: only accept if score improves.
            # If we don't accept, we revert pts to previous state?
            # Actually, if we revert, we are stuck. 
            # Better strategy: Always update pts to new_pts (even if score drops slightly) 
            # to allow navigating valleys, but keep best_pts.
            # However, with 1/d^4 forces, the system is very stiff. 
            # Let's try: Update pts always.
            pts = new_pts
            stagnation_counter += 1
            
            # Decay step size
            step *= lr_decay
            
            # Restart logic
            if stagnation_counter > patience:
                # Perturb from best
                pts = best_pts + np.random.randn(n, 2) * 0.01
                # Re-project
                radii = np.hypot(pts[:, 0], pts[:, 1])
                mask = radii > r_constraint
                if np.any(mask):
                    pts[mask] *= (r_constraint / radii[mask])[:, None]
                
                step = 0.005 # Smaller step for fine tuning
                stagnation_counter = 0
                patience = max(1000, patience // 2)
                
        # Ensure step doesn't get too small
        if step < 1e-6:
            step = 1e-6
            
    return best_pts, best_score


def construct():
    """
    Construct positions for N circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    # Initialize
    positions = _hex_lattice_init(N)
    
    # Optimize
    # Use most of the available time
    remaining_time = _DEADLINE - time.time() - 5.0 # 5s buffer for final checks
    if remaining_time < 1.0:
        # Not enough time for optimization, return init
        return positions
        
    positions, best_r = _optimize(positions, max_time=remaining_time)
    
    # Final validation and correction
    # Ensure all points are strictly inside
    radii = np.hypot(positions[:, 0], positions[:, 1])
    # Use a slightly conservative radius to avoid floating point issues at the boundary
    # The score is determined by the min distance. 
    # If a point is at 1.0 - r, it's on the boundary.
    # We want |c| + r <= 1.0 => |c| <= 1.0 - r.
    # The computed r is min(...). So 1.0 - r is the max allowed radius for centers.
    max_allowed_radius = 1.0 - _score(positions)
    
    mask = radii > max_allowed_radius
    if np.any(mask):
        # Project inward
        positions[mask] *= (max_allowed_radius / radii[mask])[:, None]
        
    return positions

# EVOLVE-BLOCK-END
