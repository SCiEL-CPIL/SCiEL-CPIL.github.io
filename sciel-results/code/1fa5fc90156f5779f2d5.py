"""Improved packing for 200 equal circles in a unit square using Hex Grid Init and Simulated Annealing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _score(pts):
    """Compute the radius for a given set of points."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    
    # Compute min distance between any two points
    dists = pdist(pts)
    min_pair_dist = np.min(dists)
    
    # Compute min distance to boundary
    # Distance to x=-0.5: 0.5 - x
    # Distance to x=0.5: 0.5 + x
    # Distance to y=-0.5: 0.5 - y
    # Distance to y=0.5: 0.5 + y
    dists_to_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_boundary_dist = np.min(dists_to_boundary)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r


def _hex_grid_init():
    """Create a hexagonal grid initialization."""
    # Estimate spacing
    # Area per hex point in hex lattice is s * s * sqrt(3)/2
    # N * s^2 * sqrt(3)/2 = 1 => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    # Add a small margin to ensure we don't start with overlaps if N is slightly off
    # Actually, we want to fit N points.
    
    points = []
    # y step is s * sqrt(3)/2
    y_step = s * np.sqrt(3) / 2.0
    x_step = s
    
    y = -0.5
    row = 0
    # Iterate while y is within bounds + margin
    while y <= 0.5 + y_step:
        x = -0.5
        # Offset for even/odd rows
        offset = (x_step / 2.0) * (row % 2)
        while x <= 0.5 + x_step:
            px = x + offset
            py = y
            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                points.append([px, py])
            x += x_step
        y += y_step
        row += 1
    
    pts = np.array(points)
    
    if len(pts) == 0:
        # Fallback to square grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()

    if len(pts) > N:
        # Randomly select N points
        # This breaks symmetry and allows SA to find a better local optimum
        # Note: Random selection from a regular grid might leave holes, 
        # but SA is good at filling them.
        indices = np.random.choice(len(pts), N, replace=False)
        return pts[indices].copy()
    elif len(pts) < N:
        # Pad with random points
        extra = N - len(pts)
        random_pts = np.random.uniform(-0.49, 0.49, size=(extra, 2))
        return np.vstack([pts, random_pts])
    else:
        return pts.copy()


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (limit is 120s)
    
    # Initialization
    pts = _hex_grid_init()
    current_score = _score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    # Simulated Annealing Parameters
    T = 0.005  # Initial temperature
    T_min = 0.00001
    alpha = 0.9999  # Cooling rate
    # Move step size scaling
    # We want moves to be significant but not too large.
    # Radius is ~0.03. Step ~0.01 is good.
    step_scale = 2.0 
    
    rng = np.random.default_rng()
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Propose move
        # Step size proportional to temperature
        step_size = T * step_scale
        move = rng.normal(0, step_size, size=2)
        
        new_pts = pts.copy()
        new_pts[idx] += move
        
        # Keep in bounds (soft constraint in score, but clipping helps SA)
        # If we clip, we might stay on boundary. 
        # The score function handles out-of-bounds by returning 0.
        # It is better to let it go out of bounds and reject, or clip and accept if score improves?
        # Standard practice: Clip to valid range to keep state valid, score will be 0 if it was invalid.
        # But if we clip, the point stays at boundary. 
        # Let's just clip to [-0.5, 0.5].
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_score = _score(new_pts)
        
        # Acceptance
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            # Accept with probability exp((new - current) / T)
            # Note: new_score <= current_score
            delta = new_score - current_score
            if delta < 0:
                # Avoid overflow/underflow issues
                if delta / T > -50: # exp(-50) is ~1e-22
                    prob = np.exp(delta / T)
                else:
                    prob = 0.0
            else:
                prob = 1.0
            
            if rng.random() < prob:
                pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Random restart or kick?
            # For now, just keep going at T_min to polish.
            
    return best_pts
# EVOLVE-BLOCK-END
