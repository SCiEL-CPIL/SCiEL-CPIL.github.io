"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hexagonal_grid(points_needed=200):
    """Generate a hexagonal grid of points centered at (0,0) covering the unit square."""
    # Hexagonal lattice spacing
    # We want to fit as many as possible, then pick the best ones.
    # Let's generate a large grid and then filter.
    
    # Approximate density: area per point ~ 0.0025 (for r=0.036, area=pi*r^2 ~ 0.0041, packing fraction ~ 0.9 -> 0.0045)
    # Side of square is 1. Area 1.
    # Let's generate points with spacing s.
    # For hex packing, vertical spacing is s * sqrt(3)/2.
    
    # Start with a guess for spacing, refine if needed.
    # Let's just generate a dense grid and pick the 200 that minimize some cost or are central.
    
    # Generate x, y ranges
    # s ~ 0.05 is too small? 1/0.05 = 20 points per side -> 400 points.
    # Let's generate 30x30 grid to be safe.
    
    s = 0.045 # Spacing
    x_coords = np.arange(-0.6, 0.6, s)
    y_coords = np.arange(-0.6, 0.6, s * np.sqrt(3) / 2)
    
    pts = []
    for i, y in enumerate(y_coords):
        x_offset = 0 if i % 2 == 0 else s / 2
        for x in x_coords:
            px = x + x_offset
            py = y
            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                pts.append((px, py))
                
    pts = np.array(pts)
    
    # If we have more than N, pick the N closest to the center?
    # Or pick a random subset?
    # Picking closest to center is a good heuristic for symmetry and boundary distance.
    if len(pts) > N:
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts = pts[indices[:N]]
    elif len(pts) < N:
        # Fallback to square grid if hex gen failed (shouldn't happen with s=0.045)
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:N]
        
    return pts

def _compute_radius(pts):
    """Compute max radius for given centers."""
    # Vectorized distance calculation
    # dist[i, j] = |pts[i] - pts[j]|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff**2).sum(axis=2))
    
    # Set diagonal to infinity
    np.fill_diagonal(dists, 1e10)
    
    min_pair_dist = dists.min()
    
    # Boundary distance
    min_bound = (0.5 - np.abs(pts).max(axis=1)).min()
    
    r = min(min_pair_dist / 2.0, min_bound)
    
    # Ensure points are strictly inside for validity in case of floating point errors on boundary
    # The score formula handles boundary, but we must not go outside.
    return r

def construct():
    t_start = time.time()
    deadline = t_start + 140.0 # 10s safety margin
    
    # 1. Initialization: Hexagonal grid
    pts = _hexagonal_grid(N).copy()
    
    # Ensure strictly inside
    pts = np.clip(pts, -0.4999, 0.4999)
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    np.random.seed(42)
    
    # 2. Local Search / Simulated Annealing
    # We will try to move one point at a time.
    # Step size decays over time.
    
    current_pts = best_pts.copy()
    current_r = best_r
    
    # Number of iterations we can afford
    # _compute_radius is O(N^2) = 40k ops. ~10-50 us.
    # 100,000 iterations should take ~5s.
    # We have 140s. We can do many more.
    # Let's do 50,000 iterations of "move one point".
    
    max_iters = 100000
    base_step = 0.02
    min_step = 0.001
    cooling_rate = 0.9999
    
    step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Decay step size
        step = max(min_step, step * cooling_rate)
        
        # Pick a random point
        idx = np.random.randint(N)
        orig = current_pts[idx].copy()
        
        # Propose new position
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        new_pos = orig + step * np.array([np.cos(angle), np.sin(angle)])
        
        # Clip to bounds
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Check if it's actually a move
        if np.allclose(new_pos, orig):
            continue
            
        # Evaluate
        temp_pts = current_pts.copy()
        temp_pts[idx] = new_pos
        temp_r = _compute_radius(temp_pts)
        
        # Accept if better
        if temp_r > current_r:
            current_pts[idx] = new_pos
            current_r = temp_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        else:
            # Simulated Annealing acceptance: accept worse moves with probability exp(-(delta_r)/T)
            # But we want to maximize radius. Delta_r = current_r - temp_r (positive if worse)
            # T should decay.
            # This is tricky to tune. Simple hill climbing is safer and often sufficient for this N.
            # Let's stick to hill climbing for robustness, maybe add small random restarts?
            pass
            
    # Final polish: try to optimize each point individually with smaller steps
    # This is a "coordinate descent" like refinement
    if time.time() < deadline - 10:
        for _ in range(1000): # Small number of passes
            if time.time() > deadline:
                break
            improved = False
            for idx in range(N):
                orig = best_pts[idx].copy()
                # Try 8 directions with small step
                for dx in [-0.001, 0, 0.001]:
                    for dy in [-0.001, 0, 0.001]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = orig + np.array([dx, dy])
                        new_pos = np.clip(new_pos, -0.4999, 0.4999)
                        temp_pts = best_pts.copy()
                        temp_pts[idx] = new_pos
                        temp_r = _compute_radius(temp_pts)
                        if temp_r > best_r:
                            best_pts[idx] = new_pos
                            best_r = temp_r
                            improved = True
            if not improved:
                break

    # Ensure final validity
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    
    return best_pts
# EVOLVE-BLOCK-END
