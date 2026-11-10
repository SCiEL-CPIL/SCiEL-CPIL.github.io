"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(pts):
    """Compute the maximum radius for the given points."""
    # 1. Distance to walls: min_i min_k (0.5 - |c_ik|)
    # pts is (N, 3)
    # coords are in [-0.5, 0.5]
    # distance to wall is 0.5 - max(abs(x), abs(y), abs(z)) for each point? 
    # No, distance to boundary is min(0.5 - x, 0.5 + x) = 0.5 - |x|
    # So for a point, the clearance is min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
    
    # Efficient calculation of wall clearance
    # abs_pts = np.abs(pts)
    # wall_clearances = 0.5 - np.max(abs_pts, axis=1)
    # min_wall = np.min(wall_clearances)
    
    # 2. Pairwise distances
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=-1)
    # np.fill_diagonal(dists, np.inf)
    # min_dist = np.min(dists) / 2.0
    
    # return min(min_wall, min_dist)

    # Vectorized score calculation
    # Wall clearance
    abs_pts = np.abs(pts)
    max_abs = np.max(abs_pts, axis=1)
    wall_clearances = 0.5 - max_abs
    min_wall = np.min(wall_clearances)
    
    # Pairwise distances
    # Using scipy or numpy broadcasting
    # To save memory/time for N=50, full matrix is fine (50x50)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq) / 2.0
    
    return min(min_wall, min_dist)

def _relax(pts, max_iters=2000, step_init=0.05, step_decay=0.999, min_step=1e-5):
    """
    Local search using gradient-like steps for max-min distance.
    For each point, calculate the "repulsion" from the closest other point and walls.
    Move in the direction of the repulsion.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    step = step_init
    
    for i in range(max_iters):
        moved = False
        for j in range(N):
            # Calculate force on point j
            # 1. From walls
            # If x > 0.5 - r, push left. If x < -0.5 + r, push right.
            # Actually, we want to maximize min distance.
            # The "bottleneck" is the closest constraint.
            
            # Let's identify the closest neighbor and closest wall for point j.
            
            # Distances to all other points
            diff = pts - pts[j]
            dists_sq = np.sum(diff * diff, axis=1)
            dists_sq[j] = np.inf
            closest_idx = np.argmin(dists_sq)
            closest_dist = np.sqrt(dists_sq[closest_idx])
            
            # Wall clearance for j
            # d_wall = 0.5 - max(abs(pts[j]))
            # The direction to push is towards the center from the closest face.
            # If pts[j][0] is the max abs coord, and it's positive, push in -x.
            abs_j = np.abs(pts[j])
            max_abs_val = np.max(abs_j)
            wall_dist = 0.5 - max_abs_val
            wall_axis = np.argmax(abs_j)
            wall_sign = np.sign(pts[j][wall_axis])
            
            # Determine the limiting constraint for point j
            # We want to increase the minimum of (closest_dist/2, wall_dist)
            # Actually, the global score is min over all pairs and walls.
            # A simple heuristic: push away from the closest "obstacle" (neighbor or wall).
            
            # Let's define a "potential" or "force".
            # Force from neighbor: direction from neighbor to j, magnitude 1.
            # Force from wall: direction from wall to j (inwards), magnitude 1.
            
            # Which one is closer?
            # If closest_dist < 2 * wall_dist, neighbor is limiting.
            # Else wall is limiting?
            # Not exactly, because moving j might affect others.
            # But locally, moving away from the closest constraint is good.
            
            # Let's compute the "score contribution" if we move j.
            # Simpler: always move away from the closest neighbor, unless wall is closer.
            
            # Effective distance to constraint:
            # d_neighbor = closest_dist
            # d_wall = wall_dist
            
            # If we move by delta in direction v, new d_neighbor ~ d_neighbor + dot(delta, v_neighbor)
            # new d_wall ~ d_wall + dot(delta, v_wall)
            
            # We want to maximize min(d_neighbor/2, d_wall) ? No, score is min(d_neighbor/2, d_wall).
            # Wait, score = min( min_pair_dist/2, min_wall_dist ).
            # So for point j, its contribution to the min is limited by min(closest_dist/2, wall_dist).
            
            limit_val = min(closest_dist / 2.0, wall_dist)
            
            if limit_val < best_score:
                # This point is likely the bottleneck or close to it.
                pass # We still try to move it to improve local geometry
                
            # Determine direction
            if closest_dist < 2.0 * wall_dist:
                # Neighbor is closer (in terms of radius constraint)
                # Direction: from neighbor to j
                dir_vec = pts[j] - pts[closest_idx]
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-9:
                    dir_vec /= norm
                else:
                    dir_vec = np.array([1.0, 0.0, 0.0])
            else:
                # Wall is closer
                dir_vec = np.zeros(3)
                dir_vec[wall_axis] = -wall_sign # Push inwards
                # Normalize? It's already unit vector.
            
            # Try a small step
            new_pt = pts[j] + step * dir_vec
            
            # Check if new_pt is inside cube (with small margin to avoid invalid states)
            # If it goes outside, we can't take this step, or we should shrink step.
            if np.any(np.abs(new_pt) > 0.5):
                # Project back? Or just skip?
                # Skipping is safer.
                continue
            
            # Update and check global score
            old_pt = pts[j].copy()
            pts[j] = new_pt
            new_score = _score(pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
                moved = True
            else:
                pts[j] = old_pt
                
        if not moved:
            # Stuck, reduce step size
            step *= step_decay
            if step < min_step:
                break
                
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_pts = grid[:N].copy()
    
    best_pts = initial_pts.copy()
    best_score = _score(best_pts)
    
    # Run relaxation
    # We can run multiple restarts if time permits, but a single good relaxation from grid is a start.
    # Let's try a few random perturbations + relaxation if time allows.
    
    # First, just relax the grid
    current_pts, current_score = _relax(initial_pts, max_iters=5000)
    if current_score > best_score:
        best_score = current_score
        best_pts = current_pts
        
    # If time remains, try random restarts
    # Random points inside the cube
    while time.time() < deadline:
        # Generate random points
        rand_pts = np.random.uniform(-0.5, 0.5, size=(N, 3))
        # Relax
        # Use fewer iterations for random starts to fit more in budget
        rand_pts_relaxed, rand_score = _relax(rand_pts, max_iters=1000)
        
        if rand_score > best_score:
            best_score = rand_score
            best_pts = rand_pts_relaxed
            
    # Final check
    if time.time() > deadline - 5.0:
         # Ensure we return valid points
         # Clip to box just in case
         best_pts = np.clip(best_pts, -0.5, 0.5)
         
    return best_pts
# EVOLVE-BLOCK-END
