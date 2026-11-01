"""
Packing 100 equal spheres in a unit cube.
Strategy: Best Candidate Position (BCP) with Simulated Annealing.
"""

import numpy as np
import time

N = 100
TIME_LIMIT = 100.0  # seconds, safety margin before 120s limit

def _min_distance_to_walls(pts):
    """Compute min distance from each point to the cube walls."""
    # Cube is [-0.5, 0.5]^3
    # Distance to wall is 0.5 - max(abs(x), abs(y), abs(z))
    d = 0.5 - np.max(np.abs(pts), axis=1)
    return d

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    # pts shape (N, 3)
    # diff shape (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist

def _current_radius(pts):
    """Compute the current valid radius for the configuration."""
    # 1. Wall constraints
    wall_dists = _min_distance_to_walls(pts)
    r_wall = np.min(wall_dists)
    
    # 2. Pairwise constraints
    if len(pts) > 1:
        dists = _pairwise_distances(pts)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
    else:
        r_pair = np.inf
        
    return min(r_wall, r_pair)

def _find_bcp(pos, others, walls):
    """
    Find the Best Candidate Position for a single sphere.
    pos: current position (3,)
    others: positions of other spheres (M, 3)
    walls: min distance to walls for the current pos (scalar) - actually we need to re-evaluate
    
    We want to maximize min( min_j |p - p_j|, min_k (0.5 - |p_k|) )
    """
    # This is a local optimization problem for 3 variables.
    # We can use a simple local search or gradient ascent on the soft-min.
    # Given the small dimension (3), a dense local grid search or Nelder-Mead is fast.
    # However, doing this for every sphere in every SA step might be slow if not optimized.
    # Let's use a simple stochastic local search or a few steps of gradient ascent on a smooth approximation.
    
    # Smooth approximation: soft-min of distances
    # f(p) = -logsumexp(-dists / T) approx min(dists)
    # We want to maximize f(p).
    
    # To keep it fast, we'll do a small local search around the current position or random points.
    # Actually, a robust BCP move usually evaluates the objective at the current pos and a few candidates.
    # A common efficient heuristic: 
    # 1. Compute current r.
    # 2. Find the "bottleneck" constraint (closest wall or closest neighbor).
    # 3. Move slightly away from it.
    
    # Let's implement a more robust local search for the single point.
    # We will try to maximize the minimum distance.
    
    p = pos.copy()
    
    # Define objective to maximize: min(dist to others, dist to walls)
    def objective(x):
        # Dist to walls
        dw = 0.5 - np.max(np.abs(x))
        if dw < 0: return -1000 # Invalid
        
        # Dist to others
        if len(others) == 0:
            return dw
            
        d_others = np.linalg.norm(others - x, axis=1)
        return min(dw, np.min(d_others))
        
    # Simple local search with random restarts within a small radius
    best_val = objective(p)
    best_p = p.copy()
    
    # Try a few random perturbations
    for _ in range(20):
        # Random direction
        dir = np.random.randn(3)
        dir /= np.linalg.norm(dir)
        
        # Try different step sizes
        for step in [0.01, 0.05, 0.1]:
            new_p = p + dir * step
            # Check bounds
            if np.any(new_p < -0.5) or np.any(new_p > 0.5):
                continue
                
            val = objective(new_p)
            if val > best_val:
                best_val = val
                best_p = new_p.copy()
                p = new_p.copy() # Continue from here
                
    return best_p

def construct():
    start_time = time.time()
    
    # 1. Initialize with a grid
    # 100 spheres. 5x5x4 = 100.
    # Or 4x5x5.
    # Let's use a simple cubic-like distribution.
    # 5x5x4 grid in [-0.5, 0.5]
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts = pts[:N]
    
    # Ensure valid start
    r = _current_radius(pts)
    if r <= 0:
        # Fallback to random
        pts = np.random.uniform(-0.4, 0.4, (N, 3))
        r = _current_radius(pts)

    best_pts = pts.copy()
    best_r = r
    
    # 2. Simulated Annealing with BCP moves
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.995  # Cooling rate
    
    # Pre-compute indices
    indices = np.arange(N)
    
    steps = 0
    while time.time() - start_time < TIME_LIMIT:
        if T < T_min:
            break
            
        # Randomly select a sphere to move
        i = np.random.choice(N)
        
        # Save old state
        old_pos = pts[i].copy()
        old_r = r
        
        # Find BCP for sphere i
        others = np.delete(pts, i, axis=0)
        new_pos = _find_bcp(old_pos, others)
        
        # Update pts temporarily
        pts[i] = new_pos
        
        # Calculate new radius
        # Optimization: We don't need to recalculate ALL pairwise distances.
        # Only distances involving sphere i changed.
        # r_new = min(
        #   min_{j != i} |c_i - c_j| / 2,
        #   min_{j != i, k != i} |c_j - c_k| / 2,  # Unchanged
        #   min_k (0.5 - |c_{i,k}|)
        # )
        
        # Let's just recalculate fully for simplicity and correctness, 
        # but we can optimize if it's too slow.
        # With N=100, full recalc is 100*100/2 = 5000 dist calcs. 
        # 20 local searches * 5000 * 3 (norm) ~ 300k ops. 
        # 100 spheres * 1000 steps = 100k steps? No, 1 step per loop.
        # 100k steps * 300k ops = 3e10 ops. Too slow in Python loops.
        
        # We MUST optimize the radius calculation.
        
        # Restore pts if we are going to do it differently
        pts[i] = old_pos
        
        # Efficient Radius Update:
        # We maintain a matrix of pairwise distances? No, memory is fine (100x100).
        # But updating it is O(N).
        
        # Let's maintain dist_matrix.
        # dist_matrix[i, j] = |c_i - c_j|
        
        # Initialize dist_matrix
        dist_matrix = _pairwise_distances(pts)
        
        # Loop again with dist_matrix optimization
        break # Break out of this initial setup block, restart loop properly below

    # Re-initialize with dist_matrix
    dist_matrix = _pairwise_distances(pts)
    
    # Recalculate r from dist_matrix
    # r = min( min_off_diag(dist_matrix)/2, min_wall_dist )
    np.fill_diagonal(dist_matrix, np.inf)
    r = min(np.min(dist_matrix) / 2.0, np.min(_min_distance_to_walls(pts)))
    
    best_pts = pts.copy()
    best_r = r
    
    while time.time() - start_time < TIME_LIMIT:
        if T < T_min:
            break
            
        i = np.random.choice(N)
        old_pos = pts[i].copy()
        
        # Find BCP
        others = np.delete(pts, i, axis=0)
        new_pos = _find_bcp(old_pos, others)
        
        # Check if new_pos is valid (inside cube)
        if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
            continue
            
        # Update dist_matrix for row/col i
        old_dists_to_i = dist_matrix[i].copy()
        new_dists_to_i = np.linalg.norm(pts - new_pos, axis=1)
        
        # Calculate new r
        # The minimum distance involving i is min(new_dists_to_i[i], min(new_dists_to_i[j] for j!=i))
        # Wait, new_dists_to_i[i] is 0.
        # min_pair_i = min(new_dists_to_i[j] for j != i)
        
        # We need to check if the global minimum changed.
        # old_global_min = min( min(old_dists_to_i[j] for j!=i), min_{j<k, j,k!=i} dist[j,k] )
        
        # This is getting complicated to maintain exactly without full recalc.
        # Given N=100, full recalc of dist_matrix is 100x100 norm. 
        # np.linalg.norm on (100,100,3) is fast in C.
        # Let's just do full recalc of dist_matrix every step. It's likely fast enough.
        
        pts[i] = new_pos
        new_dist_matrix = _pairwise_distances(pts)
        
        np.fill_diagonal(new_dist_matrix, np.inf)
        new_r_pair = np.min(new_dist_matrix) / 2.0
        new_r_wall = np.min(_min_distance_to_walls(pts))
        new_r = min(new_r_pair, new_r_wall)
        
        # Acceptance
        if new_r > old_r or np.random.rand() < np.exp((new_r - old_r) / T):
            dist_matrix = new_dist_matrix
            r = new_r
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        else:
            # Revert
            pts[i] = old_pos
            # dist_matrix is not reverted, but we will recalc next step anyway if accepted.
            # If rejected, we keep old dist_matrix? No, we overwrote pts.
            # We should just recalc dist_matrix at the start of the loop if we revert?
            # Or better: only update dist_matrix if accepted.
            
            # Since we reverted pts, the dist_matrix variable still holds the OLD matrix 
            # IF we didn't update it. But we calculated new_dist_matrix.
            # We did NOT update the variable `dist_matrix` in the accepted branch.
            # So `dist_matrix` is still the old one. Correct.
            
        T *= alpha
        steps += 1
        
    # Final Polish: Local BCP moves without SA (greedy)
    improved = True
    while improved and (time.time() - start_time < TIME_LIMIT):
        improved = False
        for i in range(N):
            old_pos = pts[i].copy()
            others = np.delete(pts, i, axis=0)
            new_pos = _find_bcp(old_pos, others)
            if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                continue
            
            pts[i] = new_pos
            new_dist_matrix = _pairwise_distances(pts)
            np.fill_diagonal(new_dist_matrix, np.inf)
            new_r_pair = np.min(new_dist_matrix) / 2.0
            new_r_wall = np.min(_min_distance_to_walls(pts))
            new_r = min(new_r_pair, new_r_wall)
            
            if new_r > r + 1e-12:
                r = new_r
                dist_matrix = new_dist_matrix
                improved = True
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
            else:
                pts[i] = old_pos
                
    return best_pts

# Ensure the function is named construct
# The code above defines construct.

# One optimization: _find_bcp is called frequently.
# The "others" array creation `np.delete` is O(N).
# We can pass indices instead to avoid copying.
# But for N=100, it's negligible.

# Let's refine _find_bcp to be faster.
# Instead of random restarts, use the current position and move away from the closest constraint.

def _find_bcp_fast(pos, others):
    """
    Faster BCP: Move away from the closest neighbor/wall.
    """
    p = pos.copy()
    
    # Find closest neighbor
    if len(others) > 0:
        dists = np.linalg.norm(others - p, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        closest_neighbor = others[min_idx]
        
        # Find closest wall
        wall_dists = 0.5 - np.abs(p)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # Determine the limiting constraint
        if min_dist < min_wall_dist:
            # Limited by neighbor
            direction = p - closest_neighbor
            if np.linalg.norm(direction) < 1e-6:
                direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            # Move in that direction
            step = 0.01
            new_p = p + direction * step
        else:
            # Limited by wall
            # Move away from the wall
            direction = np.zeros(3)
            direction[min_wall_idx] = -np.sign(p[min_wall_idx])
            if direction[min_wall_idx] == 0:
                direction[min_wall_idx] = 1
            # Normalize
            direction /= np.linalg.norm(direction)
            step = 0.01
            new_p = p + direction * step
            
        # Ensure new_p is inside
        if np.any(new_p < -0.5) or np.any(new_p > 0.5):
            return p
            
        # Check if this move improves the local min distance
        # We need to check min distance to ALL others and walls for new_p
        # This is O(N)
        def local_min_dist(x):
            dw = 0.5 - np.max(np.abs(x))
            if dw < 0: return -1
            if len(others) == 0: return dw
            do = np.min(np.linalg.norm(others - x, axis=1))
            return min(dw, do)
            
        old_val = local_min_dist(p)
        new_val = local_min_dist(new_p)
        
        if new_val > old_val:
            return new_p
        else:
            return p
    else:
        return p

# Replace _find_bcp with _find_bcp_fast in construct for speed
# I will rewrite construct to use the fast version.

def construct():
    start_time = time.time()
    
    # Initialization
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts = pts[:N]
    
    r = _current_radius(pts)
    best_pts = pts.copy()
    best_r = r
    
    T = 0.01
    T_min = 1e-6
    alpha = 0.995
    
    while time.time() - start_time < TIME_LIMIT:
        if T < T_min:
            break
            
        i = np.random.choice(N)
        old_pos = pts[i].copy()
        old_r = r
        
        others = np.delete(pts, i, axis=0)
        new_pos = _find_bcp_fast(old_pos, others)
        
        if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
            continue
            
        pts[i] = new_pos
        
        # Recalculate radius
        new_dist_matrix = _pairwise_distances(pts)
        np.fill_diagonal(new_dist_matrix, np.inf)
        new_r_pair = np.min(new_dist_matrix) / 2.0
        new_r_wall = np.min(_min_distance_to_walls(pts))
        new_r = min(new_r_pair, new_r_wall)
        
        if new_r > old_r or np.random.rand() < np.exp((new_r - old_r) / T):
            r = new_r
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        else:
            pts[i] = old_pos
            
        T *= alpha
        
    # Greedy polish
    improved = True
    while improved and (time.time() - start_time < TIME_LIMIT):
        improved = False
        for i in range(N):
            old_pos = pts[i].copy()
            others = np.delete(pts, i, axis=0)
            new_pos = _find_bcp_fast(old_pos, others)
            if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                continue
            
            pts[i] = new_pos
            new_dist_matrix = _pairwise_distances(pts)
            np.fill_diagonal(new_dist_matrix, np.inf)
            new_r_pair = np.min(new_dist_matrix) / 2.0
            new_r_wall = np.min(_min_distance_to_walls(pts))
            new_r = min(new_r_pair, new_r_wall)
            
            if new_r > r + 1e-12:
                r = new_r
                improved = True
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
            else:
                pts[i] = old_pos
                
    return best_pts
