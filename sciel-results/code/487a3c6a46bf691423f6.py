"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def _score(pos):
        """
        Compute the actual packing score: min(min_pair_dist/2, min_wall_dist).
        pos: (N, 3) array
        """
        if pos.shape != (N, 3):
            return -1.0
        if not np.all(np.isfinite(pos)):
            return -1.0
        
        # Pairwise distances
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Wall distances
        wall_dists = 0.5 - np.abs(pos)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def _greedy_construct(radius, attempts=1000):
        """
        Attempt to place N spheres of given radius using greedy random insertion.
        Returns positions if successful, else None.
        """
        pos = []
        # Pre-compute candidate points? No, just random sampling.
        # To make it faster, we can use a grid of candidates or rejection sampling.
        # For N=100, rejection sampling might be slow if radius is large.
        # Let's try a simple rejection sampling with a time limit.
        
        # We need to check distance to all existing points.
        # O(N^2) per insertion is 100^2 = 10000 ops. 100 insertions = 1M ops.
        # This is fast in NumPy if vectorized, but we are doing it sequentially.
        # Let's vectorize the check against existing points.
        
        for _ in range(attempts):
            if len(pos) >= N:
                break
            # Try to place a point
            p = np.random.uniform(-0.5 + radius, 0.5 - radius, 3)
            
            if len(pos) == 0:
                pos.append(p)
                continue
            
            existing = np.array(pos)
            dists = np.linalg.norm(existing - p, axis=1)
            if np.min(dists) >= 2 * radius:
                pos.append(p)
        
        if len(pos) == N:
            return np.array(pos)
        return None

    def _local_polish(pos, max_iters=5000):
        """
        Local search to improve the minimum distance.
        We identify the "critical" points: those involved in the minimum pair distance 
        or the minimum wall distance. We try to move them to increase the minimum.
        
        Strategy:
        1. Calculate current min pair dist and min wall dist.
        2. Identify the pair(s) and wall(s) achieving the minimum.
        3. Attempt small moves to increase these specific distances.
        4. Accept if the global score improves.
        """
        pos = pos.copy()
        current_score = _score(pos)
        
        # Pre-calculate distances for efficiency in the loop? 
        # For N=100, recalculating full distance matrix is 10k ops.
        # 5000 iters * 10k = 50M ops. Should be fine in Python/NumPy if optimized.
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Calculate distances
            d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(d**2, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            dists = np.sqrt(dist_sq)
            
            min_pair_dist = np.min(dists)
            # Find indices of min pair
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i1, i2 = min_idx[0], min_idx[1]
            
            # Wall distances
            wall_dists = 0.5 - np.abs(pos)
            min_wall_dist = np.min(wall_dists)
            # Find index of min wall dist
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i3, axis3 = min_wall_idx[0], min_wall_idx[1]
            
            # Determine which constraint is active
            r_pair = min_pair_dist / 2.0
            r_wall = min_wall_dist
            
            # We want to maximize min(r_pair, r_wall).
            # If r_pair < r_wall, we should push i1 and i2 apart.
            # If r_wall < r_pair, we should push i3 away from the wall.
            # If equal, try both.
            
            improved = False
            step_size = 0.001
            
            # Strategy 1: Push pair apart
            if r_pair <= r_wall:
                # Direction from i1 to i2
                vec = pos[i2] - pos[i1]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    dir_vec = vec / norm
                else:
                    dir_vec = np.random.randn(3)
                    dir_vec /= np.linalg.norm(dir_vec)
                
                # Try moving i1 in -dir_vec and i2 in +dir_vec
                new_pos = pos.copy()
                new_pos[i1] -= dir_vec * step_size
                new_pos[i2] += dir_vec * step_size
                
                # Clip to bounds
                new_pos = np.clip(new_pos, -0.5, 0.5)
                
                new_score = _score(new_pos)
                if new_score > current_score + 1e-9:
                    pos = new_pos
                    current_score = new_score
                    improved = True
            
            # Strategy 2: Push away from wall
            if not improved and r_wall <= r_pair:
                # Determine which wall is closest for i3, axis3
                coord = pos[i3, axis3]
                if coord > 0:
                    # Close to +0.5 wall, move to -
                    move_dir = -1.0
                else:
                    # Close to -0.5 wall, move to +
                    move_dir = 1.0
                
                new_pos = pos.copy()
                new_pos[i3, axis3] += move_dir * step_size
                new_pos = np.clip(new_pos, -0.5, 0.5)
                
                new_score = _score(new_pos)
                if new_score > current_score + 1e-9:
                    pos = new_pos
                    current_score = new_score
                    improved = True
            
            if not improved:
                # If no improvement, try a random perturbation of a random point
                # to escape local minima.
                idx = np.random.randint(N)
                new_pos = pos.copy()
                new_pos[idx] += np.random.randn(3) * 0.01
                new_pos = np.clip(new_pos, -0.5, 0.5)
                
                new_score = _score(new_pos)
                if new_score > current_score + 1e-9:
                    pos = new_pos
                    current_score = new_score
                    improved = True
                else:
                    # Simulated annealing style: accept worse moves with low probability?
                    # Or just break if stuck for a while?
                    # For simplicity, let's try a slightly larger random jump.
                    if i % 100 == 0:
                        new_pos = pos.copy()
                        idx = np.random.randint(N)
                        new_pos[idx] += np.random.randn(3) * 0.05
                        new_pos = np.clip(new_pos, -0.5, 0.5)
                        new_score = _score(new_pos)
                        if new_score > current_score + 1e-9:
                            pos = new_pos
                            current_score = new_score

        return pos, current_score

    def _init_bcc():
        """
        Body-Centered Cubic lattice initialization.
        """
        m = 4
        scale = 1.0 / m
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
        
        pts = np.array(pts)
        pts /= m
        pts -= 0.5
        
        idx = np.random.choice(len(pts), N, replace=False)
        return pts[idx]

    best_pos = None
    best_score = -1.0
    
    # 1. Try to construct with increasing radius using greedy method
    # Theoretical max is ~0.111. Let's try to construct with r=0.1, 0.105, 0.11
    # If construction fails, we fall back to BCC init and polish.
    
    radii_to_try = [0.05, 0.08, 0.10, 0.105, 0.11]
    
    for r in radii_to_try:
        if time.time() > deadline - 30.0:
            break
            
        pos = _greedy_construct(r)
        if pos is not None:
            # Polish it
            polished_pos, score = _local_polish(pos)
            if score > best_score:
                best_score = score
                best_pos = polished_pos.copy()
                
    # 2. If greedy failed or didn't produce a good result, use BCC init and polish
    if best_score < 0.1:
        pos = _init_bcc()
        polished_pos, score = _local_polish(pos, max_iters=10000)
        if score > best_score:
            best_score = score
            best_pos = polished_pos.copy()

    # 3. Final random restarts if time permits
    restarts = 5
    for _ in range(restarts):
        if time.time() > deadline - 5.0:
            break
            
        pos = _init_bcc()
        # Add small random noise to break symmetry
        pos += np.random.randn(N, 3) * 0.01
        pos = np.clip(pos, -0.5, 0.5)
        
        polished_pos, score = _local_polish(pos, max_iters=5000)
        if score > best_score:
            best_score = score
            best_pos = polished_pos.copy()

    if best_pos is None:
        # Fallback: Grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
        
    return best_pos
# EVOLVE-BLOCK-END
