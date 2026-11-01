"""
Strategy: Greedy Local Search (GLS) with Hard Constraint Projection.
Instead of simulating physics with smooth potentials, we directly attack the 
bottlenecks of the packing. The score is determined by the minimum distance 
between any pair of spheres or the minimum distance to a wall. 
We iteratively identify the pair of spheres (or sphere-wall) with the smallest 
separation and apply a large corrective displacement to them. 
This is a form of "best-improve" local search. To escape local minima, 
we periodically apply random "kicks" to the configuration and re-optimize.
This approach is structurally different from MD: it does not use forces 
from all pairs, but focuses computational effort on the most violated 
constraints, which is more efficient for maximizing the minimum distance.
"""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def _score(pos):
        """
        Compute the actual packing score.
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

    def _find_min_constraint(pos):
        """
        Find the pair of indices (i, j) or (i, -1) for wall that has the minimum separation.
        Returns (score, i, j) where j=-1 indicates a wall constraint for particle i.
        """
        # Pairwise
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        min_pair_dist_sq = np.min(dist_sq)
        min_pair_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # Walls
        wall_dists = 0.5 - np.abs(pos)
        min_wall_dist = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        # Compare
        # Score is min(min_pair_dist / 2, min_wall_dist)
        pair_score = min_pair_dist / 2.0
        wall_score = min_wall_dist
        
        if pair_score < wall_score:
            return pair_score, min_pair_idx[0], min_pair_idx[1]
        else:
            return wall_score, min_wall_idx[0], -1

    def _init_grid():
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def _init_bcc():
        m = 4
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

    def _optimize(pos, max_steps, step_scale=0.1):
        """
        Perform GLS optimization on the given position.
        """
        current_pos = pos.copy()
        best_score = _score(current_pos)
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
                
            score, i, j = _find_min_constraint(current_pos)
            
            # If we've hit a very high score, we can stop or continue polishing
            # The step size should be proportional to the "violation" if we were minimizing,
            # but here we are maximizing. We just want to increase the minimum distance.
            # So we push the closest pair apart by a fixed amount.
            
            # Determine direction
            if j != -1:
                # Pair constraint
                vec = current_pos[i] - current_pos[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-12:
                    vec = np.random.normal(0, 1, 3)
                vec /= np.linalg.norm(vec)
                
                # Push i away from j, j away from i
                # Step size: try to increase the distance. 
                # If we push by delta, distance increases by approx 2*delta.
                # We want to make this pair no longer the bottleneck.
                # A heuristic step size: 0.5 * current_min_dist? No, that's too big.
                # Let's use a fixed small step relative to the box size.
                delta = 0.02
                
                current_pos[i] += vec * delta
                current_pos[j] -= vec * delta
            else:
                # Wall constraint
                axis = np.argmin(np.abs(current_pos[i])) # This is wrong, we need the specific axis from _find_min_constraint
                # Let's re-calculate axis properly
                # Actually _find_min_constraint returns index in flattened array.
                # Let's fix _find_min_constraint to return axis for walls.
                pass

        return current_pos, best_score

    # Refined _find_min_constraint to return axis for walls
    def _find_min_constraint_v2(pos):
        # Pairwise
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        min_pair_dist_sq = np.min(dist_sq)
        min_pair_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # Walls
        wall_dists = 0.5 - np.abs(pos)
        min_wall_dist = np.min(wall_dists)
        min_wall_idx_flat = np.argmin(wall_dists)
        min_wall_i, min_wall_axis = np.unravel_index(min_wall_idx_flat, wall_dists.shape)
        
        pair_score = min_pair_dist / 2.0
        wall_score = min_wall_dist
        
        if pair_score < wall_score:
            return pair_score, min_pair_idx[0], min_pair_idx[1], None
        else:
            return wall_score, min_wall_i, -1, min_wall_axis

    def _optimize_v2(pos, max_steps):
        current_pos = pos.copy()
        best_score = _score(current_pos)
        best_pos = current_pos.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
                
            score, i, j, axis = _find_min_constraint_v2(current_pos)
            
            if j != -1:
                vec = current_pos[i] - current_pos[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-12:
                    vec = np.random.normal(0, 1, 3)
                vec /= np.linalg.norm(vec)
                
                # Adaptive step: if the gap is very small, take a small step.
                # If the gap is larger, we can take a bigger step to quickly resolve.
                # But we must not overshoot and create new violations.
                # Let's use a fixed step for simplicity first, then tune.
                delta = 0.015
                
                current_pos[i] += vec * delta
                current_pos[j] -= vec * delta
            else:
                # Wall constraint
                # Push i towards center along axis
                sign = np.sign(current_pos[i, axis])
                if sign == 0:
                    sign = 1
                delta = 0.015
                current_pos[i, axis] -= sign * delta
            
            # Clip to bounds
            current_pos = np.clip(current_pos, -0.5, 0.5)
            
            # Check score
            current_score = _score(current_pos)
            if current_score > best_score:
                best_score = current_score
                best_pos = current_pos.copy()
                
        return best_pos, best_score

    def _kick(pos, strength=0.1):
        """Apply a random perturbation to escape local minima."""
        new_pos = pos.copy()
        # Randomly select a subset of particles to kick
        kick_idx = np.random.choice(N, size=N//2, replace=False)
        new_pos[kick_idx] += np.random.normal(0, strength, (len(kick_idx), 3))
        return np.clip(new_pos, -0.5, 0.5)

    # Main Loop
    best_pos = None
    best_score = -1.0
    
    inits = [
        _init_grid(),
        _init_bcc(),
        np.random.uniform(-0.45, 0.45, (N, 3))
    ]
    
    # We will run multiple restarts.
    # Each restart: Init -> Optimize -> Kick -> Optimize -> ...
    
    max_restarts = 20
    max_steps_per_opt = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pos = inits[0].copy()
        elif restart == 1:
            pos = inits[1].copy()
        elif restart == 2:
            pos = inits[2].copy()
        else:
            # Kick the previous best or a random one
            if best_pos is not None and np.random.random() < 0.5:
                pos = _kick(best_pos, strength=0.05)
            else:
                pos = np.random.uniform(-0.45, 0.45, (N, 3))
                
        # Optimization Phase
        pos, score = _optimize_v2(pos, max_steps_per_opt)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            
        # If we still have time, do a few kick-and-optimize cycles on this best
        # This is a local search around the best found so far
        for kick_iter in range(5):
            if time.time() > deadline - 5.0:
                break
            if best_pos is None:
                break
                
            # Kick
            kicked_pos = _kick(best_pos, strength=0.02)
            # Optimize
            new_pos, new_score = _optimize_v2(kicked_pos, max_steps_per_opt // 2)
            
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()

    if best_pos is None:
        return _init_grid()
        
    return best_pos
# EVOLVE-BLOCK-END
