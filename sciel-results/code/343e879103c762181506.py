# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)

    def optimize_repulsive(positions, iterations=1000, lr_start=0.05, lr_end=0.001):
        """
        Optimize positions using repulsive forces.
        """
        pos = positions.copy()
        N = len(pos)
        
        # Precompute indices for pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        for it in range(iterations):
            # Decay learning rate
            lr = lr_start * (lr_end / lr_start) ** (it / iterations)
            
            # 1. Pairwise Repulsion
            diffs = pos[i_idx] - pos[j_idx]
            dists_sq = np.sum(diffs**2, axis=1)
            dists_sq = np.maximum(dists_sq, 1e-12)
            dists = np.sqrt(dists_sq)
            
            force_magnitudes = 1.0 / dists_sq
            unit_vecs = diffs / dists[:, np.newaxis]
            
            forces_i = unit_vecs * force_magnitudes[:, np.newaxis]
            forces_j = -forces_i
            
            forces = np.zeros_like(pos)
            np.add.at(forces, i_idx, forces_i)
            np.add.at(forces, j_idx, forces_j)
            
            # 2. Wall Repulsion
            dist_left = pos + 0.5
            dist_left_sq = np.maximum(dist_left**2, 1e-12)
            force_left = 1.0 / dist_left_sq
            forces += force_left[:, np.newaxis]
            
            dist_right = 0.5 - pos
            dist_right_sq = np.maximum(dist_right**2, 1e-12)
            force_right = 1.0 / dist_right_sq
            forces -= force_right[:, np.newaxis]
            
            # Update positions
            pos += lr * forces
            pos = np.clip(pos, -0.499999, 0.499999)
            
        return pos

    def refine_local(positions, steps=100):
        """
        Refine by iteratively pushing the most constrained pair or wall.
        """
        pos = positions.copy()
        best_r = compute_min_radius(pos)
        
        for _ in range(steps):
            # Find current min radius constraint
            dists = pdist(pos, 'euclidean')
            min_pair_dist = np.min(dists)
            min_pair_r = min_pair_dist / 2.0
            
            walls = 0.5 - np.abs(pos)
            min_wall_dist = np.min(walls)
            
            if min_pair_r < min_wall_dist:
                # Find the pair
                min_idx = np.argmin(dists)
                # Convert pdist index to i, j
                # For N=50, brute force is fast enough
                min_d = np.inf
                min_i, min_j = 0, 1
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < min_d:
                            min_d = d
                            min_i, min_j = i, j
                            
                diff = pos[min_i] - pos[min_j]
                norm = np.linalg.norm(diff)
                if norm > 1e-9:
                    # Push apart
                    step = 0.005
                    push = (step * diff) / norm
                    new_pos = pos.copy()
                    new_pos[min_i] += push
                    new_pos[min_j] -= push
                    new_pos = np.clip(new_pos, -0.499999, 0.499999)
                    
                    new_r = compute_min_radius(new_pos)
                    if new_r > best_r:
                        pos = new_pos
                        best_r = new_r
            else:
                # Find the wall constraint
                min_wall_idx = np.argmin(walls)
                idx = min_wall_idx // 3
                dim = min_wall_idx % 3
                
                new_pos = pos.copy()
                step = 0.005
                if pos[idx, dim] > 0:
                    new_pos[idx, dim] -= step
                else:
                    new_pos[idx, dim] += step
                new_pos = np.clip(new_pos, -0.499999, 0.499999)
                
                new_r = compute_min_radius(new_pos)
                if new_r > best_r:
                    pos = new_pos
                    best_r = new_r
        return pos

    best_positions = None
    best_radius = -1.0
    
    seeds = [42, 123, 456, 789, 1011, 2020, 3030, 4040]
    
    for seed in seeds:
        np.random.seed(seed)
        positions = np.random.uniform(-0.4, 0.4, size=(N, 3))
        
        # Global optimization via repulsive forces
        optimized = optimize_repulsive(positions, iterations=500, lr_start=0.1, lr_end=0.001)
        
        # Local refinement
        refined = refine_local(optimized, steps=50)
        
        r = compute_min_radius(refined)
        if r > best_radius:
            best_radius = r
            best_positions = refined
            
    if best_positions is None:
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_positions = grid[:N].copy()
        
    best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    return best_positions
# EVOLVE-BLOCK-END
