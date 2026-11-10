import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    def init_hex():
        points = []
        spacing = 0.2
        while True:
            candidates = []
            k = 10
            for i in range(-k, k+1):
                for j in range(-k, k+1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3) / 2
                    if x*x + y*y <= 1.0:
                        candidates.append([x, y])
            candidates = np.array(candidates)
            if len(candidates) >= N:
                dists = np.linalg.norm(candidates, axis=1)
                indices = np.argsort(dists)[:N]
                return candidates[indices]
            spacing *= 0.95
            
    centers = init_hex()
    
    # 2. Helper to calculate score
    def calc_score(centers):
        dists_center = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(dists_center)
        
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)
    
    best_score = calc_score(centers)
    best_centers = centers.copy()
    
    # 3. Grid-based neighbor lookup for O(N) force calculations
    def build_grid(centers, cell_size):
        """Build a spatial grid for fast neighbor lookups."""
        grid = {}
        for i, (x, y) in enumerate(centers):
            gx = int(np.floor(x / cell_size))
            gy = int(np.floor(y / cell_size))
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        return grid
    
    def get_neighbors(grid, idx, cell_size):
        """Get neighbors of point idx from grid."""
        x, y = centers[idx]
        gx = int(np.floor(x / cell_size))
        gy = int(np.floor(y / cell_size))
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                if key in grid:
                    neighbors.extend(grid[key])
        return neighbors
    
    # 4. Smooth surrogate objective using log-sum-exp for soft-min
    def soft_min(values, temperature=0.01):
        """Compute soft minimum using log-sum-exp trick."""
        if len(values) == 0:
            return 0.0
        min_val = np.min(values)
        exp_vals = np.exp(-(values - min_val) / temperature)
        sum_exp = np.sum(exp_vals)
        return min_val - temperature * np.log(sum_exp)
    
    def compute_smooth_score(centers, temperature=0.01):
        """Compute smooth surrogate score."""
        dists_center = np.linalg.norm(centers, axis=1)
        boundary_terms = 1.0 - dists_center
        
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_terms = dists / 2.0
        
        # Combine all constraints with soft-min
        all_terms = np.concatenate([boundary_terms, pair_terms])
        return soft_min(all_terms, temperature)
    
    # 5. Force calculation using grid for O(N) complexity
    def compute_forces(centers, target_r, use_grid=True):
        """Compute repulsive forces using grid-based neighbor lookup."""
        forces = np.zeros_like(centers)
        n = len(centers)
        
        if use_grid:
            cell_size = max(2 * target_r, 0.05)
            grid = build_grid(centers, cell_size)
            
            for i in range(n):
                # Boundary force
                dist_center = np.linalg.norm(centers[i])
                if dist_center > 1.0 - 0.1:
                    forces[i] -= centers[i] / max(dist_center, 1e-10) * (dist_center - (1.0 - 0.1))
                
                # Pairwise forces from neighbors only
                neighbors = get_neighbors(grid, i, cell_size)
                for j in neighbors:
                    if j <= i:
                        continue
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    target_dist = 2.0 * target_r
                    if dist < target_dist and dist > 1e-10:
                        force_mag = (target_dist - dist) / dist
                        forces[i] += diff * force_mag
                        forces[j] -= diff * force_mag
        else:
            # Fallback: full pairwise
            for i in range(n):
                # Boundary force
                dist_center = np.linalg.norm(centers[i])
                if dist_center > 1.0 - 0.1:
                    forces[i] -= centers[i] / max(dist_center, 1e-10) * (dist_center - (1.0 - 0.1))
                
                for j in range(i+1, n):
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    target_dist = 2.0 * target_r
                    if dist < target_dist and dist > 1e-10:
                        force_mag = (target_dist - dist) / dist
                        forces[i] += diff * force_mag
                        forces[j] -= diff * force_mag
        
        return forces
    
    # 6. Optimization Loop with grid-based forces
    step_size = 0.01
    stuck_count = 0
    max_stuck = 100
    temperature = 0.01
    
    while time.time() < deadline:
        current_score = calc_score(centers)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            stuck_count = 0
            step_size = max(0.001, step_size * 0.95)
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                noise = np.random.normal(0, 0.03, centers.shape)
                centers = centers + noise
                step_size = 0.03
                stuck_count = 0
                max_stuck = 50
                continue
            step_size = max(0.001, step_size * 0.995)
        
        # Compute forces using grid-based lookup
        target_r = max(best_score, 0.05)
        forces = compute_forces(centers, target_r, use_grid=True)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces = forces / max_force * step_size
        else:
            forces = np.random.normal(0, step_size, centers.shape)
            
        centers = centers + forces
        
        # Keep points inside container
        dists_center = np.linalg.norm(centers, axis=1)
        for i in range(N):
            if dists_center[i] > 0.99:
                centers[i] = centers[i] / dists_center[i] * 0.99
    
    # 7. Final high-precision polish pass
    polish_deadline = deadline - 5.0
    while time.time() < polish_deadline:
        current_score = calc_score(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
        
        # Use smaller steps for precision
        target_r = max(best_score, 0.05)
        forces = compute_forces(centers, target_r, use_grid=True)
        
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces = forces / max_force * 0.001
        else:
            break
            
        centers = centers + forces
        
        # Keep inside
        dists_center = np.linalg.norm(centers, axis=1)
        for i in range(N):
            if dists_center[i] > 0.999:
                centers[i] = centers[i] / dists_center[i] * 0.999
        
        if max_force < 1e-8:
            break
    
    # Final score check
    final_score = calc_score(centers)
    if final_score > best_score:
        best_score = final_score
        best_centers = centers.copy()
    
    return best_centers
