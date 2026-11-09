import numpy as np

N = 50

def _score(centers):
    """Compute the minimum radius (score) for a configuration."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.concatenate([dx, dy]))
    
    return min(min_pair, min_wall)

def construct():
    np.random.seed(12345)
    
    best_centers = None
    best_score = 0.0
    
    # Multiple restarts with different initial configurations
    n_restarts = 8
    iters_per_restart = 3000
    
    for restart in range(n_restarts):
        if restart == 0:
            # Grid initialization
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy().astype(float)
            centers += np.random.normal(0, 0.002, centers.shape)
        elif restart == 1:
            # Random uniform
            centers = np.random.uniform(-0.48, 0.48, (N, 2))
        elif restart == 2:
            # Hexagonal lattice perturbation
            rows = int(np.ceil(N / 8))
            centers = []
            for row in range(rows):
                for col in range(8):
                    if len(centers) >= N:
                        break
                    x = (col - 3.5) * 0.12 + (0.06 if row % 2 else 0)
                    y = (row - rows/2 + 0.5) * 0.104
                    centers.append([x, y])
            centers = np.array(centers[:N])
            centers += np.random.normal(0, 0.01, centers.shape)
        elif restart == 3:
            # Concentric ring pattern
            centers = []
            # Center
            centers.append([0.0, 0.0])
            # Ring 1: 6 points
            for i in range(6):
                angle = 2 * np.pi * i / 6
                centers.append([0.15 * np.cos(angle), 0.15 * np.sin(angle)])
            # Ring 2: 12 points
            for i in range(12):
                angle = 2 * np.pi * i / 12
                centers.append([0.30 * np.cos(angle), 0.30 * np.sin(angle)])
            # Fill remaining randomly
            while len(centers) < N:
                centers.append(np.random.uniform(-0.45, 0.45, 2))
            centers = np.array(centers[:N])
            centers += np.random.normal(0, 0.008, centers.shape)
        else:
            # Perturbed best from previous restart
            if best_centers is not None:
                centers = best_centers.copy()
                centers += np.random.normal(0, 0.03, centers.shape)
            else:
                centers = np.random.uniform(-0.48, 0.48, (N, 2))
        
        centers = np.clip(centers, -0.5, 0.5)
        current_score = _score(centers)
        
        # Simulated annealing parameters
        T_start = 0.05
        T_end = 0.0001
        n_inner = iters_per_restart
        
        for it in range(n_inner):
            t = it / max(n_inner - 1, 1)
            T = T_start * (T_end / T_start) ** t
            
            # Adaptive move size
            move_scale = 0.02 * (1.0 - 0.5 * t) + 0.002
            
            # Choose a random point to move
            idx = np.random.randint(N)
            old_pos = centers[idx].copy()
            
            # Random perturbation
            new_pos = old_pos + np.random.normal(0, move_scale, 2)
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            # Compute new score efficiently
            # Only need to check distances involving idx
            diff = centers - new_pos
            dists_to_idx = np.sqrt(np.sum(diff**2, axis=1))
            dists_to_idx[idx] = np.inf
            min_pair_new = np.min(dists_to_idx) / 2.0
            
            dx = 0.5 - np.abs(new_pos[0])
            dy = 0.5 - np.abs(new_pos[1])
            min_wall_new = min(dx, dy)
            
            new_score = min(min_pair_new, min_wall_new)
            
            # Accept or reject
            if new_score >= current_score:
                centers[idx] = new_pos
                current_score = new_score
            else:
                delta = current_score - new_score
                if delta < T:
                    accept_prob = np.exp(-delta / T)
                    if np.random.random() < accept_prob:
                        centers[idx] = new_pos
                        current_score = new_score
            
            # Track best
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        
        # Local refinement after SA: hill climbing
        for refine_iter in range(500):
            improved = False
            for idx in range(N):
                old_pos = centers[idx].copy()
                # Try small moves in 8 directions
                for dx in [-0.005, 0, 0.005]:
                    for dy in [-0.005, 0, 0.005]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = old_pos + np.array([dx, dy])
                        new_pos = np.clip(new_pos, -0.5, 0.5)
                        
                        diff = centers - new_pos
                        dists_to_idx = np.sqrt(np.sum(diff**2, axis=1))
                        dists_to_idx[idx] = np.inf
                        min_pair_new = np.min(dists_to_idx) / 2.0
                        
                        dxw = 0.5 - np.abs(new_pos[0])
                        dyw = 0.5 - np.abs(new_pos[1])
                        min_wall_new = min(dxw, dyw)
                        
                        new_score = min(min_pair_new, min_wall_new)
                        
                        if new_score > current_score:
                            centers[idx] = new_pos
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
    
    # Final fine-tuning on best solution
    if best_centers is not None:
        centers = best_centers.copy()
        current_score = best_score
        
        for fine_iter in range(2000):
            improved = False
            # Try very small moves
            step_sizes = [0.001, 0.0005, 0.0002]
            for step in step_sizes:
                for idx in range(N):
                    old_pos = centers[idx].copy()
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = old_pos + np.array([dx, dy])
                            new_pos = np.clip(new_pos, -0.5, 0.5)
                            
                            diff = centers - new_pos
                            dists_to_idx = np.sqrt(np.sum(diff**2, axis=1))
                            dists_to_idx[idx] = np.inf
                            min_pair_new = np.min(dists_to_idx) / 2.0
                            
                            dxw = 0.5 - np.abs(new_pos[0])
                            dyw = 0.5 - np.abs(new_pos[1])
                            min_wall_new = min(dxw, dyw)
                            
                            new_score = min(min_pair_new, min_wall_new)
                            
                            if new_score > current_score:
                                centers[idx] = new_pos
                                current_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
    
    return best_centers
