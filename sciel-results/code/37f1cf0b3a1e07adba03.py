"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def score_and_valid(xyz):
        """Compute sum of radii and check validity."""
        n = xyz.shape[0]
        radii = xyz[:, 2]
        score = np.sum(radii)
        # Check boundary constraints
        if np.any(radii <= 0):
            return score, False
        if np.any(xyz[:, 0] - radii < -1e-9) or np.any(xyz[:, 0] + radii > 1 + 1e-9):
            return score, False
        if np.any(xyz[:, 1] - radii < -1e-9) or np.any(xyz[:, 1] + radii > 1 + 1e-9):
            return score, False
        # Check non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = xyz[i, 0] - xyz[j, 0]
                dy = xyz[i, 1] - xyz[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return score, False
        return score, True
    
    def optimize_from_init(xyz_init, max_iters=2000, lr_init=0.01):
        """Gradient descent with soft penalty constraints."""
        xyz = xyz_init.copy()
        n = xyz.shape[0]
        
        best_xyz = xyz.copy()
        best_score, best_valid = score_and_valid(xyz)
        if best_valid:
            best_xyz = xyz.copy()
        
        lr = lr_init
        penalty_weight = 10.0
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient of objective (sum of radii)
            grad = np.zeros_like(xyz)
            grad[:, 2] = 1.0  # d(sum r)/dr_i = 1
            
            # Compute penalty gradient
            # Boundary penalty: for each circle, penalty if outside [0,1]^2
            for i in range(n):
                x, y, r = xyz[i]
                # Left boundary
                if x - r < 0:
                    viol = -(x - r)
                    grad[i, 0] -= penalty_weight * viol  # push x right
                    grad[i, 2] -= penalty_weight * viol  # push r smaller
                # Right boundary
                if x + r > 1:
                    viol = (x + r) - 1
                    grad[i, 0] += penalty_weight * viol
                    grad[i, 2] -= penalty_weight * viol
                # Bottom boundary
                if y - r < 0:
                    viol = -(y - r)
                    grad[i, 1] -= penalty_weight * viol
                    grad[i, 2] -= penalty_weight * viol
                # Top boundary
                if y + r > 1:
                    viol = (y + r) - 1
                    grad[i, 1] += penalty_weight * viol
                    grad[i, 2] -= penalty_weight * viol
            
            # Non-overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xyz[i, 0] - xyz[j, 0]
                    dy = xyz[i, 1] - xyz[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                    min_dist = xyz[i, 2] + xyz[j, 2]
                    if dist < min_dist:
                        viol = min_dist - dist
                        # Normalize direction
                        nx = dx / dist
                        ny = dy / dist
                        # Push circles apart
                        grad[i, 0] += penalty_weight * viol * nx
                        grad[i, 1] += penalty_weight * viol * ny
                        grad[j, 0] -= penalty_weight * viol * nx
                        grad[j, 1] -= penalty_weight * viol * ny
                        # Push radii smaller
                        grad[i, 2] -= penalty_weight * viol * 0.5
                        grad[j, 2] -= penalty_weight * viol * 0.5
            
            # Update
            xyz += lr * grad
            
            # Clamp radii to be positive
            xyz[:, 2] = np.maximum(xyz[:, 2], 0.01)
            
            # Clamp positions to reasonable range
            xyz[:, 0] = np.clip(xyz[:, 0], -0.5, 1.5)
            xyz[:, 1] = np.clip(xyz[:, 1], -0.5, 1.5)
            
            # Check if valid
            score, valid = score_and_valid(xyz)
            if valid and score > best_score:
                best_score = score
                best_xyz = xyz.copy()
            
            # Adaptive learning rate
            if it % 100 == 0:
                lr = lr_init * (0.95 ** (it // 100))
                penalty_weight = 10.0 * (1.1 ** (it // 200))
        
        return best_xyz, best_score, best_valid
    
    # Generate initial configurations
    def make_grid_init(n_cols, n_rows):
        """Create a grid initialization."""
        cells = []
        for i in range(n_cols):
            for j in range(n_rows):
                x = (i + 0.5) / n_cols
                y = (j + 0.5) / n_rows
                r = 0.5 / max(n_cols, n_rows) - 0.01
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    def make_random_init(seed):
        """Create a random initialization."""
        rng = np.random.RandomState(seed)
        xyz = np.zeros((N, 3))
        xyz[:, 0] = rng.uniform(0.1, 0.9, N)
        xyz[:, 1] = rng.uniform(0.1, 0.9, N)
        xyz[:, 2] = rng.uniform(0.03, 0.08, N)
        return xyz
    
    best_overall = None
    best_score_overall = 0.0
    
    # Try multiple initializations
    inits = []
    # Grid-based initializations
    inits.append(make_grid_init(5, 6))
    inits.append(make_grid_init(6, 5))
    inits.append(make_grid_init(4, 7))
    inits.append(make_grid_init(7, 4))
    
    # Random initializations
    for s in range(10):
        inits.append(make_random_init(s * 12345 + 678))
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        xyz_opt, score, valid = optimize_from_init(init)
        
        if valid and score > best_score_overall:
            best_score_overall = score
            best_overall = xyz_opt
    
    # If no valid solution found, return a trivial valid one
    if best_overall is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_overall
# EVOLVE-BLOCK-END
