import numpy as np
import time

N = 50
DEADLINE = 115.0  # Safety margin before 120s limit

def _score(centers):
    """Compute the radius for a given configuration (vectorized)."""
    # Boundary constraint
    boundary = (0.5 - np.abs(centers)).min()
    
    # Pairwise distances - vectorized
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r

def _score_fast(centers, prev_min_idx=None):
    """Faster scoring that reuses previous min info when possible."""
    boundary = (0.5 - np.abs(centers)).min()
    if boundary <= 0:
        return 0.0
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = dists_sq.min()
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(boundary, min_dist / 2.0)
    return r

def _grid_init():
    """Create a grid initial configuration for N=50."""
    # Try different grid arrangements
    best_pts = None
    best_r = -1
    
    # 7x8 grid = 56, remove 6
    # 8x7 grid = 56, remove 6
    # 5x10 = 50 exactly
    # 10x5 = 50 exactly
    
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]:
        total = rows * cols
        if total < N:
            continue
        
        # Create grid
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        
        if total == N:
            pts = grid
        else:
            # Remove points from corners/edges to maximize spacing
            # Compute distance to center for each point
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            # Keep the N points closest to center
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]
        
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Also try hexagonal-like arrangement
    s = 1.0 / 7.0  # spacing
    pts = []
    for i in range(8):
        for j in range(8):
            x = j * s + (i % 2) * s / 2 - 0.5 + s/4
            y = i * s * np.sqrt(3) / 2 - 0.5 + s * np.sqrt(3) / 4
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) >= N:
        # Select N points with max min distance (greedy)
        selected = []
        remaining = pts.copy()
        for _ in range(N):
            if len(remaining) == 0:
                break
            if len(selected) == 0:
                idx = 0
            else:
                sel = np.array(selected)
                dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
                idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
        
        if len(selected) == N:
            selected = np.array(selected)
            r = _score(selected)
            if r > best_r:
                best_r = r
                best_pts = selected.copy()
    
    return best_pts

def _force_relax(centers, n_steps=200, lr_init=0.02, lr_decay=0.995):
    """Force-directed relaxation to push circles apart and keep them in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Boundary forces (vectorized)
        # Push points away from boundaries
        for k in range(2):
            # Right boundary
            mask_r = centers[:, k] > 0.4
            forces[mask_r, k] -= (centers[mask_r, k] - 0.4) * 10
            # Left boundary
            mask_l = centers[:, k] < -0.4
            forces[mask_l, k] += ((-0.4) - centers[mask_l, k]) * 10
            # Top boundary
            mask_t = centers[:, k] > 0.4
            forces[mask_t, k] -= (centers[mask_t, k] - 0.4) * 10
            # Bottom boundary
            mask_b = centers[:, k] < -0.4
            forces[mask_b, k] += ((-0.4) - centers[mask_b, k]) * 10
        
        # Pairwise repulsion (vectorized)
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion force: F = (1/d^2) * direction, only for d < threshold
        threshold = 0.15
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        # Force magnitude
        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 10)  # Cap force
        
        # Sum forces
        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)
        
        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)
        
        lr *= lr_decay
    
    return centers

def _optimize_sa(centers, time_budget=60.0):
    """Simulated Annealing to maximize radius (vectorized scoring)."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    current_centers = centers.copy()
    current_score = _score(current_centers)
    
    T = 0.003
    T_min = 1e-7
    alpha = 0.999
    
    while time.time() - start_time < time_budget:
        # Generate multiple moves at once for speed
        n_moves = 5
        idxs = np.random.randint(N, size=n_moves)
        moves = np.random.randn(n_moves, 2) * 0.008
        
        for i in range(n_moves):
            idx = idxs[i]
            new_centers = current_centers.copy()
            new_centers[idx] += moves[i]
            
            # Keep in bounds
            if np.any(np.abs(new_centers) > 0.5):
                continue
            
            new_score = _score_fast(new_centers)
            
            # Acceptance probability
            delta = new_score - current_score
            if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
                current_centers = new_centers
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Reheat occasionally
        if np.random.random() < 0.01:
            T = max(T, 0.001)
            
    return best_centers, best_score

def _local_refine(centers, time_budget=10.0):
    """Local refinement with small perturbations."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    current_centers = centers.copy()
    current_score = _score(current_centers)
    
    step_size = 0.002
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_c = current_centers.copy()
                    new_c[i, 0] += dx
                    new_c[i, 1] += dy
                    if np.any(np.abs(new_c) > 0.5):
                        continue
                    s = _score_fast(new_c)
                    if s > current_score + 1e-10:
                        current_centers = new_c
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_centers = current_centers.copy()
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_centers, best_score

def construct():
    """Main entry point."""
    start_time = time.time()
    
    np.random.seed(42)
    
    # 1. Initialization
    centers = _grid_init()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # 2. Force-directed relaxation
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 40:
        centers = _force_relax(centers, n_steps=150, lr_init=0.015, lr_decay=0.99)
        score = _score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # 3. Simulated Annealing
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        t_sa = min(remaining * 0.6, 45.0)
        centers_sa, score_sa = _optimize_sa(centers, time_budget=t_sa)
        if score_sa > best_score:
            best_score = score_sa
            best_centers = centers_sa.copy()
    
    # 4. Local Refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5:
        t_lr = min(remaining * 0.5, 15.0)
        centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
        if score_lr > best_score:
            best_score = score_lr
            best_centers = centers_lr.copy()
    
    # 5. Final validation and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    final_score = _score(best_centers)
    
    # Ensure validity
    if final_score <= 0 or not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        best_centers = _grid_init()
    
    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_score(c)}")
    print(f"Shape: {c.shape}")
