import numpy as np
import time
from scipy.spatial import Voronoi

N = 50

def _score(centers):
    """Compute the radius for a given configuration."""
    # Boundary constraint
    boundary = (0.5 - np.abs(centers)).min()
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r

def _score_fast(centers):
    """Faster scoring using squared distances where possible."""
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
    """Create a grid initial configuration for N=50, trying multiple arrangements."""
    best_pts = None
    best_r = -1
    
    # Try different grid arrangements
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7), (8, 8)]:
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
            # Compute distance to center for each point, keep closest N
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]
        
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Also try hexagonal-like arrangement
    s = 1.0 / 7.0
    pts = []
    for i in range(8):
        for j in range(8):
            x = j * s + (i % 2) * s / 2 - 0.5 + s / 4
            y = i * s * np.sqrt(3) / 2 - 0.5 + s * np.sqrt(3) / 4
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) >= N:
        # Select N points with max min distance (greedy farthest-point)
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
    
    # Random init as fallback
    if best_pts is None:
        best_pts = (np.random.rand(N, 2) - 0.5) * 0.9
        best_r = _score(best_pts)
    
    return best_pts, best_r

def _random_init():
    """Random initial configuration inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9

def _force_relax(centers, n_steps=300, lr_init=0.02, lr_decay=0.995):
    """Force-directed relaxation to push circles apart and keep them in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Boundary forces
        for k in range(2):
            mask_r = centers[:, k] > 0.4
            forces[mask_r, k] -= (centers[mask_r, k] - 0.4) * 10
            mask_l = centers[:, k] < -0.4
            forces[mask_l, k] += ((-0.4) - centers[mask_l, k]) * 10
        
        # Pairwise repulsion (vectorized)
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        threshold = 0.15
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 10)
        
        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)
        
        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)
        
        lr *= lr_decay
    
    return centers

def _optimize_sa(centers, time_budget=60.0):
    """Simulated Annealing to maximize radius."""
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
            
            if np.any(np.abs(new_centers) > 0.5):
                continue
            
            new_score = _score_fast(new_centers)
            
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
                    new_centers = current_centers.copy()
                    new_centers[i, 0] += dx
                    new_centers[i, 1] += dy
                    
                    if np.any(np.abs(new_centers) > 0.5):
                        continue
                    
                    new_score = _score_fast(new_centers)
                    if new_score > current_score:
                        current_centers = new_centers
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_centers = current_centers.copy()
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_centers, best_score

def construct():
    """Main entry point: find optimal packing of 50 circles in unit square."""
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -1
    
    # Start with best grid init
    grid_pts, grid_r = _grid_init()
    best_centers = grid_pts.copy()
    best_score = grid_r
    
    # Generate multiple initial configurations
    inits = [grid_pts]
    
    # Random inits
    for _ in range(5):
        inits.append(_random_init())
    
    # Process each init with time budget
    n_inits = len(inits)
    elapsed = time.time() - start_time
    remaining_total = DEADLINE - elapsed
    time_per_init = remaining_total / max(n_inits, 1)
    time_per_init = min(time_per_init, 15.0)
    
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
            break
        
        # Force relaxation phase
        t_force = min(time_per_init * 0.2, remaining * 0.25)
        if t_force > 0.5:
            centers_force = _force_relax(init, n_steps=min(200, int(t_force * 100)))
            score_force = _score(centers_force)
            if score_force > best_score:
                best_score = score_force
                best_centers = centers_force.copy()
        else:
            centers_force = init
        
        # Annealing phase
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        t_anneal = min(time_per_init * 0.5, remaining * 0.4)
        if t_anneal > 1:
            centers_anneal, score_anneal = _optimize_sa(centers_force, time_budget=t_anneal)
            if score_anneal > best_score:
                best_score = score_anneal
                best_centers = centers_anneal.copy()
        else:
            centers_anneal = centers_force
        
        # Local refinement phase
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5 and best_centers is not None:
            t_refine = min(time_per_init * 0.3, remaining * 0.3)
            if t_refine > 1:
                centers_refine, score_refine = _local_refine(centers_anneal, time_budget=t_refine)
                if score_refine > best_score:
                    best_score = score_refine
                    best_centers = centers_refine.copy()
    
    # Final refinement on best result
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_centers is not None:
        centers_final, score_final = _local_refine(best_centers, time_budget=min(remaining * 0.8, 30.0))
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()
    
    # Ensure validity: clip to square
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
