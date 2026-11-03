import numpy as np
import time
from scipy.optimize import minimize

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


def _objective(x):
    """Objective function for Nelder-Mead: minimize negative radius."""
    centers = x.reshape(N, 2)
    # Add penalty if any point is outside bounds
    if np.any(np.abs(centers) > 0.5):
        return 1.0
    return -_score(centers)


def _grid_init_7x7():
    """
    Create a 7x7 grid initialization for 50 circles.
    A 7x7 grid has 49 points. We need 50, so we add one more point
    optimally placed. The 7x7 grid with spacing 1/7 gives radius 1/14 ≈ 0.07143.
    """
    # 7x7 grid centered at origin
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # This gives ticks in [-0.5+0.5/7, 0.5-0.5/7] = [-3/7, 3/7]
    # Wait, (0+0.5)/7 - 0.5 = 0.5/7 - 0.5 = (0.5 - 3.5)/7 = -3/7
    # (6+0.5)/7 - 0.5 = 6.5/7 - 0.5 = (6.5 - 3.5)/7 = 3/7
    # So the grid spans from -3/7 to 3/7 in both directions
    # The spacing is 1/7, and the distance from center to edge is 3/7
    # The radius would be min(0.5 - 3/7, 1/7/2) = min(1/14, 1/14) = 1/14
    
    # Actually let me reconsider. For a 7x7 grid:
    # Points are at x = (i - 3)/7 for i=0..6, so x ranges from -3/7 to 3/7
    # Spacing = 1/7
    # Distance to boundary = 0.5 - 3/7 = (7-6)/14 = 1/14
    # Half spacing = 1/14
    # So radius = 1/14
    
    # Create 49 grid points
    grid_pts = []
    for i in range(m):
        for j in range(m):
            x = (i + 0.5) / m - 0.5
            y = (j + 0.5) / m - 0.5
            grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    
    # We have 49 points, need 50. Add a 50th point.
    # The best place for the 50th point is likely at the center, but center is occupied.
    # Let's try adding a point and letting optimization handle it.
    # For now, start with 49 points and we'll add the 50th via optimization.
    
    # Actually, let's just create 50 points. We can start with a slightly perturbed
    # 7x7 grid plus one extra point, or use a different configuration.
    
    # Alternative: use a 7x7 grid (49 pts) and add one point near the center
    # but offset slightly. The optimizer will figure out the best placement.
    
    # Let's add a point at a random position near the center
    np.random.seed(42)
    extra_pt = np.array([np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)])
    
    all_pts = np.vstack([grid_pts, extra_pt.reshape(1, 2)])
    
    return all_pts


def _optimize_nelder_mead(centers, time_budget=60.0):
    """Optimize using Nelder-Mead method."""
    start_time = time.time()
    x0 = centers.flatten()
    
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        new_centers = res.x.reshape(N, 2)
        new_score = _score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return best_centers, best_score, elapsed


def _local_refine(centers, time_budget=20.0):
    """Local refinement with small perturbations."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break
            
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_c = best_centers.copy()
                        new_c[i, 0] += dx
                        new_c[i, 1] += dy
                        if np.any(np.abs(new_c) > 0.5):
                            continue
                        s = _score(new_c)
                        if s > best_score + 1e-12:
                            best_centers = new_c
                            best_score = s
                            improved = True
            if time.time() - start_time > time_budget:
                break
    
    return best_centers, best_score


def construct():
    """Main entry point."""
    start_time = time.time()
    
    np.random.seed(42)
    
    # 1. Grid initialization
    centers = _grid_init_7x7()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # 2. Nelder-Mead optimization
    elapsed_so_far = time.time() - start_time
    remaining = DEADLINE - elapsed_so_far
    if remaining > 10:
        t_nm = min(remaining * 0.5, 50.0)
        centers_nm, score_nm, elapsed_nm = _optimize_nelder_mead(centers, time_budget=t_nm)
        if score_nm > best_score:
            best_score = score_nm
            best_centers = centers_nm.copy()
    
    # 3. Local refinement
    elapsed_so_far = time.time() - start_time
    remaining = DEADLINE - elapsed_so_far
    if remaining > 5:
        t_lr = min(remaining * 0.4, 30.0)
        centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
        if score_lr > best_score:
            best_score = score_lr
            best_centers = centers_lr.copy()
    
    # 4. Second Nelder-Mead run from refined solution if time permits
    elapsed_so_far = time.time() - start_time
    remaining = DEADLINE - elapsed_so_far
    if remaining > 15:
        t_nm2 = min(remaining * 0.6, 40.0)
        centers_nm2, score_nm2, _ = _optimize_nelder_mead(best_centers, time_budget=t_nm2)
        if score_nm2 > best_score:
            best_score = score_nm2
            best_centers = centers_nm2.copy()
    
    # 5. Final local refinement
    elapsed_so_far = time.time() - start_time
    remaining = DEADLINE - elapsed_so_far
    if remaining > 3:
        t_lr2 = min(remaining * 0.5, 10.0)
        centers_lr2, score_lr2 = _local_refine(best_centers, time_budget=t_lr2)
        if score_lr2 > best_score:
            best_score = score_lr2
            best_centers = centers_lr2.copy()
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers


if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_score(c)}")
    print(f"Shape: {c.shape}")
