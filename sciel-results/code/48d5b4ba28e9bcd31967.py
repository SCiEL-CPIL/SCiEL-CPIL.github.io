import numpy as np
import scipy.optimize as so
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Helper to compute score
    def calc_score(points):
        if len(points) != N:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        # Check bounds
        if np.any(np.abs(points) > 0.5 + 1e-9):
            return 0.0
        # Compute min distance
        d2 = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_dist = np.sqrt(np.min(d2))
        
        # Compute min dist to wall
        min_wall = np.min(0.5 - np.abs(points))
        
        r = min(min_dist / 2, min_wall)
        return r

    # Helper to run SLSQP
    def run_slsqp(x0, r0):
        x = np.concatenate([x0.flatten(), [r0]])
        
        def objective(x):
            return -x[-1]
            
        def constraints(x):
            coords = x[:N*2].reshape(N, 2)
            r = x[-1]
            
            # Distances
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            d2 = np.sum(diff**2, axis=2)
            i, j = np.triu_indices(N, 1)
            dist_c = d2[i, j] - 4*r*r
            
            # Walls
            x_c = coords[:, 0]
            y_c = coords[:, 1]
            wall_c = np.concatenate([
                0.5 - r - np.abs(x_c),
                0.5 - r + np.abs(x_c),
                0.5 - r - np.abs(y_c),
                0.5 - r + np.abs(y_c)
            ])
            
            return np.concatenate([dist_c, wall_c])
            
        cons = [{'type': 'ineq', 'fun': constraints}]
        bounds = [(-0.5, 0.5)] * (N*2) + [(0.0, 0.5)]
        
        try:
            res = so.minimize(objective, x, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                r_opt = res.x[-1]
                pts_opt = res.x[:N*2].reshape(N, 2)
                return pts_opt, r_opt
            else:
                # Even if not successful, check if it's better than current
                r_opt = res.x[-1]
                pts_opt = res.x[:N*2].reshape(N, 2)
                return pts_opt, r_opt
        except Exception:
            return None, 0.0

    # Generate Initial Guesses
    
    # 1. 7x7 Grid + 1 Random
    m = 7
    ticks = (np.arange(m) - (m-1)/2) / m * 1.0 # Spacing 1/m? No.
    # Correct grid:
    # Centers at k/m for k in -3..3?
    # Spacing 1/7.
    ticks = np.arange(-3, 4) / 7.0
    grid_7_7 = np.array([(x, y) for y in ticks for x in ticks])
    
    guess1_pts = np.vstack([grid_7_7, np.array([0.1, 0.1])])
    guess1_r = 0.07
    
    # 2. 6x9 Grid (54 pts) -> Remove 4
    m_x, m_y = 6, 9
    ticks_x = np.arange(-2.5, 3.5, 1.0) / 6.0 # Spacing 1/6? No.
    # For 6 cols, spacing 1/6?
    # Centers: -5/12, -3/12, -1/12, 1/12, 3/12, 5/12
    ticks_x = (np.arange(6) - 2.5) / 6.0
    ticks_y = (np.arange(9) - 4.0) / 9.0
    grid_6_9 = np.array([(x, y) for y in ticks_y for x in ticks_x])
    # Remove 4 random points
    indices = np.random.choice(len(grid_6_9), 4, replace=False)
    mask = np.ones(len(grid_6_9), dtype=bool)
    mask[indices] = False
    guess2_pts = grid_6_9[mask]
    guess2_r = 0.07
    
    # 3. Random Relaxed
    guess3_pts = np.random.uniform(-0.4, 0.4, (N, 2))
    # Simple relaxation
    for _ in range(50):
        # Calculate repulsive forces
        forces = np.zeros_like(guess3_pts)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = guess3_pts[i] - guess3_pts[j]
                dist = np.linalg.norm(d)
                if dist < 0.15: # Threshold
                    force = d / (dist + 1e-6) * (0.15 - dist)
                    forces[i] += force
        guess3_pts += forces * 0.1
        # Clamp to box
        guess3_pts = np.clip(guess3_pts, -0.45, 0.45)
    guess3_r = 0.05

    guesses = [
        (guess1_pts, guess1_r),
        (guess2_pts, guess2_r),
        (guess3_pts, guess3_r)
    ]
    
    # Add a few random restarts of the 7x7 grid with different 50th points
    for k in range(5):
        pt50 = np.random.uniform(-0.4, 0.4, 2)
        g_pts = np.vstack([grid_7_7, [pt50]])
        guesses.append((g_pts, 0.07))
        
    # Run optimization for each guess
    for i, (pts, r) in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        opt_pts, opt_r = run_slsqp(pts, r)
        if opt_pts is not None:
            score = calc_score(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
                
    # Final check
    if best_points is None:
        # Fallback to seed grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
