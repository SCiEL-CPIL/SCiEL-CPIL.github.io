# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition - sofa points
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    # Precompute base L-shape mask for fast evaluation
    # The base L-shape: R1: 0<=u<=1, -3<=v<=1 and R2: -3<=u<=1, 0<=v<=1
    # We'll compute this on the fly for each placement
    
    def check_inside(x, y, a_deg):
        """Check which grid points are inside the L-shaped region placed at (x,y) with rotation a_deg."""
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        u = wx * c - wy * s
        v = wx * s + wy * c
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        return in_r1 | in_r2
    
    def evaluate_path(path):
        """Evaluate the score for a given path. Returns the number of sofa points * 0.0001."""
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        mask = np.ones(n_points, dtype=bool)
        
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            ts = np.linspace(0, 1, m+1)
            
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                
                inside = check_inside(x, y, a)
                mask &= inside
                
                if not mask.any():
                    return 0.0
        
        return np.sum(mask) * 0.0001
    
    # Fast evaluation using precomputed masks for common angles
    # We'll precompute the inside masks for a set of angles and positions
    
    # Strategy: Use a smart path design based on the geometry of the problem.
    # The key insight is that we need to move the L-shaped region from (0,0,0) to (-2,0,90)
    # while keeping as much overlap as possible.
    
    # Let's try a parametric approach where we optimize the path
    # using a different optimization strategy: coordinate descent with restarts
    
    start_time = time.time()
    time_limit = 580.0
    
    def make_path(n_ctrl, x0, y0, a0, x1, y1, a1):
        """Create a path with n_ctrl control points interpolated between start and end."""
        t = np.linspace(0, 1, n_ctrl + 2)[1:-1]  # exclude endpoints
        xs = x0 + t * (x1 - x0)
        ys = y0 + t * (y1 - y0)
        as_ = a0 + t * (a1 - a0)
        return np.column_stack([xs, ys, as_])
    
    # Try several different path designs
    
    # Design 1: Linear interpolation (baseline)
    path1 = make_path(5, 0, 0, 0, -2, 0, 90)
    score1 = evaluate_path(path1)
    
    best_score = score1
    best_path = path1
    
    # Design 2: Curved path - move left first, then rotate
    t = np.linspace(0.1, 0.9, 6)
    path2 = []
    for ti in t:
        x = -2.0 * ti
        y = 0.3 * np.sin(np.pi * ti)  # small upward bulge
        a = 90.0 * ti
        path2.append([x, y, a])
    path2 = np.array(path2)
    score2 = evaluate_path(path2)
    if score2 > best_score:
        best_score = score2
        best_path = path2
    
    # Design 3: Stay near origin longer, then move
    t = np.linspace(0.1, 0.9, 8)
    path3 = []
    for ti in t:
        # Slow down the movement initially
        x = -2.0 * (ti ** 1.5)
        y = 0.2 * np.sin(np.pi * ti)
        a = 90.0 * ti
        path3.append([x, y, a])
    path3 = np.array(path3)
    score3 = evaluate_path(path3)
    if score3 > best_score:
        best_score = score3
        best_path = path3
    
    # Design 4: Use the known good seed path as starting point
    seed_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    score_seed = evaluate_path(seed_path)
    if score_seed > best_score:
        best_score = score_seed
        best_path = seed_path
    
    # Now optimize using coordinate descent on the path points
    # We'll perturb each point individually and keep improvements
    
    def optimize_path(path, max_rounds=30, step=0.1):
        """Optimize path using coordinate descent."""
        current = path.copy()
        current_score = evaluate_path(current)
        best = current.copy()
        best_score_val = current_score
        
        for round_num in range(max_rounds):
            if time.time() > time_limit - 30:
                break
            
            improved = False
            for i in range(len(current)):
                # Try perturbing point i in each direction
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        for da in [-step, 0, step]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            
                            trial = current.copy()
                            trial[i][0] += dx
                            trial[i][1] += dy
                            trial[i][2] += da
                            
                            # Ensure angle stays in reasonable range
                            if trial[i][2] < 0 or trial[i][2] > 90:
                                continue
                            
                            trial_score = evaluate_path(trial)
                            if trial_score > current_score:
                                current = trial
                                current_score = trial_score
                                improved = True
                                if current_score > best_score_val:
                                    best = current.copy()
                                    best_score_val = current_score
            if not improved:
                break
        
        return best, best_score_val
    
    # Optimize the best path found so far
    if time.time() < time_limit - 60:
        opt_path, opt_score = optimize_path(best_path, max_rounds=20, step=0.15)
        if opt_score > best_score:
            best_score = opt_score
            best_path = opt_path
    
    # Try optimization with different initial perturbations
    np.random.seed(123)
    for restart in range(3):
        if time.time() > time_limit - 60:
            break
        
        # Create a perturbed version of the best path
        perturbed = best_path + np.random.normal(0, 0.2, size=best_path.shape)
        # Keep angles in [5, 85] range
        perturbed[:, 2] = np.clip(perturbed[:, 2], 5, 85)
        
        opt_path, opt_score = optimize_path(perturbed, max_rounds=10, step=0.1)
        if opt_score > best_score:
            best_score = opt_score
            best_path = opt_path
    
    # Final validation
    if len(best_path) < 1:
        best_path = make_path(3, 0, 0, 0, -2, 0, 90)
    
    if not np.all(np.isfinite(best_path)):
        best_path = make_path(3, 0, 0, 0, -2, 0, 90)
    
    return best_path

# EVOLVE-BLOCK-END
