# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Two-phase search: coarse grid for fast exploration, full grid for refinement.
    Phase 1: Use a coarse evaluation grid (0.05 spacing) to enable many more
             local search iterations. Phase 2: Refine the best trajectory with
             the full-resolution grid.
    """
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s limit
    
    # Full grid for final evaluation
    xs_full = np.arange(-3.0, 1.5, 0.01)
    ys_full = np.arange(-1.5, 1.5, 0.01)
    Xf, Yf = np.meshgrid(xs_full, ys_full, indexing='ij')
    gxf, gyf = Xf.ravel(), Yf.ravel()
    n_full = len(gxf)
    
    # Coarse grid for fast search (4x fewer points in each dim => 16x fewer total)
    xs_coarse = np.arange(-3.0, 1.5, 0.05)
    ys_coarse = np.arange(-1.5, 1.5, 0.05)
    Xc, Yc = np.meshgrid(xs_coarse, ys_coarse, indexing='ij')
    gxc, gyc = Xc.ravel(), Yc.ravel()
    n_coarse = len(gxc)
    
    def eval_single(gx, gy, x, y, a_deg):
        """Returns boolean mask of grid points inside this placement."""
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = gx - x
        dy = gy - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        arm1 = (u >= -0.001) & (u <= 1.001) & (v >= -3.001) & (v <= 1.001)
        arm2 = (u >= -3.001) & (u <= 1.001) & (v >= -0.001) & (v <= 1.001)
        return arm1 | arm2
    
    def densify(trajectory):
        """Densify trajectory as the evaluator does."""
        full = np.vstack([[0.0, 0.0, 0.0], trajectory, [-2.0, 0.0, 90.0]])
        dense = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                dense.append(interp)
        return dense
    
    def evaluate_trajectory_coarse(trajectory):
        """Evaluate on coarse grid - fast."""
        dense = densify(trajectory)
        mask = np.ones(n_coarse, dtype=bool)
        for x, y, a in dense:
            in_place = eval_single(gxc, gyc, x, y, a)
            mask &= in_place
            if not mask.any():
                return 0.0
        return mask.sum() / n_coarse
    
    def evaluate_trajectory_full(trajectory):
        """Evaluate on full grid - accurate but slower."""
        dense = densify(trajectory)
        mask = np.ones(n_full, dtype=bool)
        for x, y, a in dense:
            if time.time() > deadline:
                break
            in_place = eval_single(gxf, gyf, x, y, a)
            mask &= in_place
            if not mask.any():
                return 0.0
        return mask.sum() / n_full
    
    # Initial best
    best_score = 0.0
    best_traj = None
    
    # --- PHASE 1: Coarse search with many iterations ---
    traj_len = 12  # More control points for finer shape control
    
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:  # Leave time for refinement
            break
        
        # Generate initial trajectory
        t = np.linspace(0, 1, traj_len)
        
        if restart % 4 == 0:
            # Smooth arch
            amp = 0.3 + 0.1 * (restart % 3)
            x = -0.25 * t
            y = amp * np.sin(np.pi * t)
            angle = 12.5 * t
        elif restart % 4 == 1:
            # Asymmetric
            x = -0.25 * t + 0.05 * np.sin(np.pi * t) * ((restart % 5) - 2) * 0.1
            y = 0.4 * np.sin(np.pi * t) ** 1.5
            angle = 12.5 * t
        elif restart % 4 == 2:
            # Random perturbation
            x = -0.25 * t + np.random.normal(0, 0.03, traj_len)
            y = 0.35 * np.sin(np.pi * t) + np.random.normal(0, 0.03, traj_len)
            angle = 12.5 * t + np.random.normal(0, 1.5, traj_len)
        else:
            # Multi-arc
            x = -0.25 * t
            y = 0.3 * np.sin(np.pi * t) + 0.1 * np.sin(2 * np.pi * t) * ((restart % 3) - 1) * 0.2
            angle = 12.5 * t
        
        # Ensure proper endpoints
        x[0] = 0.0
        y[0] = 0.0
        angle[0] = 0.0
        x[-1] = -2.0
        y[-1] = 0.0
        angle[-1] = 90.0
        
        traj = np.column_stack([x, y, angle])
        
        score = evaluate_trajectory_coarse(traj)
        
        if score > best_score:
            best_score = score
            best_traj = traj.copy()
        
        # Local improvement with coarse evaluation (much faster)
        local_iters = 40
        for local_iter in range(local_iters):
            if time.time() > deadline - 60:
                break
            
            improved = False
            # Shuffle point order for diversity
            point_order = np.random.permutation(traj_len)
            
            for i in point_order:
                if i == 0 or i == traj_len - 1:
                    continue
                
                for dim in range(3):
                    # Adaptive step sizes
                    for step in [0.05, 0.1, 0.15, 0.2, -0.05, -0.1, -0.15, -0.2]:
                        trial = traj.copy()
                        trial[i, dim] += step
                        
                        if dim == 2:
                            trial[i, 2] = np.clip(trial[i, 2], -5, 95)
                        
                        trial_score = evaluate_trajectory_coarse(trial)
                        if trial_score > score:
                            score = trial_score
                            traj = trial
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_traj = traj.copy()
                            break
            
            if not improved:
                break
    
    # --- PHASE 2: Refine with full grid ---
    if best_traj is not None and time.time() < deadline - 10:
        # Fine-tune the best trajectory using full grid evaluation
        refine_iters = 15
        current_score = evaluate_trajectory_full(best_traj)
        refine_traj = best_traj.copy()
        
        for riter in range(refine_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            point_order = np.random.permutation(traj_len)
            
            for i in point_order:
                if i == 0 or i == traj_len - 1:
                    continue
                
                for dim in range(3):
                    for step in [0.01, 0.02, 0.03, 0.05, -0.01, -0.02, -0.03, -0.05]:
                        trial = refine_traj.copy()
                        trial[i, dim] += step
                        
                        if dim == 2:
                            trial[i, 2] = np.clip(trial[i, 2], -5, 95)
                        
                        trial_score = evaluate_trajectory_full(trial)
                        if trial_score > current_score:
                            current_score = trial_score
                            refine_traj = trial
                            improved = True
                            break
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_traj = refine_traj
    
    # If no trajectory was found, return a minimal valid one
    if best_traj is None:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    
    return best_traj

# EVOLVE-BLOCK-END
