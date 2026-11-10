# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Direct optimization of waypoints for each rotation angle.
    Instead of constraining the path to a low-dimensional sinusoidal manifold,
    we treat x(a) and y(a) for each integer angle a in [1, 89] as free variables.
    
    We use a vectorized evaluation function that checks many candidate paths 
    simultaneously against the sofa grid to rapidly identify collision-free 
    regions. A multi-restart hill-climbing search then refines the best candidates.
    """
    
    # 1. Setup Grid for Evaluation
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    # The angles we need to check internally
    # The evaluator adds (0,0,0) and (-2,0,90).
    # We return rows for angles 1 to 89.
    angles = np.arange(1, 90)
    N_angles = len(angles)
    
    # Precompute rotation matrices for each angle
    # R(-a) is used in the problem description:
    # u = dx * cos(a) + dy * sin(a)
    # v = -dx * sin(a) + dy * cos(a)
    # where dx = wx - x, dy = wy - y
    # u = wx*cos - x*cos + wy*sin - y*sin = (wx*cos + wy*sin) - (x*cos + y*sin)
    # v = -wx*sin + x*sin + wy*cos - y*cos = (-wx*sin + wy*cos) - (-x*sin + y*cos)
    
    cos_a = np.cos(np.deg2rad(angles))
    sin_a = np.sin(np.deg2rad(angles))
    
    # Project sofa points onto the local frame for each angle
    # Shape: (N_angles, N_points)
    # proj_u[angle_idx, point_idx] = wx*cos + wy*sin
    # proj_v[angle_idx, point_idx] = -wx*sin + wy*cos
    proj_u = wx[None, :] * cos_a[:, None] + wy[None, :] * sin_a[:, None]
    proj_v = -wx[None, :] * sin_a[:, None] + wy[None, :] * cos_a[:, None]
    
    # Base corridor constraints in local frame (u, v):
    # R1: 0 <= u <= 1, -3 <= v <= 1
    # R2: -3 <= u <= 1, 0 <= v <= 1
    # Union: (0<=u<=1 AND -3<=v<=1) OR (-3<=u<=1 AND 0<=v<=1)
    
    # We need to check if for ALL points, the condition holds for the specific (x, y) of that angle.
    # Condition: (proj_u - (x*cos + y*sin) is in [0,1] AND proj_v - (-x*sin + y*cos) is in [-3,1]) 
    #            OR (proj_u - (x*cos + y*sin) is in [-3,1] AND proj_v - (-x*sin + y*cos) is in [0,1])
    
    # Let local_u = x*cos + y*sin
    # Let local_v = -x*sin + y*cos
    # u_local = proj_u - local_u
    # v_local = proj_v - local_v
    
    # in_r1 = (0 <= u_local <= 1) & (-3 <= v_local <= 1)
    # in_r2 = (-3 <= u_local <= 1) & (0 <= v_local <= 1)
    # valid = in_r1 | in_r2
    
    # We want to maximize the number of points that are valid for ALL angles.
    # This is a hard constraint. If any angle fails for a point, the point is "lost".
    # Score = (number of points never lost) * 0.0001
    
    def evaluate_paths_batch(x_path, y_path):
        """
        x_path, y_path: np.ndarray of shape (N_angles,)
        Returns: number of valid points (int)
        """
        local_u = x_path * cos_a + y_path * sin_a
        local_v = -x_path * sin_a + y_path * cos_a
        
        # u_local[angle_idx, point_idx]
        u_local = proj_u - local_u[:, None]
        v_local = proj_v - local_v[:, None]
        
        in_r1 = (u_local >= 0) & (u_local <= 1) & (v_local >= -3) & (v_local <= 1)
        in_r2 = (u_local >= -3) & (u_local <= 1) & (v_local >= 0) & (v_local <= 1)
        
        valid_per_angle = in_r1 | in_r2
        
        # A point is valid if it is valid for ALL angles
        # valid_points[point_idx] = all(valid_per_angle[:, point_idx])
        # Using logical AND reduction along axis 0
        valid_points = np.all(valid_per_angle, axis=0)
        
        return int(np.sum(valid_points))
    
    # 2. Search Strategy
    # Start with a simple linear path as a baseline
    # x(0)=0, x(90)=-2
    # x(a) = -2 * a/90
    # y(a) = 0
    
    best_score = 0
    best_x = None
    best_y = None
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: Linear interpolation
    init_x = -2.0 * angles / 90.0
    init_y = np.zeros(N_angles)
    
    # Evaluate initial guess
    score = evaluate_paths_batch(init_x, init_y)
    if score > best_score:
        best_score = score
        best_x = init_x.copy()
        best_y = init_y.copy()
    
    # Multi-restart Hill Climbing
    # We will perform several restarts. Each restart starts from a slightly perturbed version of the current best
    # or a random path, and performs local improvements.
    
    np.random.seed(42)
    
    max_restarts = 50
    steps_per_restart = 200
    
    for r in range(max_restarts):
        if time.time() > deadline - 60.0: # Leave time for final checks
            break
            
        # Initialize current path
        if r == 0:
            cur_x = init_x.copy()
            cur_y = init_y.copy()
            cur_score = score
        else:
            # Perturb the best solution found so far
            # Or try a random start occasionally?
            # Let's stick to refining the best, but with larger steps initially in each restart?
            # Actually, let's just do a fresh random walk from the best solution with decaying step size.
            cur_x = best_x + np.random.normal(0, 0.5, N_angles)
            cur_y = best_y + np.random.normal(0, 0.5, N_angles)
            cur_score = evaluate_paths_batch(cur_x, cur_y)
            
            # If this random perturbation is worse, revert to best?
            # Or just continue climbing from this worse point?
            # Standard hill climbing: start from best, take steps that improve.
            # If we start from a worse point, we might get stuck.
            # Let's restart FROM the best point, but take a big initial jump.
            cur_x = best_x.copy()
            cur_y = best_y.copy()
            cur_score = best_score
            
            # Big jump
            jump_size = 1.0
            cur_x += np.random.normal(0, jump_size, N_angles)
            cur_y += np.random.normal(0, jump_size, N_angles)
            cur_score = evaluate_paths_batch(cur_x, cur_y)
            
            # If the big jump made it much worse, maybe revert?
            # For now, just try to climb from there. If it doesn't improve, we lose this restart.
            # To be more robust, we could check if cur_score > 0.
            if cur_score == 0:
                # Try a smaller jump
                cur_x = best_x.copy()
                cur_y = best_y.copy()
                cur_score = best_score
                jump_size = 0.1
                cur_x += np.random.normal(0, jump_size, N_angles)
                cur_y += np.random.normal(0, jump_size, N_angles)
                cur_score = evaluate_paths_batch(cur_x, cur_y)
                if cur_score == 0:
                    continue # Skip this restart if even small jump fails
        
        # Local Search
        step_size = 0.5
        min_step = 0.01
        no_improve_count = 0
        
        while step_size > min_step and time.time() < deadline - 60.0:
            improved = False
            # Try a few steps at this size
            for _ in range(10):
                if time.time() > deadline - 60.0:
                    break
                    
                # Perturb a random subset of angles or all angles?
                # All angles is smoother.
                dx = np.random.normal(0, step_size, N_angles)
                dy = np.random.normal(0, step_size, N_angles)
                
                new_x = cur_x + dx
                new_y = cur_y + dy
                
                new_score = evaluate_paths_batch(new_x, new_y)
                
                if new_score > cur_score:
                    cur_x = new_x
                    cur_y = new_y
                    cur_score = new_score
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            if not improved:
                step_size *= 0.7
                if no_improve_count > 10:
                    break
            else:
                # If improved, keep step size or reduce slightly?
                # Keep it to explore the basin
                pass
        
        # Update global best
        if cur_score > best_score:
            best_score = cur_score
            best_x = cur_x.copy()
            best_y = cur_y.copy()

    # 3. Construct Final Path
    # The function must return an array of shape (n, 3).
    # Rows are (x, y, a).
    # We have optimized x and y for angles 1 to 89.
    # The evaluator will add (0,0,0) before and (-2,0,90) after.
    
    final_path = np.zeros((N_angles, 3))
    final_path[:, 0] = best_x
    final_path[:, 1] = best_y
    final_path[:, 2] = angles
    
    return final_path

# EVOLVE-BLOCK-END
