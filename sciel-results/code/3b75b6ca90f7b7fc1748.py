# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 630.0  # Safety margin from 600s limit
    
    def compute_score(points):
        """Compute the actual score R = (d_min / d_max)^2"""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params, t):
        """
        Negative of the smoothed score.
        params: flattened array of shape (16, 2)
        t: temperature parameter for softmin/softmax
        """
        points = params.reshape(n, d)
        # Compute pairwise squared distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        
        # Mask diagonal (distances to self are 0, which would dominate exp(-t*0)=1)
        # We only care about i < j
        # Create a mask for upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        sq_dists_masked = sq_dists[mask]
        
        if len(sq_dists_masked) == 0:
            return 0.0
            
        # Softmin: -1/t * log(sum(exp(-t * d^2)))
        # Softmax: 1/t * log(sum(exp(t * d^2)))
        # To avoid overflow/underflow, we can shift by min/max
        # For softmin: let a = -t * d^2. max_a = max(a). sum(exp(a - max_a))
        # log(sum(exp(a))) = max_a + log(sum(exp(a - max_a)))
        
        a = -t * sq_dists_masked
        max_a = np.max(a)
        log_sum_exp_neg = max_a + np.log(np.sum(np.exp(a - max_a)))
        d_min_sq = -log_sum_exp_neg / t
        
        b = t * sq_dists_masked
        max_b = np.max(b)
        log_sum_exp_pos = max_b + np.log(np.sum(np.exp(b - max_b)))
        d_max_sq = log_sum_exp_pos / t
        
        if d_max_sq <= 1e-12:
            return 0.0
            
        ratio_sq = d_min_sq / d_max_sq
        # We want to maximize ratio_sq, so minimize -ratio_sq
        return -ratio_sq

    def optimize_from_initial(points_init, max_time_per_opt=10.0):
        """
        Run the annealing optimization starting from points_init.
        Returns the best points found.
        """
        best_points = points_init.copy()
        best_score = compute_score(best_points)
        
        # Temperatures to use
        temps = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        current_params = points_init.flatten()
        
        for t in temps:
            if time.time() > deadline - 5.0:
                break
                
            # Define the function to minimize for this t
            # L-BFGS-B requires bounds? Not strictly, but points can drift.
            # We can add a penalty for large coordinates or just let it go.
            # Scale invariance means large coords are bad for numerical stability.
            # We can normalize after each step?
            # Better: constrain the sum of squares or fix two points.
            
            # Let's fix p0 = (0,0) and p1 = (1,0) to remove translation/rotation/scale ambiguity.
            # This reduces variables from 32 to 30.
            # However, the smooth objective works on all points.
            # If we fix p0, p1, we must ensure the optimizer respects it.
            # Easiest: optimize all 32 variables, then normalize at the end.
            # But scale invariance makes the landscape flat in radial direction.
            # L-BFGS-B might struggle with flat directions.
            
            # Let's stick to optimizing all 32 variables but adding a small regularization
            # to keep points near origin, e.g., penalty on sum of squares.
            # Or just normalize the input to the optimizer? No, optimizer changes it.
            
            # Alternative: Fix p0=(0,0), p1=(1,0) in the parameterization.
            # params = [p2, ..., p15] (30 vars)
            # We modify smooth_objective to reconstruct full points from these 30 vars.
            
            def objective_fixed(params):
                # Reconstruct points
                pts = np.zeros((n, d))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = params.reshape(n-2, d)
                return smooth_objective(pts.flatten(), t)
                
            # Initial params for fixed version
            p_init = points_init.copy()
            # Normalize p_init such that p0=0, p1=(1,0)
            # This is a rigid transform.
            # Translate p0 to 0
            p_init = p_init - p_init[0]
            # Rotate/Scale so p1 is at (1,0)
            # p1 is now at some vector v.
            v = p_init[1]
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-6:
                # Degenerate, pick random
                p_init[1] = [1.0, 0.0]
            else:
                # Scale
                p_init = p_init / norm_v
                # Rotate
                # v is now unit vector. We want it to be (1,0).
                # Angle of v
                angle = np.arctan2(v[1], v[0])
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                p_init = p_init @ R.T
                
            params_init = p_init[2:].flatten()
            
            try:
                res = minimize(objective_fixed, params_init, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success or res.fun < objective_fixed(params_init):
                    # Reconstruct full points
                    new_pts = np.zeros((n, d))
                    new_pts[0] = [0.0, 0.0]
                    new_pts[1] = [1.0, 0.0]
                    new_pts[2:] = res.x.reshape(n-2, d)
                    
                    score = compute_score(new_pts)
                    if score > best_score:
                        best_score = score
                        best_points = new_pts
            except Exception:
                pass

        return best_points, best_score

    # Initial guesses
    initial_guesses = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(poly)
    
    # 2. Grid
    grid_pts = []
    # 4x4 grid
    for i in range(4):
        for j in range(4):
            grid_pts.append([i, j])
    grid_pts = np.array(grid_pts)
    # Center and scale
    grid_pts = grid_pts - np.mean(grid_pts, axis=0)
    initial_guesses.append(grid_pts)
    
    # 3. Random
    np.random.seed(42)
    rand_pts = np.random.randn(n, d)
    initial_guesses.append(rand_pts)
    
    # 4. Two concentric circles?
    # 8 inner, 8 outer
    ang_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ang_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    pts_in = np.column_stack([0.5*np.cos(ang_in), 0.5*np.sin(ang_in)])
    pts_out = np.column_stack([np.cos(ang_out), np.sin(ang_out)])
    circ_pts = np.vstack([pts_in, pts_out])
    initial_guesses.append(circ_pts)

    best_overall_points = initial_guesses[0]
    best_overall_score = 0.0
    
    for i, init in enumerate(initial_guesses):
        if time.time() > deadline - 30.0:
            break
        pts, score = optimize_from_initial(init)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = pts
            
        # Also try a few random perturbations of the best found so far?
        # Let's just run the optimizer on the current best as well if time permits
        if time.time() < deadline - 10.0:
             # Run one more refinement on the best found
             pts2, score2 = optimize_from_initial(best_overall_points)
             if score2 > best_overall_score:
                 best_overall_score = score2
                 best_overall_points = pts2

    # Final check
    return best_overall_points

# EVOLVE-BLOCK-END
