# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    """
    N = 21
    
    def get_lp_solution(centers, W, H):
        """
        Solve LP for optimal radii given fixed centers and box size.
        Returns (radii, success_flag, objective_value).
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
        
        # Boundary constraints: r_i <= dist_to_boundary
        for i in range(n):
            # Left
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][0])
            # Right
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(W - centers[i][0])
            # Bottom
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][1])
            # Top
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(H - centers[i][1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x, True, -res.fun
            else:
                return None, False, -np.inf
        except Exception:
            return None, False, -np.inf

    def optimize_centers(centers_init, W, H, max_iters=300, temp_start=0.05, temp_end=0.001):
        """
        Simulated Annealing on centers to maximize sum of radii.
        """
        centers = centers_init.copy()
        n = len(centers)
        
        # Initial evaluation
        radii, success, val = get_lp_solution(centers, W, H)
        if not success:
            # Fallback if LP fails (should not happen)
            radii = np.zeros(n)
            val = 0.0
            
        best_val = val
        best_centers = centers.copy()
        best_radii = radii.copy()
        
        T = temp_start
        cooling_rate = (temp_end / temp_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            # Generate neighbor
            # Step size proportional to temperature
            step = T * 0.5
            new_centers = centers + np.random.normal(0, step, size=centers.shape)
            # Clamp to box
            new_centers[:, 0] = np.clip(new_centers[:, 0], 0, W)
            new_centers[:, 1] = np.clip(new_centers[:, 1], 0, H)
            
            new_radii, new_success, new_val = get_lp_solution(new_centers, W, H)
            
            if new_success:
                # Accept if better, or with probability exp((new-old)/T)
                if new_val >= val or np.random.rand() < np.exp((new_val - val) / max(T, 1e-10)):
                    centers = new_centers
                    radii = new_radii
                    val = new_val
                    if val > best_val:
                        best_val = val
                        best_centers = centers.copy()
                        best_radii = radii.copy()
            
            T *= cooling_rate
            
        return best_centers, best_radii, best_val

    # Search over aspect ratios
    # W + H = 2 => H = 2 - W
    # W ranges from 0.5 to 1.5 (symmetric)
    # Let's try a grid of W values
    W_values = np.linspace(0.5, 1.5, 21) # 21 values, step 0.05
    
    best_global_val = -np.inf
    best_global_result = None
    
    np.random.seed(42)
    
    # Initial configurations
    # 1. Jittered Grid
    # 2. Random
    
    for W in W_values:
        H = 2.0 - W
        
        # Try a few random starts
        for start_idx in range(3):
            if start_idx == 0:
                # Grid start
                # 5x5 grid
                gx, gy = 5, 5
                cx = np.linspace(0.1, 0.9, gx) * W
                cy = np.linspace(0.1, 0.9, gy) * H
                grid_pts = np.array([[x, y] for y in cy for x in cx])
                centers_init = grid_pts[:21]
                # Add small jitter
                centers_init += np.random.normal(0, 0.01, size=centers_init.shape)
            else:
                # Random start
                centers_init = np.random.rand(21, 2) * np.array([W, H])
            
            centers, radii, val = optimize_centers(centers_init, W, H, max_iters=200)
            
            if val > best_global_val:
                best_global_val = val
                result = np.zeros((21, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_global_result = result

    if best_global_result is not None:
        return best_global_result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
