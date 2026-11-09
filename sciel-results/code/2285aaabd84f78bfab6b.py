# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    # State: x = [s_0..s_9, cx_0..cx_9, cy_0..cy_9]
    # Total 30 parameters
    
    def compute_cost(x):
        """
        Returns cost (to be minimized).
        Cost = -sum(sides) + penalty_overlap + penalty_boundary
        """
        s = x[:n]
        cx = x[n:2*n]
        cy = x[2*n:3*n]
        
        # Ensure sides are positive for calculation stability
        s_eff = np.maximum(s, 1e-9)
        
        # 1. Boundary Penalty
        # Squares must be in [0, 1]
        # cx - s/2 >= 0  => cx >= s/2
        # cx + s/2 <= 1  => cx <= 1 - s/2
        # Same for cy
        
        # Violation amount
        viol_x_min = np.maximum(0, s_eff/2 - cx)
        viol_x_max = np.maximum(0, cx - (1 - s_eff/2))
        viol_y_min = np.maximum(0, s_eff/2 - cy)
        viol_y_max = np.maximum(0, cy - (1 - s_eff/2))
        
        pen_boundary = np.sum(viol_x_min**2) + np.sum(viol_x_max**2) + \
                       np.sum(viol_y_min**2) + np.sum(viol_y_max**2)
        
        # 2. Overlap Penalty
        # For each pair (i, j), if they overlap, add penalty.
        # Overlap condition: |cx_i - cx_j| < (s_i + s_j)/2 AND |cy_i - cy_j| < (s_i + s_j)/2
        # We want a smooth penalty that is 0 when separated/touching.
        
        # Vectorized pairwise differences
        # cx_i - cx_j for all i,j
        dx = cx[:, np.newaxis] - cx[np.newaxis, :]
        dy = cy[:, np.newaxis] - cy[np.newaxis, :]
        s_sum = s_eff[:, np.newaxis] + s_eff[np.newaxis, :]
        
        # Required separation in x and y to avoid overlap
        # If |dx| >= s_sum/2 OR |dy| >= s_sum/2, no overlap.
        # Overlap depth in x: max(0, s_sum/2 - |dx|)
        # Overlap depth in y: max(0, s_sum/2 - |dy|)
        
        # We use the minimum of the two depths as the "overlap severity"
        # Actually, if they overlap in both x and y, the area of overlap is proportional to depth_x * depth_y.
        # A simpler penalty: if depth_x > 0 and depth_y > 0, penalty = (depth_x * depth_y) or min(depth_x, depth_y).
        # Let's use min(depth_x, depth_y) to encourage separation in at least one axis.
        
        depth_x = np.maximum(0, s_sum/2 - np.abs(dx))
        depth_y = np.maximum(0, s_sum/2 - np.abs(dy))
        
        # Penalty is sum of min(depth_x, depth_y) for i != j
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Only count pairs where BOTH depths are positive (true overlap)
        # Actually, if depth_x > 0 and depth_y > 0, they overlap.
        overlap_mask = (depth_x > 0) & (depth_y > 0) & mask
        overlap_depth = np.minimum(depth_x, depth_y)
        
        pen_overlap = np.sum(overlap_depth[overlap_mask]**2)
        
        # Objective: Maximize sum(s) => Minimize -sum(s)
        # Weights: Penalty must be large enough to dominate the gain from increasing size.
        # If we increase s by 0.1, gain is 0.1.
        # If we overlap by 0.1, penalty should be > 0.1.
        # Let's use weight 100 for penalties.
        
        cost = -np.sum(s_eff) + 100.0 * (pen_boundary + pen_overlap)
        
        return cost

    def check_validity(centers, sides):
        tol = 1e-7
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if t <= 0:
                    continue
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    return False
        return True

    def get_best_solution(centers, sides):
        if check_validity(centers, sides):
            return np.sum(sides)
        return -np.inf

    # Initial Heuristics
    def init_grid():
        # 4x4 grid, 10 squares
        side = 0.25
        idx = np.arange(n)
        cols = idx % 4
        rows = idx // 4
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        return np.concatenate([sides, centers[:, 0], centers[:, 1]])

    def init_random(seed):
        rng = np.random.RandomState(seed)
        sides = rng.uniform(0.1, 0.3, n)
        cx = rng.uniform(0.1, 0.9, n)
        cy = rng.uniform(0.1, 0.9, n)
        return np.concatenate([sides, cx, cy])

    def init_mixed(seed):
        # 4 large in corners, 6 small in center
        rng = np.random.RandomState(seed)
        s_large = 0.3
        s_small = 0.15
        sides = np.array([s_large]*4 + [s_small]*6)
        cx = np.zeros(n)
        cy = np.zeros(n)
        # Corners
        corners = [
            [s_large/2, s_large/2],
            [1 - s_large/2, s_large/2],
            [s_large/2, 1 - s_large/2],
            [1 - s_large/2, 1 - s_large/2]
        ]
        for i in range(4):
            cx[i], cy[i] = corners[i]
        
        # Center 2x3 grid
        # Space available: [s_large, 1-s_large]
        # Width = 1 - 2*s_large = 0.4
        # Height = 1 - 2*s_large = 0.4
        # 2 cols of 0.15 = 0.3. Fits.
        # 3 rows of 0.15 = 0.45. Doesn't fit in 0.4.
        # So s_small must be <= 0.4/3 = 0.133.
        # Let's set s_small = 0.13
        sides[4:] = 0.13
        s_small = 0.13
        start_x = s_large + (0.4 - 2*s_small)/2
        start_y = s_large + (0.4 - 3*s_small)/2
        
        k = 0
        for row in range(3):
            for col in range(2):
                idx = 4 + k
                cx[idx] = start_x + col * s_small + s_small/2
                cy[idx] = start_y + row * s_small + s_small/2
                k += 1
                
        # Add noise
        cx += rng.normal(0, 0.01, n)
        cy += rng.normal(0, 0.01, n)
        sides += rng.normal(0, 0.01, n)
        sides = np.maximum(sides, 0.1)
        
        return np.concatenate([sides, cx, cy])

    best_value = -np.inf
    best_x = None
    
    # Run Simulated Annealing from multiple starts
    starts = [
        init_grid(),
        init_mixed(0),
        init_mixed(1),
        init_random(2),
        init_random(3)
    ]
    
    for x0 in starts:
        if time.time() > deadline - 60:
            break
            
        x = x0.copy()
        current_cost = compute_cost(x)
        best_local_cost = current_cost
        best_local_x = x.copy()
        
        T = 1.0
        T_min = 1e-5
        alpha = 0.995 # Cooling rate
        steps = 20000
        step_size = 0.05
        
        for i in range(steps):
            if time.time() > deadline - 30:
                break
                
            # Propose new state
            # Perturb sides and centers
            dx = np.random.normal(0, step_size, 3*n)
            x_new = x + dx
            
            # Keep sides positive
            x_new[:n] = np.maximum(x_new[:n], 0.01)
            
            # Keep centers in [0, 1] roughly to avoid huge boundary penalties
            # Actually, let the penalty handle it, but clip to [-0.5, 1.5] to avoid overflow
            x_new[n:2*n] = np.clip(x_new[n:2*n], -0.5, 1.5)
            x_new[2*n:3*n] = np.clip(x_new[2*n:3*n], -0.5, 1.5)
            
            new_cost = compute_cost(x_new)
            
            delta = new_cost - current_cost
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x = x_new
                current_cost = new_cost
                if current_cost < best_local_cost:
                    best_local_cost = current_cost
                    best_local_x = x.copy()
            
            T *= alpha
            if T < T_min:
                break
        
        # Evaluate best local
        s_best = best_local_x[:n]
        cx_best = best_local_x[n:2*n]
        cy_best = best_local_x[2*n:3*n]
        centers = np.stack([cx_best, cy_best], axis=1)
        
        val = get_best_solution(centers, s_best)
        if val > best_value:
            best_value = val
            best_x = best_local_x.copy()
            
    # Final Validation and Cleanup
    if best_x is None:
        # Fallback to grid
        best_x = init_grid()
        
    s_final = best_x[:n]
    cx_final = best_x[n:2*n]
    cy_final = best_x[2*n:3*n]
    centers_final = np.stack([cx_final, cy_final], axis=1)
    
    # If the best found is invalid (due to numerical issues), try to project or fallback
    if not check_validity(centers_final, s_final):
        # Try a simple repair: shrink all sides slightly and move to center if boundary violated
        # Or just use the grid which is known valid
        centers_final, s_final = init_grid()[:n*2], init_grid()[n*2:]
        # Re-extract
        s_final = init_grid()[:n]
        cx_final = init_grid()[n:2*n]
        cy_final = init_grid()[2*n:3*n]
        centers_final = np.stack([cx_final, cy_final], axis=1)
        
    return centers_final, s_final

# EVOLVE-BLOCK-END
