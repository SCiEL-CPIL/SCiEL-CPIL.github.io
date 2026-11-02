import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer for final validation and return
    
    # Pre-compute indices for pairwise distances for vectorization
    # We need to calculate distances between all pairs (i, j) where i < j
    i_indices, j_indices = np.triu_indices(N, k=1)
    num_pairs = len(i_indices)
    
    def get_state(params):
        """
        params: [W, x1..xN, y1..yN, r1..rN]
        Returns W, H, x, y, r
        """
        W = params[0]
        H = 2.0 - W
        x = params[1:1+N]
        y = params[1+N:1+2*N]
        r = params[1+2*N:1+3*N]
        return W, H, x, y, r

    def compute_cost(params):
        """
        Computes the cost: -sum(r) + penalties.
        Vectorized for speed.
        """
        W, H, x, y, r = get_state(params)
        
        # Invalid dimensions
        if W <= 0 or H <= 0:
            return 1e10
            
        # 1. Boundary Penalties
        # x must be in [r, W-r]
        # y must be in [r, H-r]
        # Violation is max(0, r - x) + max(0, r - (W-x)) + ...
        
        # Vectorized boundary checks
        # Left: x < r -> viol = r - x
        # Right: x > W-r -> viol = x - (W-r) = x - W + r
        # Bottom: y < r -> viol = r - y
        # Top: y > H-r -> viol = y - (H-r) = y - H + r
        
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, x - (W - r))
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, y - (H - r))
        
        boundary_pen = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # 2. Overlap Penalties
        # dist_ij >= r_i + r_j
        # Vectorized distance calculation
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dists = r[i_indices] + r[j_indices]
        
        # Overlap is max(0, min_dist - dist)
        overlaps = np.maximum(0, min_dists - dists)
        overlap_pen = np.sum(overlaps**2)
        
        # 3. Objective
        # We want to maximize sum(r), so minimize -sum(r)
        # Penalty weights: 
        # If boundary_pen or overlap_pen is non-zero, we are invalid.
        # We use a high weight to push out of invalid regions quickly.
        weight = 100.0
        total_pen = weight * (boundary_pen + overlap_pen)
        
        cost = -np.sum(r) + total_pen
        
        return cost

    def is_valid(params, tol=1e-6):
        """
        Strict validity check for final result.
        """
        W, H, x, y, r = get_state(params)
        if W <= 0 or H <= 0:
            return False
        if np.any(r < tol):
            return False
        
        # Check boundaries
        if np.any(x < r - tol) or np.any(x > W - r + tol) or np.any(y < r - tol) or np.any(y > H - r + tol):
            return False
            
        # Check overlaps
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = r[i_indices] + r[j_indices]
        
        if np.any(dists < min_dists - tol):
            return False
            
        return True

    def polish(params, time_limit=10.0):
        """
        Local optimization using L-BFGS-B to refine a good candidate.
        """
        start_p = time.time()
        
        def objective(params):
            return compute_cost(params)
        
        # Define bounds
        # W in [0.5, 1.5]
        # x, y, r in [0, 1.5] (loose bounds, penalties handle the rest)
        bounds = [(0.5, 1.5)]
        bounds += [(0, 1.5)] * N  # x
        bounds += [(0, 1.5)] * N  # y
        bounds += [(0, 1.0)] * N  # r
        
        try:
            # Use L-BFGS-B for local refinement
            res = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() - start_p > time_limit:
                return params # Return original if timeout
            
            # Check if the polished result is valid and better
            if res.fun < compute_cost(params):
                if is_valid(res.x):
                    return res.x
                else:
                    # If invalid, try to shrink radii slightly to make it valid?
                    # For now, stick to original if invalid
                    return params
            else:
                return params
        except Exception:
            return params

    def run_sa(initial_params, time_budget):
        """
        Run Simulated Annealing.
        Returns best_params found.
        """
        start_t = time.time()
        T = 0.1
        T_min = 1e-5
        cooling_rate = 0.995
        
        best_params = initial_params.copy()
        best_cost = compute_cost(best_params)
        current_params = initial_params.copy()
        current_cost = best_cost
        
        n_params = len(current_params)
        
        while time.time() - start_t < time_budget:
            # Propose move
            # Random perturbation
            # Scale perturbation with T
            step_size = 0.05 * T + 0.001
            new_params = current_params + np.random.randn(n_params) * step_size
            
            # Keep W within reasonable bounds
            new_params[0] = np.clip(new_params[0], 0.5, 1.5)
            # Keep radii positive
            new_params[1+2*N:] = np.maximum(0, new_params[1+2*N:])
            
            new_cost = compute_cost(new_params)
            
            # Acceptance criterion
            delta = new_cost - current_cost
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_params = new_params
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_params

    # Initializations
    # 1. Grid 7x3 (aspect ratio 1.4/0.6)
    # 2. Grid 5x5 (aspect ratio 1.0/1.0)
    # 3. Grid 6x4 (aspect ratio 1.2/0.8)
    
    inits = []
    
    # Init 1: 7x3
    W1 = 1.4
    H1 = 0.6
    x1 = np.zeros(N)
    y1 = np.zeros(N)
    r1 = np.full(N, 0.08)
    for i in range(N):
        col = i % 7
        row = i // 7
        x1[i] = (col + 0.5) * W1 / 7
        y1[i] = (row + 0.5) * H1 / 3
    inits.append(np.concatenate(([W1], x1, y1, r1)))
    
    # Init 2: 5x5 (first 21)
    W2 = 1.0
    H2 = 1.0
    x2 = np.zeros(N)
    y2 = np.zeros(N)
    r2 = np.full(N, 0.09)
    for i in range(N):
        col = i % 5
        row = i // 5
        x2[i] = (col + 0.5) * W2 / 5
        y2[i] = (row + 0.5) * H2 / 5
    inits.append(np.concatenate(([W2], x2, y2, r2)))
    
    # Init 3: 6x4 (first 21)
    W3 = 1.2
    H3 = 0.8
    x3 = np.zeros(N)
    y3 = np.zeros(N)
    r3 = np.full(N, 0.085)
    for i in range(N):
        col = i % 6
        row = i // 6
        x3[i] = (col + 0.5) * W3 / 6
        y3[i] = (row + 0.5) * H3 / 4
    inits.append(np.concatenate(([W3], x3, y3, r3)))
    
    # Add random inits based on best grid
    np.random.seed(42)
    for _ in range(3):
        base = inits[np.random.randint(len(inits))]
        noise = np.random.randn(len(base)) * 0.05
        new_init = base + noise
        new_init[0] = np.clip(new_init[0], 0.6, 1.4)
        new_init[1+2*N:] = np.maximum(0.01, new_init[1+2*N:])
        inits.append(new_init)
        
    best_overall_params = None
    best_overall_val = -np.inf
    
    # Time management
    # We have ~115s total.
    # Let's allocate 60s for SA (across all restarts) and 50s for polishing.
    # Or, run SA for a fixed number of iterations/time, then polish the best.
    
    num_restarts = len(inits)
    sa_time_per_restart = 10.0  # 10s per restart * 6 restarts = 60s
    
    for i in range(num_restarts):
        if time.time() > deadline - 30.0: # Leave 30s for polishing and final checks
            break
            
        current_init = inits[i]
        
        # Run SA
        sa_best = run_sa(current_init, sa_time_per_restart)
        
        # Evaluate SA best
        sa_cost = compute_cost(sa_best)
        # The "value" we care about is sum(r). 
        # Cost = -sum(r) + penalty. If penalty is 0, Cost = -Value.
        # So Value = -Cost (if valid).
        
        if is_valid(sa_best):
            val = np.sum(sa_best[1+2*N:])
            if val > best_overall_val:
                best_overall_val = val
                best_overall_params = sa_best.copy()
        else:
            # If SA didn't find a valid solution, we might still try to polish it
            # but it's risky. Let's just store it as a candidate for polishing
            # if we don't have a valid one yet.
            if best_overall_params is None:
                best_overall_params = sa_best.copy()
                best_overall_val = -np.inf # Mark as invalid for now
                
    # Polish phase
    if best_overall_params is not None:
        # If we have a valid best, polish it.
        # If we don't, polish the best candidate hoping it becomes valid.
        
        # We can try polishing multiple times with slightly different starting points if time allows
        # But usually, one good polish on the best SA result is best.
        
        polish_start = time.time()
        polish_budget = min(50.0, deadline - time.time() - 5.0) # Leave 5s for final validation
        
        if polish_budget > 5.0:
            polished_params = polish(best_overall_params, time_limit=polish_budget)
            
            # Check if polished is valid and better
            if is_valid(polished_params):
                val = np.sum(polished_params[1+2*N:])
                if val > best_overall_val:
                    best_overall_val = val
                    best_overall_params = polished_params
            
            # If the polished one is invalid but the original was valid, keep original
            # If both invalid, we might need a fallback

    # Final Fallback
    if best_overall_params is None or not is_valid(best_overall_params):
        # Construct a simple valid grid solution
        W = 1.4
        H = 0.6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.08)
        for i in range(N):
            col = i % 7
            row = i // 7
            x[i] = (col + 0.5) * W / 7
            y[i] = (row + 0.5) * H / 3
        best_overall_params = np.concatenate(([W], x, y, r))
        # Ensure validity
        if not is_valid(best_overall_params):
             # Shrink radii
             r = np.full(N, 0.05)
             best_overall_params[1+2*N:] = r
             
    # Final validation and formatting
    W, H, x, y, r = get_state(best_overall_params)
    
    # Ensure strict validity by shrinking if necessary (safety net)
    # This is a conservative step to ensure we don't return invalid results
    if not is_valid(best_overall_params, tol=1e-7):
        # Shrink all radii by 10%
        r = r * 0.9
        best_overall_params[1+2*N:] = r
        W, H, x, y, r = get_state(best_overall_params)
        
    result = np.column_stack((x, y, r))
    
    return result

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(res)
