"""Rewritten program for circle packing in unit square; Simulated Annealing approach."""
import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses Simulated Annealing with a penalty-based energy function.
    """
    n = N
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
    
    x0 = np.array(cells, dtype=float)
    
    def energy(x, y, r, penalty_weight=100.0):
        """
        Calculate energy. Lower is better.
        Energy = -sum(r) + penalty_weight * (boundary_violations + overlap_violations)
        """
        e = -np.sum(r)
        
        # Boundary penalties
        # x < r or x > 1-r
        e += penalty_weight * np.sum(np.maximum(0, r - x)**2)
        e += penalty_weight * np.sum(np.maximum(0, x + r - 1)**2)
        # y < r or y > 1-r
        e += penalty_weight * np.sum(np.maximum(0, r - y)**2)
        e += penalty_weight * np.sum(np.maximum(0, y + r - 1)**2)
        
        # Overlap penalties
        # (x_i - x_j)^2 + (y_i - y_j)^2 < (r_i + r_j)^2
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2:
                    violation = min_dist**2 - dist_sq
                    e += penalty_weight * violation
        
        return e
    
    def valid(x, y, r, eps=1e-9):
        """Check strict validity for final result."""
        if np.any(x < r - eps) or np.any(x > 1 - r + eps):
            return False
        if np.any(y < r - eps) or np.any(y > 1 - r + eps):
            return False
        if np.any(r < eps):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2 - eps:
                    return False
        return True
    
    best_x, best_y, best_r = None, None, None
    best_score = -np.inf
    
    np.random.seed(42)
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-6
    alpha = 0.999 # Cooling rate
    n_iters = 5000
    n_restarts = 20
    
    for restart in range(n_restarts):
        # Initialize
        if restart == 0:
            x_curr = x0[:, 0].copy()
            y_curr = x0[:, 1].copy()
            r_curr = x0[:, 2].copy()
        else:
            # Random initialization
            x_curr = np.random.uniform(0.1, 0.9, n)
            y_curr = np.random.uniform(0.1, 0.9, n)
            r_curr = np.full(n, 0.05)
        
        T = T_start
        curr_e = energy(x_curr, y_curr, r_curr)
        
        best_x_curr, best_y_curr, best_r_curr = x_curr.copy(), y_curr.copy(), r_curr.copy()
        best_e_curr = curr_e
        
        for it in range(n_iters):
            # Pick a random circle
            idx = np.random.randint(n)
            
            # Perturb
            # Move position
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            # Change radius
            dr = np.random.normal(0, 0.01)
            
            old_x, old_y, old_r = x_curr[idx], y_curr[idx], r_curr[idx]
            
            new_x = np.clip(old_x + dx, 0.01, 0.99)
            new_y = np.clip(old_y + dy, 0.01, 0.99)
            new_r = np.clip(old_r + dr, 0.01, 0.4)
            
            x_curr[idx] = new_x
            y_curr[idx] = new_y
            r_curr[idx] = new_r
            
            new_e = energy(x_curr, y_curr, r_curr)
            
            # Acceptance criterion
            delta_e = new_e - curr_e
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                curr_e = new_e
                if curr_e < best_e_curr:
                    best_e_curr = curr_e
                    best_x_curr = x_curr.copy()
                    best_y_curr = y_curr.copy()
                    best_r_curr = r_curr.copy()
            else:
                # Revert
                x_curr[idx] = old_x
                y_curr[idx] = old_y
                r_curr[idx] = old_r
            
            T *= alpha
        
        # Check best configuration of this restart
        if valid(best_x_curr, best_y_curr, best_r_curr):
            score = np.sum(best_r_curr)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = best_x_curr, best_y_curr, best_r_curr
        
        # If no valid config found, try to repair the best energy config? 
        # For simplicity, we rely on the penalty function to guide towards valid states.
        # However, if penalty is high, it might not be strictly valid.
        # We can do a final polish step on the best valid candidate.
    
    # If no valid configuration was found (should not happen with grid start), return grid
    if best_x is None:
        return x0
    
    return np.column_stack([best_x, best_y, best_r])
# EVOLVE-BLOCK-END
