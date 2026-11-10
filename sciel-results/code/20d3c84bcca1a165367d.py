# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Optimizes the placement of 10 squares in a unit square to maximize the sum of their side lengths.
    Uses Simulated Annealing with multiple restarts and a final polishing phase.
    """
    n = 10
    start_time = time.time()
    max_time = 580.0  # Safety margin before 600s limit

    # State representation:
    # x[0:n] = cx, x[n:2n] = cy, x[2n:3n] = s
    # Total dimension 30.

    def get_bounds(x):
        # Extract centers and sides
        cx = x[:n]
        cy = x[n:2*n]
        s = x[2*n:3*n]
        return cx, cy, s

    def check_validity(cx, cy, s, tol=1e-7):
        # Check bounds
        if np.any(cx < -tol) or np.any(cx > 1 + tol):
            return False
        if np.any(cy < -tol) or np.any(cy > 1 + tol):
            return False
        if np.any(s < -tol):
            return False
        # Check square containment: cx - s/2 >= 0 and cx + s/2 <= 1
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
            
        # Check overlaps
        # For each pair i, j
        for i in range(n):
            for j in range(i+1, n):
                # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_dx = (s[i] + s[j]) / 2.0 - tol
                min_dy = (s[i] + s[j]) / 2.0 - tol
                if dx < min_dx and dy < min_dy:
                    return False
        return True

    def calculate_violation(cx, cy, s, tol=1e-7):
        """
        Returns a penalty value. 0 if valid.
        Penalty is sum of overlap depths + boundary violations.
        """
        penalty = 0.0
        # Boundary violations
        # Lower bound: cx - s/2 >= 0 => violation if cx - s/2 < 0
        v = np.maximum(0, s/2 - cx)
        penalty += np.sum(v)
        # Upper bound: cx + s/2 <= 1 => violation if cx + s/2 > 1
        v = np.maximum(0, cx + s/2 - 1)
        penalty += np.sum(v)
        v = np.maximum(0, s/2 - cy)
        penalty += np.sum(v)
        v = np.maximum(0, cy + s/2 - 1)
        penalty += np.sum(v)
        
        # Overlap violations
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_dx = (s[i] + s[j]) / 2.0
                min_dy = (s[i] + s[j]) / 2.0
                
                # Overlap in x-axis if dx < min_dx
                # Depth of overlap in x is min_dx - dx
                # Overlap in y-axis if dy < min_dy
                # Depth of overlap in y is min_dy - dy
                # If both overlap, the squares intersect.
                # We can define penalty as the area of intersection or just the sum of depths.
                # Sum of depths is easier and differentiable-ish.
                ox = max(0, min_dx - dx)
                oy = max(0, min_dy - dy)
                if ox > 0 and oy > 0:
                    # Intersection area is ox * oy
                    # Or just ox + oy? Area is better for magnitude.
                    penalty += ox * oy
        return penalty

    def objective(x):
        cx, cy, s = get_bounds(x)
        # Ensure sides are positive
        if np.any(s < 0):
            return 1e10
        violation = calculate_violation(cx, cy, s)
        score = np.sum(s)
        # We want to maximize score, minimize violation.
        # If violation > 0, we are in a "bad" region.
        # Return a combined cost.
        if violation > 0:
            return 1e6 * violation - score
        else:
            return -score

    def generate_initial_config(seed_val):
        np.random.seed(seed_val)
        random.seed(seed_val)
        
        # Strategy: Random valid packing
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        # Another template: 2x5 grid
        # cx = [0.1, 0.3, 0.5, 0.7, 0.9] repeated for 2 rows
        # s = 0.2
        if seed_val % 2 == 0:
            cols = [0.1, 0.3, 0.5, 0.7, 0.9]
            rows = [0.2, 0.8]
            cx = []
            cy = []
            s = []
            for r in rows:
                for c in cols:
                    cx.append(c)
                    cy.append(r)
                    s.append(0.2)
            # We have 10 squares.
            return np.array([cx, cy, s]).T.flatten()
            
        # Template 2: 3x3 + 1
        # 3x3 grid of side 0.3 (total 0.9).
        # Centers at 0.15, 0.5, 0.85
        # 10th square?
        # Let's just use random for odd seeds.
        cx = np.random.rand(n)
        cy = np.random.rand(n)
        s = np.random.rand(n) * 0.3 + 0.1
        return np.array([cx, cy, s]).T.flatten()

    best_score = -np.inf
    best_x = None
    
    # Run multiple restarts
    num_restarts = 10
    steps_per_restart = 50000
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        x = generate_initial_config(r)
        current_score = objective(x)
        
        # Simulated Annealing
        T = 1.0
        T_min = 1e-6
        cooling = 0.999
        
        # Initial T should be high enough to accept bad moves
        # Estimate violation magnitude
        cx, cy, s = get_bounds(x)
        init_violation = calculate_violation(cx, cy, s)
        if init_violation > 0:
            T = max(T, init_violation * 100)
            
        for step in range(steps_per_restart):
            if time.time() - start_time > max_time:
                break
                
            # Propose move
            # Randomly pick a parameter to change
            idx = np.random.randint(0, 3*n)
            delta = np.random.normal(0, 0.05)
            x_new = x.copy()
            x_new[idx] += delta
            
            # Clip sides to be positive
            if idx >= 2*n:
                x_new[idx] = max(0.01, x_new[idx])
            # Clip centers to be within [0,1] roughly
            if idx < 2*n:
                x_new[idx] = np.clip(x_new[idx], 0, 1)
                
            new_score = objective(x_new)
            
            # Acceptance probability
            delta_score = new_score - current_score
            if delta_score < 0 or np.random.rand() < np.exp(-delta_score / T):
                x = x_new
                current_score = new_score
                
                # Track best valid
                cx, cy, s = get_bounds(x)
                if check_validity(cx, cy, s):
                    val = np.sum(s)
                    if val > best_score:
                        best_score = val
                        best_x = x.copy()
                        
            T *= cooling
            
    # Final Polish
    if best_x is not None:
        # Use a simple local search to improve the best valid solution
        # We can try to increase sides slightly and adjust centers to maintain validity
        cx, cy, s = get_bounds(best_x)
        current_val = np.sum(s)
        
        # Try small perturbations
        for _ in range(1000):
            if time.time() - start_time > max_time:
                break
            idx = np.random.randint(0, 3*n)
            # Bias towards increasing sides
            if idx >= 2*n:
                delta = np.random.normal(0.005, 0.01) # Positive bias
            else:
                delta = np.random.normal(0, 0.01)
                
            x_new = best_x.copy()
            x_new[idx] += delta
            if idx >= 2*n:
                x_new[idx] = max(0.01, x_new[idx])
            if idx < 2*n:
                x_new[idx] = np.clip(x_new[idx], 0, 1)
                
            cx_n, cy_n, s_n = get_bounds(x_new)
            if check_validity(cx_n, cy_n, s_n):
                new_val = np.sum(s_n)
                if new_val > current_val:
                    best_x = x_new
                    current_val = new_val
                    cx, cy, s = get_bounds(best_x)

    if best_x is None:
        # Fallback to seed
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        return centers, sides

    cx, cy, s = get_bounds(best_x)
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    # Final safety check
    if not check_validity(cx, cy, sides):
        # If invalid, return seed
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        
    return centers, sides

# EVOLVE-BLOCK-END
