import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in a unit square.
    Uses Simulated Annealing with a penalty-based cost function.
    """
    n = 10
    start_time = time.time()
    max_time = 580.0  # Safety margin before 600s limit

    def calculate_penalty(cx, cy, s, tol=1e-7):
        """
        Calculate the penalty for violations.
        Returns (overlap_penalty, bound_penalty, total_penalty).
        """
        # Boundary violations
        # x_min = cx - s/2, x_max = cx + s/2
        # Need x_min >= 0 and x_max <= 1
        # Violation x_min: max(0, -x_min) = max(0, s/2 - cx)
        # Violation x_max: max(0, x_max - 1) = max(0, cx + s/2 - 1)
        
        # Vectorized boundary check
        # cx: (n,), s: (n,)
        viol_x_min = np.maximum(0, s/2 - cx)
        viol_x_max = np.maximum(0, cx + s/2 - 1)
        viol_y_min = np.maximum(0, s/2 - cy)
        viol_y_max = np.maximum(0, cy + s/2 - 1)
        
        bound_penalty = np.sum(viol_x_min**2) + np.sum(viol_x_max**2) + \
                        np.sum(viol_y_min**2) + np.sum(viol_y_max**2)
        
        # Overlap penalties
        # For each pair i, j
        # dx = abs(cx[i] - cx[j])
        # dy = abs(cy[i] - cy[j])
        # req_dx = (s[i] + s[j]) / 2
        # req_dy = (s[i] + s[j]) / 2
        # overlap_x = max(0, req_dx - dx)
        # overlap_y = max(0, req_dy - dy)
        # penalty_ij = overlap_x * overlap_y (or squared)
        
        # Using broadcasting for O(n^2) calculation
        # cx: (n, 1), cx: (1, n) -> diff: (n, n)
        cx_col = cx[:, np.newaxis]
        cx_row = cx[np.newaxis, :]
        cy_col = cy[:, np.newaxis]
        cy_row = cy[np.newaxis, :]
        s_col = s[:, np.newaxis]
        s_row = s[np.newaxis, :]
        
        dx = np.abs(cx_col - cx_row)
        dy = np.abs(cy_col - cy_row)
        
        req_dx = (s_col + s_row) / 2.0
        req_dy = (s_col + s_row) / 2.0
        
        overlap_x = np.maximum(0, req_dx - dx)
        overlap_y = np.maximum(0, req_dy - dy)
        
        # We only care about i < j, so we can take the upper triangle
        # But calculating full matrix and summing half is easier
        overlap_area = overlap_x * overlap_y
        # Mask out diagonal and lower triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlap_penalty = np.sum(overlap_area[mask] ** 2)
        
        return overlap_penalty, bound_penalty, overlap_penalty + bound_penalty

    def check_validity(cx, cy, s, tol=1e-7):
        """Strict validity check for final result."""
        if np.any(s < tol):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
            
        cx_col = cx[:, np.newaxis]
        cx_row = cx[np.newaxis, :]
        cy_col = cy[:, np.newaxis]
        cy_row = cy[np.newaxis, :]
        s_col = s[:, np.newaxis]
        s_row = s[np.newaxis, :]
        
        dx = np.abs(cx_col - cx_row)
        dy = np.abs(cy_col - cy_row)
        
        req_dx = (s_col + s_row) / 2.0 - tol
        req_dy = (s_col + s_row) / 2.0 - tol
        
        # If dx < req_dx AND dy < req_dy, they overlap
        overlap_mask = (dx < req_dx) & (dy < req_dy)
        # Check if any off-diagonal element is True
        np.fill_diagonal(overlap_mask, False)
        if np.any(overlap_mask):
            return False
        return True

    def generate_initial_config():
        """Generate a valid initial configuration."""
        # Strategy 1: 2x5 grid
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
        cy = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8])
        s = np.full(10, 0.2)
        return cx, cy, s

    def sa_optimize(cx, cy, s, time_limit):
        """
        Simulated Annealing optimization.
        """
        current_cx, current_cy, current_s = cx.copy(), cy.copy(), s.copy()
        
        # Initial penalty
        _, _, current_penalty = calculate_penalty(current_cx, current_cy, current_s)
        current_objective = -np.sum(current_s) + 100.0 * current_penalty
        
        best_cx, best_cy, best_s = current_cx.copy(), current_cy.copy(), current_s.copy()
        best_objective = current_objective
        best_penalty = current_penalty
        
        T = 0.1
        cooling_rate = 0.9999
        min_T = 1e-5
        
        # Move parameters
        move_size_pos = 0.05
        move_size_side = 0.02
        
        while time.time() - start_time < time_limit:
            # Generate move
            i = np.random.randint(n)
            
            # Randomly choose to move position or size
            if np.random.rand() < 0.7:
                # Move position
                dx = np.random.normal(0, move_size_pos)
                dy = np.random.normal(0, move_size_pos)
                new_cx = current_cx.copy()
                new_cy = current_cy.copy()
                new_s = current_s.copy()
                new_cx[i] += dx
                new_cy[i] += dy
            else:
                # Change size
                ds = np.random.normal(0, move_size_side)
                # Bias towards increasing size slightly? Or just symmetric.
                # Let's try symmetric first.
                new_cx = current_cx.copy()
                new_cy = current_cy.copy()
                new_s = current_s.copy()
                new_s[i] += ds
                # Ensure side is positive
                if new_s[i] < 0.01:
                    new_s[i] = 0.01
                # Clip to reasonable max
                if new_s[i] > 1.0:
                    new_s[i] = 1.0
            
            # Calculate new penalty
            _, _, new_penalty = calculate_penalty(new_cx, new_cy, new_s)
            new_objective = -np.sum(new_s) + 100.0 * new_penalty
            
            # Acceptance criterion
            delta_obj = new_objective - current_objective
            if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / T):
                current_cx, current_cy, current_s = new_cx, new_cy, new_s
                current_penalty = new_penalty
                current_objective = new_objective
                
                # Update best
                if new_penalty < best_penalty or (abs(new_penalty - best_penalty) < 1e-6 and new_objective < best_objective):
                    best_cx, best_cy, best_s = new_cx.copy(), new_cy.copy(), new_s.copy()
                    best_objective = new_objective
                    best_penalty = new_penalty
            
            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T
            
            # Occasionally restart temperature to escape local minima
            if np.random.rand() < 0.001:
                T = 0.05

        return best_cx, best_cy, best_s, best_objective

    # Main execution
    # Start with a good initial guess
    cx, cy, s = generate_initial_config()
    
    # Run SA
    best_cx, best_cy, best_s, best_obj = sa_optimize(cx, cy, s, max_time - 10)
    
    # Final check and return
    if not check_validity(best_cx, best_cy, best_s):
        # If invalid, try to repair or fallback
        # Simple repair: shrink sides until valid
        scale = 1.0
        while not check_validity(best_cx, best_cy, best_s * scale) and scale > 0.1:
            scale *= 0.95
        best_s = best_s * scale
        
        if not check_validity(best_cx, best_cy, best_s):
             # Fallback to grid
             cx, cy, s = generate_initial_config()
             return np.stack([cx, cy], axis=1).astype(np.float64), s.astype(np.float64)
    
    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
