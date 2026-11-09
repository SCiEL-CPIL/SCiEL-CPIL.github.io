# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Optimizes the placement of 10 squares in a unit square to maximize the sum of their side lengths.
    Uses Simulated Annealing with a penalty-based cost function to escape local minima
    and find heterogeneous packings (e.g., one small, many large).
    """
    n = 10
    start_time = time.time()
    max_time = 590.0  # Safety margin before 600s limit

    def calculate_penalty(cx, cy, s, tol=1e-7):
        """
        Calculate the penalty for violations.
        Returns total penalty. Lower is better.
        """
        penalty = 0.0
        
        # Boundary violations
        # x_min = cx - s/2, x_max = cx + s/2
        # Need x_min >= 0 and x_max <= 1
        # Violation x_min: max(0, -x_min) = max(0, -(cx - s/2))
        # Violation x_max: max(0, x_max - 1) = max(0, (cx + s/2) - 1)
        
        x_min = cx - s / 2.0
        x_max = cx + s / 2.0
        y_min = cy - s / 2.0
        y_max = cy + s / 2.0
        
        # Sum of positive violations
        penalty += np.sum(np.maximum(0, -x_min))
        penalty += np.sum(np.maximum(0, x_max - 1))
        penalty += np.sum(np.maximum(0, -y_min))
        penalty += np.sum(np.maximum(0, y_max - 1))
        
        # Negative side length penalty (should not happen if initialized well, but safe)
        penalty += np.sum(np.maximum(0, -s))
        
        # Overlap penalties
        # For each pair i, j:
        # dx = cx[i] - cx[j]
        # dy = cy[i] - cy[j]
        # overlap_x = max(0, (s[i] + s[j])/2 - abs(dx))
        # overlap_y = max(0, (s[i] + s[j])/2 - abs(dy))
        # penalty += overlap_x * overlap_y (area of overlap approximation)
        # Or simpler: penalty += max(0, overlap_x) + max(0, overlap_y) if both > 0?
        # Standard SA for packing often uses:
        # If overlapping, push apart.
        # Let's use the area of overlap as penalty, scaled.
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                min_dist_x = (s[i] + s[j]) / 2.0
                min_dist_y = (s[i] + s[j]) / 2.0
                
                # If they overlap in x
                if abs(dx) < min_dist_x:
                    # If they overlap in y
                    if abs(dy) < min_dist_y:
                        # Calculate overlap area
                        ov_x = min_dist_x - abs(dx)
                        ov_y = min_dist_y - abs(dy)
                        penalty += ov_x * ov_y
        
        return penalty

    def check_validity(cx, cy, s, tol=1e-7):
        """Check if configuration is strictly valid for the task."""
        if np.any(s <= 0):
            return False
            
        x_min = cx - s / 2.0
        x_max = cx + s / 2.0
        y_min = cy - s / 2.0
        y_max = cy + s / 2.0
        
        if np.any(x_min < -tol) or np.any(x_max > 1 + tol):
            return False
        if np.any(y_min < -tol) or np.any(y_max > 1 + tol):
            return False
            
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_dx = (s[i] + s[j]) / 2.0 - tol
                min_dy = (s[i] + s[j]) / 2.0 - tol
                if dx < min_dx and dy < min_dy:
                    return False
        return True

    # Initialize with a valid grid to ensure we start in a good region
    # 3x3 grid + 1 small? Or 2x5 grid?
    # Let's start with a 3x3 grid of 1/3 size, and one small square in a corner?
    # Actually, a 2x5 grid of 0.2 is sum 2.0.
    # A 3x3 grid of 1/3 is sum 3.0.
    # Let's try to initialize with 9 squares of size 1/3 and 1 square of size 1/3?
    # 3x3 grid of 1/3 squares:
    # Centers: (1/6, 1/6), (1/2, 1/6), (5/6, 1/6), ...
    # This fits perfectly. Sum = 3.0.
    # Can we do better?
    # If we shrink 8 squares slightly, we might fit a 10th larger one?
    # Or shrink 1 square to fit others larger?
    
    # Start with 3x3 grid of size 1/3, and 10th square small (e.g. 0.1) in a gap?
    # There are no gaps in 3x3 grid of 1/3.
    # So we must shrink.
    
    # Let's initialize randomly but validly.
    # Generate a random valid packing using a simple greedy or random restart.
    
    def random_valid_packing():
        # Try to place 10 squares randomly
        # We can use a simple rejection sampling or just start with a grid and perturb.
        # Let's start with 2x5 grid of 0.2.
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        # 2 rows, 5 cols
        for i in range(n):
            row = i // 5
            col = i % 5
            cx[i] = col * 0.2 + 0.1
            cy[i] = row * 0.2 + 0.1
            s[i] = 0.2
            
        return cx, cy, s

    cx, cy, s = random_valid_packing()
    
    best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
    best_score = np.sum(s)
    best_penalty = calculate_penalty(cx, cy, s)
    
    # Simulated Annealing Parameters
    T = 1.0  # Temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    # We will run SA until time limit
    # To speed up, we can use NumPy vectorization for penalty where possible,
    # but the overlap part is O(N^2). For N=10, N^2=100, which is fast.
    
    while time.time() - start_time < max_time - 5.0:
        # Generate a new state
        cx_new = cx.copy()
        cy_new = cy.copy()
        s_new = s.copy()
        
        # Perturb
        # Randomly pick a few squares to move/resize
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(n, num_perturb, replace=False)
        
        for idx in indices:
            # Move center
            cx_new[idx] += np.random.normal(0, 0.01)
            cy_new[idx] += np.random.normal(0, 0.01)
            # Resize
            s_new[idx] += np.random.normal(0, 0.01)
            
            # Keep s positive
            if s_new[idx] < 0.01:
                s_new[idx] = 0.01
                
            # Keep centers roughly in bounds to help convergence
            # Don't clip hard, let penalty handle it, but extreme values are bad.
            if cx_new[idx] < -0.5: cx_new[idx] = -0.5
            if cx_new[idx] > 1.5: cx_new[idx] = 1.5
            if cy_new[idx] < -0.5: cy_new[idx] = -0.5
            if cy_new[idx] > 1.5: cy_new[idx] = 1.5
            if s_new[idx] > 1.5: s_new[idx] = 1.5

        penalty_new = calculate_penalty(cx_new, cy_new, s_new)
        
        # Objective: Maximize Sum(s) - Penalty
        # Let's define Cost = Penalty - Sum(s)
        # We want to minimize Cost.
        cost_old = best_penalty - best_score # This is not quite right for SA steps, we need current state cost
        # Let's track current state cost
        # Actually, standard SA accepts based on delta E.
        # E = Penalty - Sum(s) ?
        # If we accept a move with higher penalty but higher sum, is it good?
        # Yes, if sum increases more than penalty.
        
        # Let's use: Score = Sum(s) - Penalty
        # We want to maximize Score.
        # So Delta Score = Score_new - Score_old
        # Accept if Delta > 0 or exp(Delta/T) > random
        
        score_old = np.sum(s) - penalty_new # Wait, penalty_new is for new state.
        # We need score for old state.
        # Let's maintain current_score.
        
        # Reset logic:
        # Current state: cx, cy, s
        # Current Score: curr_score = sum(s) - penalty(cx, cy, s)
        
        # Let's restart the loop with proper state tracking.
        break # Break to rewrite loop properly

    # Proper SA Loop
    cx, cy, s = best_cx.copy(), best_cy.copy(), best_s.copy()
    curr_penalty = calculate_penalty(cx, cy, s)
    curr_score = np.sum(s) - curr_penalty
    
    T = 1.0
    
    # To ensure we don't waste time, estimate steps per second.
    # Each step involves calculate_penalty (O(100) ops).
    # Python overhead is high. Let's aim for ~10k-50k steps.
    
    while time.time() - start_time < max_time - 5.0:
        cx_new = cx.copy()
        cy_new = cy.copy()
        s_new = s.copy()
        
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(n, num_perturb, replace=False)
        
        for idx in indices:
            cx_new[idx] += np.random.normal(0, 0.02)
            cy_new[idx] += np.random.normal(0, 0.02)
            s_new[idx] += np.random.normal(0, 0.02)
            
            if s_new[idx] < 0.01:
                s_new[idx] = 0.01
            if s_new[idx] > 1.0:
                s_new[idx] = 1.0
                
            # Soft bounds
            if cx_new[idx] < -0.2: cx_new[idx] = -0.2
            if cx_new[idx] > 1.2: cx_new[idx] = 1.2
            if cy_new[idx] < -0.2: cy_new[idx] = -0.2
            if cy_new[idx] > 1.2: cy_new[idx] = 1.2

        penalty_new = calculate_penalty(cx_new, cy_new, s_new)
        score_new = np.sum(s_new) - penalty_new
        
        delta_score = score_new - curr_score
        
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            cx, cy, s = cx_new, cy_new, s_new
            curr_penalty = penalty_new
            curr_score = score_new
            
            # Update best
            if penalty_new < 1e-6: # Valid enough
                if np.sum(s) > best_score:
                    best_score = np.sum(s)
                    best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
            elif np.sum(s) - penalty_new > best_score - best_penalty: # Track best valid or near-valid?
                # We only want to return valid solutions.
                # But we can track best "score" to guide search.
                pass

        # Cool down
        T *= 0.999
        if T < T_min:
            T = T_min
            # Optionally restart with higher T if stuck?
            # For now, just continue.

    # Final check and return
    if not check_validity(best_cx, best_cy, best_s):
        # If best found is invalid (due to tolerance issues in penalty vs check),
        # try to find a valid one from the neighborhood or fallback.
        # The penalty function should drive it to valid.
        # If not, fallback to grid.
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
        cy = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8])
        s = np.full(10, 0.2)
        best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
        best_score = np.sum(s)

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
