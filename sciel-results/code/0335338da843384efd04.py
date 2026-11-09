import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 550  # 550s limit to be safe
    
    # Initial guess: 3x3 grid for 9 squares, 1 small in center
    # This might be too cramped. 
    # Better: 2x5 grid is too small.
    # Let's use a random valid packing as fallback.
    
    # Fallback: 3x3 grid, 9 squares of 1/3, 1 square of 0.1 in a corner? 
    # No, just use the 3x3 grid for 9 squares and put the 10th one very small 
    # in the center, but that will be invalid initially. 
    # We need a valid starting point for SA to work well, or use a penalty method 
    # that allows invalid states.
    
    # Let's use a penalty method SA.
    
    def compute_energy(centers, sides):
        # centers: (n, 2), sides: (n,)
        # Boundary penalty
        # x in [0, 1], y in [0, 1]
        # Square i: [cx - s/2, cx + s/2]
        
        # Vectorized boundary check
        # left = cx - s/2, right = cx + s/2
        # top = cy + s/2, bottom = cy - s/2
        
        left = centers[:, 0] - sides / 2
        right = centers[:, 0] + sides / 2
        bottom = centers[:, 1] - sides / 2
        top = centers[:, 1] + sides / 2
        
        # Violations
        v_left = np.maximum(0, -left)
        v_right = np.maximum(0, right - 1)
        v_bottom = np.maximum(0, -bottom)
        v_top = np.maximum(0, top - 1)
        
        boundary_pen = np.sum(v_left**2 + v_right**2 + v_bottom**2 + v_top**2)
        
        # Overlap penalty
        # For each pair (i, j), i < j
        # dx = |cx_i - cx_j|, dy = |cy_i - cy_j|
        # t = (s_i + s_j)/2
        # Overlap if dx < t and dy < t
        # Penalty: max(0, t - dx) * max(0, t - dy)
        
        # This is O(n^2). n=10, so 45 pairs. Very fast.
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0
                ox = max(0.0, t - dx)
                oy = max(0.0, t - dy)
                if ox > 0 and oy > 0:
                    overlap_pen += (ox * oy) ** 2 # Squared to be smoother? Or just ox*oy?
        
        # Objective: Maximize sum of sides
        # Energy = -sum(sides) + lambda_b * boundary_pen + lambda_o * overlap_pen
        # We want to minimize Energy.
        
        lambda_b = 100.0
        lambda_o = 100.0
        
        return -np.sum(sides) + lambda_b * boundary_pen + lambda_o * overlap_pen

    def is_valid(centers, sides):
        tol = 1e-7
        # Check bounds
        if np.any(centers[:, 0] - sides/2 < -tol) or np.any(centers[:, 0] + sides/2 > 1+tol):
            return False
        if np.any(centers[:, 1] - sides/2 < -tol) or np.any(centers[:, 1] + sides/2 > 1+tol):
            return False
        # Check overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        # Check sides > 0
        if np.any(sides <= 0):
            return False
        return True

    # Initial state
    # Let's try a few random restarts
    best_val = -np.inf
    best_centers = None
    best_sides = None
    
    # Fallback valid solution: 3x3 grid of 1/3, and one 0.1 square? 
    # Actually, just return the 3x3 grid with 9 squares and one tiny square 
    # placed in a corner if it fits? No, 3x3 fills the square.
    # We must shrink the 3x3 grid to fit the 10th.
    # Let's start SA from a random valid-ish configuration.
    
    # Generate a valid starting point:
    # 10 random squares, scaled to fit?
    # Simpler: 3x3 grid of side 0.3, and 10th square side 0.1 in center?
    # 3x3 of 0.3 leaves gaps. 
    # Let's use 3x3 grid of side 0.3.
    # Gaps are 0.1 wide.
    # Center square (1,1) is at (0.5, 0.5) with side 0.3.
    # We can put the 10th square in the gap?
    
    # Let's just run SA from a random initialization.
    n_restarts = 50
    steps_per_restart = 5000
    
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        # Random init
        # Centers in [0.1, 0.9], Sides in [0.05, 0.3]
        centers = np.random.uniform(0.1, 0.9, (n, 2))
        sides = np.random.uniform(0.05, 0.3, n)
        
        # SA
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        step_size = 0.1
        
        curr_centers = centers.copy()
        curr_sides = sides.copy()
        curr_energy = compute_energy(curr_centers, curr_sides)
        
        best_centers_local = curr_centers.copy()
        best_sides_local = curr_sides.copy()
        best_energy_local = curr_energy
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 10:
                break
            
            # Propose move
            # Perturb centers and sides
            d_centers = np.random.normal(0, step_size, (n, 2))
            d_sides = np.random.normal(0, step_size, n)
            
            new_centers = curr_centers + d_centers
            new_sides = curr_sides + d_sides
            
            # Ensure sides > 0
            new_sides = np.maximum(new_sides, 0.01)
            
            new_energy = compute_energy(new_centers, new_sides)
            
            delta_E = new_energy - curr_energy
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                curr_centers = new_centers
                curr_sides = new_sides
                curr_energy = new_energy
                
                if curr_energy < best_energy_local:
                    best_energy_local = curr_energy
                    best_centers_local = curr_centers.copy()
                    best_sides_local = curr_sides.copy()
            
            T *= alpha
            
        # Check if best local is valid
        if is_valid(best_centers_local, best_sides_local):
            val = np.sum(best_sides_local)
            if val > best_val:
                best_val = val
                best_centers = best_centers_local
                best_sides = best_sides_local
    
    # If no valid solution found, return a known valid one
    if best_centers is None:
        # Construct a valid solution
        # 3x3 grid of side 0.3, and 1 square of side 0.1 in the center?
        # 3x3 of 0.3:
        # Rows/cols at 0.15, 0.5, 0.85
        # Centers:
        cs = []
        ss = []
        for i in range(3):
            for j in range(3):
                cs.append([0.15 + 0.3*i, 0.15 + 0.3*j])
                ss.append(0.3)
        # 10th square: Center (0.5, 0.5) is occupied by square (1,1).
        # We need to move square (1,1) or make it smaller.
        # Let's shrink all 3x3 squares to 0.25.
        # Sum = 9 * 0.25 = 2.25.
        # Add 10th square of 0.25 in the center?
        # 3x3 of 0.25 leaves gaps.
        # Gaps are 0.25 wide.
        # Center gap is 0.25x0.25.
        # So we can put a 0.25 square in the center.
        # Total 10 squares of 0.25. Sum = 2.5.
        
        # Let's try to make them bigger.
        # If we use 3x3 grid of side s, and 1 square of side s in center?
        # No, the center is one of the 9 cells.
        # So we have 9 cells. We need 10 squares.
        # So one cell must have 2 squares.
        # If we split the center cell into 2 squares of side s/2?
        # Then we have 8 squares of side s, and 2 squares of side s/2.
        # Total area: 8*s^2 + 2*(s/2)^2 = 8.5 s^2 <= 1 => s <= 1/sqrt(8.5) ~ 0.34.
        # Sum = 8s + 2(s/2) = 9s.
        # If s = 0.34, Sum = 3.06.
        # This is worse than 3.33.
        
        # So the 3x3 grid of 1/3 is better.
        # How to fit the 10th?
        # Shrink the 4 squares adjacent to the center?
        
        # Let's just use the 3x3 grid of 1/3 for 9 squares, 
        # and put the 10th square of side 0.01 in a corner? 
        # No, they would overlap.
        
        # Fallback: 10 squares in a 2x5 grid. Side 0.2. Sum 2.0.
        # Or 10 squares in a 1x10 grid. Side 0.1. Sum 1.0.
        
        # Let's use the 3x3 grid of 0.3 for 9 squares, and 1 square of 0.1 in the center.
        # The center square (1,1) of the 3x3 grid is at (0.5, 0.5) with side 0.3.
        # It occupies [0.35, 0.65] x [0.35, 0.65].
        # We can't put a 0.1 square there.
        
        # Let's shrink the 4 central squares (0,1), (1,0), (1,2), (2,1) to 0.2?
        # And keep corners at 0.3?
        # This is getting messy.
        
        # Safe fallback: 10 squares of side 0.25 in a 2x5 grid?
        # No, 2x5 grid of 0.25 fits in 1x1?
        # 5 cols of 0.25 = 1.25 > 1. No.
        
        # 10 squares of side 0.2 in a 2x5 grid.
        # 5 * 0.2 = 1.0. Fits.
        # 2 * 0.2 = 0.4. Fits.
        # Sum = 2.0.
        
        # Better fallback: 3x3 grid of 1/3 (9 squares) and 1 square of 0.01 
        # placed in the center, but we must move the center square of the 3x3 grid.
        
        # Actually, the SA should find something valid.
        # If not, we return the 2x5 grid of 0.2.
        
        centers = np.zeros((10, 2))
        sides = np.full(10, 0.2)
        for i in range(10):
            row = i // 5
            col = i % 5
            centers[i, 0] = (col + 0.5) * 0.2
            centers[i, 1] = (row + 0.5) * 0.2
        return centers, sides

    return best_centers, best_sides
