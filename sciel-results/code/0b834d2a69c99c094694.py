# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    n = 10
    
    # Initialize with a 4x4 grid, taking the first 10 squares
    # This is a valid packing with side 0.25, sum = 2.5
    # We will try to expand them via LP relaxation of ordering constraints
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Place 10 squares in a 4x4 grid
    for i in range(n):
        col = i % 4
        row = i // 4
        centers[i, 0] = (col + 0.5) * 0.25
        centers[i, 1] = (row + 0.5) * 0.25
        sides[i] = 0.25
        
    best_sum = np.sum(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    # Run alternating LP optimization
    for iteration in range(100):
        # Determine constraints based on current order
        A_ub = []
        b_ub = []
        
        # Variables: x_0..x_9, y_0..y_9, s_0..s_9 (30 vars)
        # Maximize sum(s_i) -> Minimize -sum(s_i)
        c = np.zeros(3 * n)
        for i in range(n):
            c[2 * n + i] = -1.0
        
        # Bounds
        lb = np.zeros(3 * n)
        ub = np.ones(3 * n)
        # x, y in [0, 1]
        # s in [0, 1]
        
        # Boundary constraints:
        # x_i - s_i/2 >= 0  => -x_i + 0.5*s_i <= 0
        # x_i + s_i/2 <= 1  =>  x_i - 0.5*s_i <= 1
        # y_i - s_i/2 >= 0  => -y_i + 0.5*s_i <= 0
        # y_i + s_i/2 <= 1  =>  y_i - 0.5*s_i <= 1
        for i in range(n):
            # -x_i + 0.5*s_i <= 0
            row = np.zeros(3 * n)
            row[i] = -1.0
            row[2 * n + i] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            # x_i - 0.5*s_i <= 1
            row = np.zeros(3 * n)
            row[i] = 1.0
            row[2 * n + i] = -0.5
            A_ub.append(row)
            b_ub.append(1.0)
            
            # -y_i + 0.5*s_i <= 0
            row = np.zeros(3 * n)
            row[n + i] = -1.0
            row[2 * n + i] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            # y_i - 0.5*s_i <= 1
            row = np.zeros(3 * n)
            row[n + i] = 1.0
            row[2 * n + i] = -0.5
            A_ub.append(row)
            b_ub.append(1.0)
        
        # Pairwise separation constraints
        for i in range(n):
            for j in range(i + 1, n):
                # Determine if they are separated in X or Y currently
                # If |x_i - x_j| >= (s_i + s_j)/2, they are separated in X
                # If |y_i - y_j| >= (s_i + s_j)/2, they are separated in Y
                # If both, we can pick either. If neither, we have a problem (invalid state).
                # We assume valid state.
                
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                sep_x = (sides[i] + sides[j]) / 2.0
                sep_y = (sides[i] + sides[j]) / 2.0
                
                if abs(dx) >= sep_x - 1e-9:
                    # Separated in X
                    if dx < 0:
                        # i is left of j: x_i + s_i/2 <= x_j - s_j/2
                        # x_i - x_j + 0.5*s_i + 0.5*s_j <= 0
                        row = np.zeros(3 * n)
                        row[i] = 1.0
                        row[j] = -1.0
                        row[2 * n + i] = 0.5
                        row[2 * n + j] = 0.5
                        A_ub.append(row)
                        b_ub.append(0.0)
                    else:
                        # j is left of i: x_j + s_j/2 <= x_i - s_i/2
                        # x_j - x_i + 0.5*s_j + 0.5*s_i <= 0
                        row = np.zeros(3 * n)
                        row[j] = 1.0
                        row[i] = -1.0
                        row[2 * n + j] = 0.5
                        row[2 * n + i] = 0.5
                        A_ub.append(row)
                        b_ub.append(0.0)
                elif abs(dy) >= sep_y - 1e-9:
                    # Separated in Y
                    if dy < 0:
                        # i is below j: y_i + s_i/2 <= y_j - s_j/2
                        # y_i - y_j + 0.5*s_i + 0.5*s_j <= 0
                        row = np.zeros(3 * n)
                        row[n + i] = 1.0
                        row[n + j] = -1.0
                        row[2 * n + i] = 0.5
                        row[2 * n + j] = 0.5
                        A_ub.append(row)
                        b_ub.append(0.0)
                    else:
                        # j is below i: y_j + s_j/2 <= y_i - s_i/2
                        # y_j - y_i + 0.5*s_j + 0.5*s_i <= 0
                        row = np.zeros(3 * n)
                        row[n + j] = 1.0
                        row[n + i] = -1.0
                        row[2 * n + j] = 0.5
                        row[2 * n + i] = 0.5
                        A_ub.append(row)
                        b_ub.append(0.0)
                else:
                    # Overlapping in both X and Y. This should not happen if we start valid.
                    # If it does, we arbitrarily pick X separation to break the tie.
                    # This might lead to infeasibility if the constraint is tight, but let's try.
                    if dx < 0:
                        row = np.zeros(3 * n)
                        row[i] = 1.0
                        row[j] = -1.0
                        row[2 * n + i] = 0.5
                        row[2 * n + j] = 0.5
                        A_ub.append(row)
                        b_ub.append(0.0)
                    else:
                        row = np.zeros(3 * n)
                        row[j] = 1.0
                        row[i] = -1.0
                        row[2 * n + j] = 0.5
                        row[2 * n + i] = 0.5
                        A_ub.append(row)
                        b_ub.append(0.0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=zip(lb, ub), method='highs')
        
        if res.success:
            new_centers = np.zeros((n, 2), dtype=np.float64)
            new_sides = np.zeros(n, dtype=np.float64)
            for i in range(n):
                new_centers[i, 0] = res.x[i]
                new_centers[i, 1] = res.x[n + i]
                new_sides[i] = res.x[2 * n + i]
            
            # Ensure sides are positive
            new_sides = np.maximum(new_sides, 1e-9)
            
            new_sum = np.sum(new_sides)
            if new_sum > best_sum:
                best_sum = new_sum
                best_centers = new_centers
                best_sides = new_sides
                
            # Update for next iteration
            centers = new_centers
            sides = new_sides
        else:
            # If LP fails, break
            break
            
    # Final check and return
    # Clamp to [0,1] just in case of numerical errors
    best_centers[:, 0] = np.clip(best_centers[:, 0], 0, 1)
    best_centers[:, 1] = np.clip(best_centers[:, 1], 0, 1)
    best_sides = np.clip(best_sides, 1e-9, 1.0)
    
    # Verify validity roughly
    # If invalid, fallback to 3x3 grid + 1 small square
    valid = True
    tol = 1e-7
    for i in range(n):
        if best_centers[i, 0] - best_sides[i]/2 < -tol or best_centers[i, 0] + best_sides[i]/2 > 1 + tol:
            valid = False
            break
        if best_centers[i, 1] - best_sides[i]/2 < -tol or best_centers[i, 1] + best_sides[i]/2 > 1 + tol:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(best_centers[i, 0] - best_centers[j, 0])
                dy = abs(best_centers[i, 1] - best_centers[j, 1])
                t = (best_sides[i] + best_sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Fallback: 3x3 grid of 1/3, plus one tiny square? 
        # Actually, 10 squares of side 0.25 is always valid.
        # Let's just return the 4x4 grid if validation fails.
        centers = np.zeros((n, 2), dtype=np.float64)
        sides = np.zeros(n, dtype=np.float64)
        for i in range(n):
            col = i % 4
            row = i // 4
            centers[i, 0] = (col + 0.5) * 0.25
            centers[i, 1] = (row + 0.5) * 0.25
            sides[i] = 0.25
        return centers, sides
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
