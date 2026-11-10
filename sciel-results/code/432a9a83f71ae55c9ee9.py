import numpy as np
from scipy.optimize import differential_evolution, linprog

def squares_in_square10() -> tuple:
    n = 10
    
    # Bounds for variables: x, y, s for each square
    # x, y in [0, 1], s in [0.01, 0.5]
    # We will flatten the variables: [x1, y1, s1, x2, y2, s2, ...]
    lower_bounds = []
    upper_bounds = []
    for _ in range(n):
        lower_bounds.extend([0.0, 0.0, 0.01])
        upper_bounds.extend([1.0, 1.0, 0.5])
    
    def penalty_objective(vars_flat):
        # Unpack
        centers = vars_flat[0:2*n:3]
        sides = vars_flat[2:3*n:3]
        
        # Check bounds (should be handled by DE, but safe to check)
        for i in range(n):
            if centers[i] - sides[i]/2 < -1e-6 or centers[i] + sides[i]/2 > 1 + 1e-6:
                return 1e6 # Large penalty
            if sides[i] < 0.005:
                return 1e6
        
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # Calculate overlap in x and y
                # dx = distance between centers
                dx = abs(centers[i] - centers[j])
                dy = abs(centers[i+1] - centers[j+1]) # Note: centers array is x1, x2... so y is separate? 
                # Wait, my unpacking was: x1, y1, s1, x2, y2, s2
                # So x_i is vars_flat[3*i], y_i is vars_flat[3*i+1], s_i is vars_flat[3*i+2]
                
                # Let's re-unpack correctly
                xi = vars_flat[3*i]
                yi = vars_flat[3*i+1]
                si = vars_flat[3*i+2]
                
                xj = vars_flat[3*j]
                yj = vars_flat[3*j+1]
                sj = vars_flat[3*j+2]
                
                dx = abs(xi - xj)
                dy = abs(yi - yj)
                
                # Overlap if dx < (si + sj)/2 AND dy < (si + sj)/2
                overlap_x = (si + sj)/2 - dx
                overlap_y = (si + sj)/2 - dy
                
                if overlap_x > 0 and overlap_y > 0:
                    # Penalty proportional to overlap area or depth
                    overlap_penalty += (overlap_x * overlap_y)
        
        # We want to maximize sum of sides, so we minimize negative sum
        # Objective for DE: Minimize ( -sum(sides) + Penalty )
        # If penalty is large, it dominates.
        return -np.sum(sides) + 100 * overlap_penalty

    # Run Differential Evolution
    # maxiter and popsize tuned for speed and accuracy
    res = differential_evolution(
        penalty_objective, 
        bounds=list(zip(lower_bounds, upper_bounds)), 
        maxiter=100, 
        popsize=20, 
        seed=42, 
        tol=1e-6
    )
    
    best_vars = res.x
    best_score = -res.fun
    
    # Refinement using LP for the best topology
    # Extract best configuration
    x_best = best_vars[0:2*n:3]
    y_best = best_vars[1:2*n:3]
    s_best = best_vars[2:2*n:3]
    
    # Determine topology constraints
    # For each pair (i, j), determine which of the 4 non-overlap conditions is active
    # Conditions:
    # 1. i left of j: x_i + s_i/2 <= x_j - s_j/2  => x_i - x_j + s_i/2 + s_j/2 <= 0
    # 2. j left of i: x_j + s_j/2 <= x_i - s_i/2  => x_j - x_i + s_j/2 + s_i/2 <= 0  => -x_i + x_j - s_i/2 - s_j/2 <= 0
    # 3. i below j: y_i + s_i/2 <= y_j - s_j/2  => y_i - y_j + s_i/2 + s_j/2 <= 0
    # 4. j below i: y_j + s_j/2 <= y_i - s_i/2  => -y_i + y_j - s_i/2 - s_j/2 <= 0
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i+1, n):
            xi = x_best[i]
            yi = y_best[i]
            si = s_best[i]
            
            xj = x_best[j]
            yj = y_best[j]
            sj = s_best[j]
            
            # Calculate "slack" for each condition
            # Slack > 0 means condition is satisfied with margin
            # We want to pick the condition that is most satisfied (largest slack)
            # Or simply pick one that is satisfied.
            
            slack_1 = (xj - sj/2) - (xi + si/2) # i left of j
            slack_2 = (xi - si/2) - (xj + sj/2) # j left of i
            slack_3 = (yj - sj/2) - (yi + si/2) # i below j
            slack_4 = (yi - si/2) - (yj + sj/2) # j below i
            
            slacks = [slack_1, slack_2, slack_3, slack_4]
            max_slack_idx = np.argmax(slacks)
            
            if max_slack_idx == 0:
                # x_i - x_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3*n)
                row[3*i] = 1.0
                row[3*j] = -1.0
                row[3*i+2] = 0.5
                row[3*j+2] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif max_slack_idx == 1:
                # -x_i + x_j - s_i/2 - s_j/2 <= 0
                row = np.zeros(3*n)
                row[3*i] = -1.0
                row[3*j] = 1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif max_slack_idx == 2:
                # y_i - y_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3*n)
                row[3*i+1] = 1.0
                row[3*j+1] = -1.0
                row[3*i+2] = 0.5
                row[3*j+2] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            else:
                # -y_i + y_j - s_i/2 - s_j/2 <= 0
                row = np.zeros(3*n)
                row[3*i+1] = -1.0
                row[3*j+1] = 1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(row)
                b_ub.append(0.0)
                
    # Box constraints
    # x_i - s_i/2 >= 0  => -x_i + s_i/2 <= 0
    # x_i + s_i/2 <= 1  => x_i + s_i/2 <= 1
    # y_i - s_i/2 >= 0  => -y_i + s_i/2 <= 0
    # y_i + s_i/2 <= 1  => y_i + s_i/2 <= 1
    # s_i >= 0.01       => -s_i <= -0.01
    
    for i in range(n):
        # -x_i + s_i/2 <= 0
        row = np.zeros(3*n)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # x_i + s_i/2 <= 1
        row = np.zeros(3*n)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -y_i + s_i/2 <= 0
        row = np.zeros(3*n)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # y_i + s_i/2 <= 1
        row = np.zeros(3*n)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -s_i <= -0.01
        row = np.zeros(3*n)
        row[3*i+2] = -1.0
        A_ub.append(row)
        b_ub.append(-0.01)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Objective: Minimize -sum(s_i)
    c = np.zeros(3*n)
    for i in range(n):
        c[3*i+2] = -1.0
        
    bounds_lp = list(zip(lower_bounds, upper_bounds))
    
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    
    if res_lp.success:
        lp_vars = res_lp.x
        centers_lp = np.column_stack((lp_vars[0:2*n:3], lp_vars[1:2*n:3]))
        sides_lp = lp_vars[2:2*n:3]
        
        # Verify validity
        valid = True
        for i in range(n):
            if not (0 <= centers_lp[i,0] <= 1 and 0 <= centers_lp[i,1] <= 1 and sides_lp[i] > 0):
                valid = False
                break
            for j in range(i+1, n):
                dx = abs(centers_lp[i,0] - centers_lp[j,0])
                dy = abs(centers_lp[i,1] - centers_lp[j,1])
                t = (sides_lp[i] + sides_lp[j])/2
                if dx < t - 1e-7 and dy < t - 1e-7:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return centers_lp, sides_lp
            
    # Fallback to DE result if LP failed or invalid
    centers_de = np.column_stack((x_best, y_best))
    # Ensure DE result is valid (it should be, but check)
    valid_de = True
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers_de[i,0] - centers_de[j,0])
            dy = abs(centers_de[i,1] - centers_de[j,1])
            t = (s_best[i] + s_best[j])/2
            if dx < t - 1e-7 and dy < t - 1e-7:
                valid_de = False
                break
        if not valid_de:
            break
            
    if valid_de:
        return centers_de, s_best
        
    # Fallback to baseline if all else fails
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides

