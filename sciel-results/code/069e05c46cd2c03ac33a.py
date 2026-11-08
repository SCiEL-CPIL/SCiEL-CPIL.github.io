"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    """
    Pack 26 circles in unit square maximizing sum of radii.
    Uses multi-start local optimization with gradient-based methods.
    """
    
    def objective(params, circles_init):
        """
        Objective function: minimize negative sum of radii (to maximize it)
        params: [x1, y1, r1, x2, y2, r2, ...]
        """
        n = len(circles_init)
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for going out of bounds
        penalty = 0.0
        for i in range(n):
            if x[i] - r[i] < 0:
                penalty += (x[i] - r[i]) ** 2 * 1000
            if x[i] + r[i] > 1:
                penalty += (x[i] + r[i] - 1) ** 2 * 1000
            if y[i] - r[i] < 0:
                penalty += (y[i] - r[i]) ** 2 * 1000
            if y[i] + r[i] > 1:
                penalty += (y[i] + r[i] - 1) ** 2 * 1000
        
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        # Negative sum of radii (we minimize, so maximize sum)
        return -np.sum(r) + penalty
    
    def is_valid(circles_arr, tol=1e-5):
        """Check if all circles are non-overlapping and within [0,1]^2."""
        n = len(circles_arr)
        for i in range(n):
            x, y, r = circles_arr[i]
            if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
                return False
            if r <= 0:
                return False
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles_arr[i]
                xj, yj, rj = circles_arr[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - tol:
                    return False
        return True
    
    def hexagonal_init(n_circles, n_rows):
        """Create hexagonal-like initial configuration."""
        circles = []
        for i in range(n_rows):
            # Alternate row lengths for hexagonal pattern
            if i % 2 == 0:
                n_in_row = int(np.ceil(n_circles / n_rows))
            else:
                n_in_row = int(np.floor(n_circles / n_rows))
            
            if i == n_rows - 1:
                # Last row: adjust to get exact count
                remaining = n_circles - len(circles)
                n_in_row = remaining
                if n_in_row <= 0:
                    break
            
            y_pos = (i + 0.5) / n_rows
            offset = 0.5 / n_in_row if i % 2 == 1 else 0.0
            
            for j in range(n_in_row):
                x_pos = (j + 0.5 + offset) / n_in_row
                r_init = 0.1
                circles.append([x_pos, y_pos, r_init])
        
        # Trim or pad to get exactly n_circles
        circles = circles[:n_circles]
        while len(circles) < n_circles:
            circles.append([0.5, 0.5, 0.1])
        
        return np.array(circles)
    
    def random_init(n_circles):
        """Create random initial configuration."""
        circles = []
        for i in range(n_circles):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.05, 0.15)
            circles.append([x, y, r])
        return np.array(circles)
    
    def optimize_single(circles_init, max_iter=200):
        """Optimize a single initial configuration."""
        n = len(circles_init)
        params = circles_init.flatten()
        
        # Bounds: x, y in [0, 1], r in [0.01, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.01, 0.5))  # r
        
        # Try multiple optimization methods
        best_result = None
        best_score = -np.inf
        
        for method in ['L-BFGS-B', 'Nelder-Mead']:
            try:
                result = minimize(
                    objective,
                    params,
                    args=(circles_init,),
                    method=method,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-8}
                )
                score = -result.fun  # Convert back to sum of radii
                if score > best_score:
                    best_score = score
                    best_result = result
            except:
                pass
        
        if best_result is None:
            return circles_init, 0.0
        
        # Convert back to circles array
        x = best_result.x[0::3]
        y = best_result.x[1::3]
        r = best_result.x[2::3]
        
        circles_opt = np.column_stack([x, y, r])
        
        # Clip to valid range
        for i in range(n):
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            r[i] = np.clip(r[i], 0.01, 0.5)
            # Ensure circle stays in bounds
            r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        
        return circles_opt, -best_result.fun
    
    # Try multiple initial configurations
    best_circles = None
    best_sum = 0.0
    
    # 1. Hexagonal-like initialization
    for n_rows in range(3, 7):
        circles_init = hexagonal_init(N, n_rows)
        circles_opt, score = optimize_single(circles_init)
        if is_valid(circles_opt) and score > best_sum:
            best_sum = score
            best_circles = circles_opt.copy()
    
    # 2. Random initializations
    np.random.seed(42)
    for attempt in range(20):
        circles_init = random_init(N)
        circles_opt, score = optimize_single(circles_init, max_iter=150)
        if is_valid(circles_opt) and score > best_sum:
            best_sum = score
            best_circles = circles_opt.copy()
    
    # 3. Grid-based initialization
    side = int(np.ceil(np.sqrt(N)))
    for offset in [0.0, 0.1, 0.2]:
        circles_init = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5 + offset * (i % 2)) / side
                y = (j + 0.5) / side
                x = np.clip(x, 0.1, 0.9)
                r = 0.5 / side * 0.8
                circles_init.append([x, y, r])
                count += 1
        if len(circles_init) == N:
            circles_init = np.array(circles_init)
            circles_opt, score = optimize_single(circles_init, max_iter=150)
            if is_valid(circles_opt) and score > best_sum:
                best_sum = score
                best_circles = circles_opt.copy()
    
    # Final local improvement pass
    if best_circles is not None:
        # Try to increase radii slightly
        for iteration in range(100):
            improved = False
            for idx in range(N):
                x, y, r = best_circles[idx]
                new_r = r + 0.002
                if new_r > 0.5:
                    continue
                
                # Check if increasing radius is valid
                valid = True
                # Boundary check
                if x - new_r < -1e-6 or x + new_r > 1 + 1e-6 or y - new_r < -1e-6 or y + new_r > 1 + 1e-6:
                    valid = False
                else:
                    # Overlap check
                    for jdx in range(N):
                        if jdx == idx:
                            continue
                        xj, yj, rj = best_circles[jdx]
                        dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                        if dist < r + rj + 0.001:
                            valid = False
                            break
                
                if valid:
                    best_circles[idx][2] = new_r
                    improved = True
            
            if not improved:
                break
    
    # Final verification
    if best_circles is None or not is_valid(best_circles):
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_circles = np.array(cells)
    
    return best_circles
# EVOLVE-BLOCK-END
