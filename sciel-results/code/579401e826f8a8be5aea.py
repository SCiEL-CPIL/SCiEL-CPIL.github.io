"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """Pack 21 circles in a rectangle of perimeter <= 4, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s (or 150s) kill
    
    # We search over rectangle dimensions w, h with 2(w+h) <= 4 => w+h <= 2
    # And over circle positions (x_i, y_i) and radii r_i
    # Objective: maximize sum(r_i)
    # Constraints: 
    #   1. w + h <= 2 (perimeter constraint)
    #   2. For each circle i: x_i >= r_i, x_i + r_i <= w, y_i >= r_i, y_i + r_i <= h
    #   3. For each pair (i,j): (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    
    # We'll use a parameterization where we fix w and h, and optimize circle positions/radii
    # Or we can optimize all together. Let's try optimizing all together with bounds.
    
    # Actually, let's use a smarter approach:
    # For a given rectangle (w, h), the maximum sum of radii for N non-overlapping circles
    # is a hard problem. Let's use a force-directed / penalty method.
    
    # Alternative: use scipy.optimize.minimize with a penalty for violations.
    
    # Let's parameterize: 
    #   z = [w, h, x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
    #   Length = 2 + 3*N = 2 + 63 = 65
    
    # Objective: -sum(r_i)  (we minimize negative sum)
    # Penalty for constraint violations.
    
    def objective_and_penalty(z, w_max=2.0):
        w, h = z[0], z[1]
        # Perimeter constraint: w + h <= 2
        pen = 0.0
        if w + h > w_max:
            pen += 100 * (w + h - w_max) ** 2
        if w < 0.1 or h < 0.1:
            pen += 1000
        
        # Extract circles
        circles = z[2:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints
        for i in range(N):
            if r[i] < 0:
                pen += 100 * r[i] ** 2
            if x[i] - r[i] < 0:
                pen += 10 * (x[i] - r[i]) ** 2
            if x[i] + r[i] > w:
                pen += 10 * (x[i] + r[i] - w) ** 2
            if y[i] - r[i] < 0:
                pen += 10 * (y[i] - r[i]) ** 2
            if y[i] + r[i] > h:
                pen += 10 * (y[i] + r[i] - h) ** 2
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    # Penalty for overlap
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    if overlap > 0:
                        pen += 50 * overlap ** 2
        
        # Objective: minimize -sum(r) + penalty
        return -np.sum(r) + pen
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    w0, h0 = 1.0, 1.0  # square, perimeter = 4
    z0 = np.zeros(2 + 3*N)
    z0[0] = w0
    z0[1] = h0
    for i in range(N):
        row = i // side
        col = i % side
        z0[2 + 3*i] = (col + 0.5) * w0 / side
        z0[3 + 3*i] = (row + 0.5) * h0 / side
        z0[4 + 3*i] = 0.45 * w0 / side  # slightly less than half cell
    
    # Bounds
    bounds = [(0.1, 2.0), (0.1, 2.0)]  # w, h
    for i in range(N):
        bounds.append((0.0, 2.0))  # x_i
        bounds.append((0.0, 2.0))  # y_i
        bounds.append((0.0, 1.0))  # r_i
    
    # Try multiple restarts with different initial configurations
    best_result = None
    best_score = -np.inf
    
    # Generate several initial configurations
    configs = []
    
    # Config 1: square grid
    configs.append(z0.copy())
    
    # Config 2: rectangular grid (wider)
    z2 = z0.copy()
    z2[0] = 1.2
    z2[1] = 0.8
    side_w = 5
    side_h = 5
    for i in range(N):
        row = i // side_w
        col = i % side_w
        z2[2 + 3*i] = (col + 0.5) * 1.2 / side_w
        z2[3 + 3*i] = (row + 0.5) * 0.8 / side_h
        z2[4 + 3*i] = 0.4 * min(1.2/side_w, 0.8/side_h)
    configs.append(z2)
    
    # Config 3: different aspect ratio
    z3 = z0.copy()
    z3[0] = 0.8
    z3[1] = 1.2
    side_w = 5
    side_h = 5
    for i in range(N):
        row = i // side_w
        col = i % side_w
        z3[2 + 3*i] = (col + 0.5) * 0.8 / side_w
        z3[3 + 3*i] = (row + 0.5) * 1.2 / side_h
        z3[4 + 3*i] = 0.4 * min(0.8/side_w, 1.2/side_h)
    configs.append(z3)
    
    # Config 4: hexagonal-like packing in square
    z4 = np.zeros(2 + 3*N)
    z4[0] = 1.0
    z4[1] = 1.0
    # Place circles in a hex pattern
    r_est = 0.18
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    count = 0
    for row in range(6):
        for col in range(6):
            if count >= N:
                break
            x = (col + 0.5) * dx + (0.5 * dx if row % 2 else 0)
            y = (row + 0.5) * dy
            if x < 1.0 and y < 1.0:
                z4[2 + 3*count] = x
                z4[3 + 3*count] = y
                z4[4 + 3*count] = r_est
                count += 1
        if count >= N:
            break
    # Fill remaining
    for i in range(count, N):
        z4[2 + 3*i] = 0.5
        z4[3 + 3*i] = 0.5
        z4[4 + 3*i] = 0.1
    configs.append(z4)
    
    for cfg in configs:
        if time.time() > deadline:
            break
        
        # Optimize
        try:
            res = minimize(
                objective_and_penalty,
                cfg,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_score:
                # Extract and validate
                z_opt = res.x
                w, h = z_opt[0], z_opt[1]
                circles = z_opt[2:].reshape(N, 3)
                x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
                
                # Check validity
                valid = True
                if w + h > 2.0 + 1e-6:
                    valid = False
                for i in range(N):
                    if r[i] < 1e-10:
                        valid = False
                        break
                    if x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
                        valid = False
                        break
                    if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            if dx*dx + dy*dy < (r[i] + r[j])**2 - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        # Clamp positions to be strictly inside
                        result = np.zeros((N, 3))
                        for i in range(N):
                            result[i, 0] = np.clip(x[i], r[i], w - r[i])
                            result[i, 1] = np.clip(y[i], r[i], h - r[i])
                            result[i, 2] = r[i]
                        best_result = result
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    w, h = 1.0, 1.0
    result = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        result[i, 0] = (col + 0.5) / side
        result[i, 1] = (row + 0.5) / side
        result[i, 2] = 0.5 / side - 1e-3
    return result

# EVOLVE-BLOCK-END
