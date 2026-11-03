"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    np.random.seed(42)
    
    # Perimeter = 4 => w + h = 2
    # We'll optimize over circle positions and radii
    # Variables: for each circle i: x_i, y_i, r_i
    # Constraints: 
    #   - Circles don't overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #   - Circles inside rectangle: r_i <= x_i <= w - r_i, r_i <= y_i <= h - r_i
    #   - w + h = 2, w > 0, h > 0
    # Objective: maximize sum(r_i)
    
    # We'll use a penalty method with gradient-based optimization
    # Let's try a hexagonal-like initial configuration
    
    # Start with a rectangular grid but with some optimization
    # Try different aspect ratios
    
    best_result = None
    best_sum = -1
    
    # Try several initial configurations and aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    for ar in aspect_ratios:
        # w/h = ar, w + h = 2 => h(1 + ar) = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize with a grid pattern
        side = int(np.ceil(np.sqrt(N * ar)))
        if side < 2:
            side = 2
        # Adjust to fit N circles
        rows = int(np.ceil(N / side))
        cols = side
        # Make sure rows * cols >= N
        while rows * cols < N:
            if rows < cols:
                rows += 1
            else:
                cols += 1
        
        # Initial positions
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= N:
                    break
                x = (j + 0.5) * w / cols
                y = (i + 0.5) * h / rows
                positions.append([x, y])
        
        positions = np.array(positions[:N])
        
        # Initial radii: small
        max_r = min(w, h) / max(rows, cols) * 0.8
        radii = np.full(N, max_r * 0.5)
        
        # Combine into single vector: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = positions[i][0]
            x0[3*i+1] = positions[i][1]
            x0[3*i+2] = radii[i]
        
        # Define objective (negative for minimization)
        def objective(v):
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            
            # Penalty for boundary violations
            penalty = 0.0
            for i in range(N):
                if xs[i] < rs[i]:
                    penalty += (rs[i] - xs[i]) ** 2
                if xs[i] > w - rs[i]:
                    penalty += (xs[i] - (w - rs[i])) ** 2
                if ys[i] < rs[i]:
                    penalty += (rs[i] - ys[i]) ** 2
                if ys[i] > h - rs[i]:
                    penalty += (ys[i] - (h - rs[i])) ** 2
            
            # Penalty for overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            # Maximize sum of radii => minimize negative
            return -np.sum(rs) + 100.0 * penalty
        
        # Try to optimize
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-6})
        
        if result.success or result.fun < -1e-10:
            xs = result.x[0::3]
            ys = result.x[1::3]
            rs = result.x[2::3]
            
            # Check validity
            valid = True
            for i in range(N):
                if rs[i] <= 0:
                    valid = False
                    break
                if xs[i] < rs[i] - 1e-6 or xs[i] > w - rs[i] + 1e-6:
                    valid = False
                    break
                if ys[i] < rs[i] - 1e-6 or ys[i] > h - rs[i] + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                        if dist < rs[i] + rs[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                s = np.sum(rs)
                if s > best_sum:
                    best_sum = s
                    best_result = np.column_stack([xs, ys, rs])
    
    # Fallback to seed if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
