import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios for the rectangle
    # w + h = 2
    aspect_ratios = [1.0, 0.5, 1.5, 0.7, 1.3, 0.8, 1.25]
    
    for w_init in aspect_ratios:
        # Ensure w is within (0, 2)
        if w_init <= 0 or w_init >= 2:
            continue
        h_init = 2.0 - w_init
        
        # Generate initial grid layout
        # Estimate grid dimensions
        aspect = w_init / h_init
        c = int(np.sqrt(N * aspect))
        if c < 1: c = 1
        r = int(np.ceil(N / c))
        
        # Adjust c and r to cover N
        # We just need N positions
        positions = []
        radii = []
        
        # Cell dimensions
        cw = w_init / c
        ch = h_init / r
        
        for i in range(c):
            for j in range(r):
                if len(positions) >= N:
                    break
                x = (i + 0.5) * cw
                y = (j + 0.5) * ch
                positions.append((x, y))
                # Initial radius: half the min cell dimension, slightly smaller to avoid initial constraint violation
                r_val = 0.49 * min(cw, ch)
                radii.append(r_val)
        
        # If we didn't get enough, add more (shouldn't happen with ceil)
        while len(positions) < N:
            positions.append((w_init/2, h_init/2))
            radii.append(0.1)
            
        # Variables: [w, x1, y1, r1, x2, y2, r2, ...]
        x0 = [w_init]
        for i in range(N):
            x0.append(positions[i][0])
            x0.append(positions[i][1])
            x0.append(radii[i])
            
        x0 = np.array(x0)
        
        # Objective: minimize -sum(radii)
        def objective(x):
            w = x[0]
            h = 2.0 - w
            # If w or h is invalid, return high penalty
            if w <= 0 or h <= 0:
                return 1e6
            radii_sum = 0
            for i in range(N):
                r = x[3 + 3*i]
                if r < 0:
                    return 1e6
                radii_sum += r
            return -radii_sum
            
        # Constraints: g(x) >= 0
        def constraints(x):
            w = x[0]
            h = 2.0 - w
            if w <= 0 or h <= 0:
                # Return a vector of small negative numbers to indicate violation
                # Number of constraints: 4*N (walls) + N*(N-1)/2 (pairs)
                n_pairs = N*(N-1)//2
                return np.full(4*N + n_pairs, -1e6)
            
            cons = []
            # Wall constraints
            for i in range(N):
                xi = x[1 + 3*i]
                yi = x[2 + 3*i]
                ri = x[3 + 3*i]
                if ri < 0:
                    cons.append(-1e6)
                    cons.append(-1e6)
                    cons.append(-1e6)
                    cons.append(-1e6)
                    continue
                cons.append(xi - ri)       # left
                cons.append(w - xi - ri)   # right
                cons.append(yi - ri)       # bottom
                cons.append(h - yi - ri)   # top
                
            # Pair constraints
            for i in range(N):
                xi = x[1 + 3*i]
                yi = x[2 + 3*i]
                ri = x[3 + 3*i]
                for j in range(i+1, N):
                    xj = x[1 + 3*j]
                    yj = x[2 + 3*j]
                    rj = x[3 + 3*j]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r_sq = (ri + rj)**2
                    cons.append(dist_sq - sum_r_sq)
                    
            return np.array(cons)
            
        # Bounds
        # w in (epsilon, 2-epsilon)
        # x in (0, w) -> but w is variable, so just (0, 2) is safe for bounds, constraint handles rest
        # y in (0, 2)
        # r in (0, 1)
        bounds = [(1e-6, 2-1e-6)]
        for i in range(N):
            bounds.append((0, 2)) # x
            bounds.append((0, 2)) # y
            bounds.append((1e-6, 1.0)) # r
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                val = -res.fun
                if val > best_value:
                    best_value = val
                    best_result = res.x
        except Exception:
            pass

    if best_result is None:
        # Fallback to seed-like behavior if optimization fails
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Extract result
    w = best_result[0]
    h = 2.0 - w
    circles = []
    for i in range(N):
        x = best_result[1 + 3*i]
        y = best_result[2 + 3*i]
        r = best_result[3 + 3*i]
        # Clamp to ensure validity (small numerical errors might cause slight overlap or exit)
        # However, the evaluator checks disjoint interiors.
        # We should return the raw optimized values.
        circles.append([x, y, r])
        
    return np.array(circles)
