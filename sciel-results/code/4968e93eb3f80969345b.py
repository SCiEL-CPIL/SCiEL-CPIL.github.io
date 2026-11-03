import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial guesses
    # 1. Hexagonal packing
    # 2. Random perturbed grid
    
    def hexagonal_initial():
        # Approximate hexagonal grid
        # Side length of unit square is 1.
        # Estimate radius for hexagonal packing of N circles.
        # Area of circle = pi * r^2. Total area = N * pi * r^2.
        # Packing density ~ 0.9. N * pi * r^2 ~ 0.9.
        # r ~ sqrt(0.9 / (N * pi))
        r_est = np.sqrt(0.9 / (N * np.pi))
        
        # Create a grid of points
        # Hexagonal lattice points
        # dx = 2*r, dy = r*sqrt(3)
        # Adjust to fit in [0,1]^2
        
        # Simple approach: generate points in a hexagonal pattern
        # and center them.
        pts = []
        # Estimate number of rows
        # For N=26, roughly 5x5 or 6x5
        # Let's try a 5x6 grid with hexagonal offset
        rows = 5
        cols = 6
        # Adjust to fit
        # Width = (cols - 1) * 2*r + 2*r = cols * 2*r
        # Height = (rows - 1) * r*sqrt(3) + 2*r
        
        # Let's just use a grid and optimize.
        # Grid 5x6 = 30 points, take 26.
        xs = np.linspace(0.1, 0.9, cols)
        ys = np.linspace(0.1, 0.9, rows)
        
        pts = []
        for i in range(rows):
            for j in range(cols):
                x = xs[j]
                y = ys[i]
                if i % 2 == 1:
                    x += (xs[1] - xs[0]) * 0.5
                pts.append([x, y, r_est])
        
        pts = np.array(pts[:N])
        return pts

    def random_initial():
        # Random positions, small radii
        r_est = np.sqrt(0.5 / (N * np.pi))
        xs = np.random.uniform(0.2, 0.8, N)
        ys = np.random.uniform(0.2, 0.8, N)
        rs = np.full(N, r_est)
        return np.column_stack((xs, ys, rs))

    def objective_and_constraints(x):
        # x is flat array [x0, y0, r0, x1, y1, r1, ...]
        pts = x.reshape(N, 3)
        xs = pts[:, 0]
        ys = pts[:, 1]
        rs = pts[:, 2]
        
        # Objective: maximize sum of radii -> minimize -sum(rs)
        obj = -np.sum(rs)
        
        # Constraints: g(x) >= 0
        cons = []
        
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]}) # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3] - x[i*3+2]}) # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]}) # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3+1] - x[i*3+2]}) # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6}) # r_i >= eps
            
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (x[i*3] - x[j*3])**2 + (x[i*3+1] - x[j*3+1])**2 - (x[i*3+2] + x[j*3+2])**2
                })
        
        return obj, cons

    # This constraint definition is very slow due to lambda creation in loops.
    # Better to define vectorized constraint functions or use a penalty method.
    # Let's use a penalty method with L-BFGS-B or Nelder-Mead for speed, 
    # or optimize SLSQP constraints.
    
    # Actually, for N=26, SLSQP with explicit constraints might be too slow if not careful.
    # Let's try a different approach: minimize a penalty function.
    # Penalty = -sum(r) + lambda * sum(violations^2)
    # This allows using gradient-based methods like L-BFGS-B without explicit constraint handling,
    # but we need to ensure feasibility.
    
    # Alternatively, use `scipy.optimize.minimize` with `method='SLSQP'` but define constraints more efficiently.
    
    def solve_one(x0):
        x0 = x0.flatten()
        
        # We will use SLSQP
        # To speed up, we can pre-allocate constraint functions? 
        # Or just rely on the fact that N is small.
        
        # Let's try to vectorize the constraint evaluation for SLSQP.
        # SLSQP expects a list of constraint dicts.
        
        # A faster way: define a single function that returns all constraint values.
        # But SLSQP requires individual constraint functions or a single function with 'fun' returning array?
        # No, SLSQP 'fun' for constraints must return a scalar for each dict.
        # There is no vectorized constraint support in standard SLSQP interface directly in a simple way.
        
        # Given the time limit, let's try a multi-start with a simpler penalty-based optimization using L-BFGS-B.
        # We will minimize: -sum(r) + K * (sum of squared constraint violations)
        # This is not guaranteed to be feasible, but we can check feasibility afterwards and adjust.
        
        def penalty_obj(x):
            pts = x.reshape(N, 3)
            xs = pts[:, 0]
            ys = pts[:, 1]
            rs = pts[:, 2]
            
            # Objective
            obj = -np.sum(rs)
            
            # Boundary violations
            # x < r, x > 1-r, etc.
            v1 = np.maximum(0, rs - xs)
            v2 = np.maximum(0, xs + rs - 1)
            v3 = np.maximum(0, rs - ys)
            v4 = np.maximum(0, ys + rs - 1)
            v5 = np.maximum(0, 1e-6 - rs)
            
            # Pairwise violations
            # dist^2 - (ri+rj)^2 < 0
            # We want dist >= ri+rj
            # violation = max(0, (ri+rj) - dist)
            
            # Vectorized pairwise distance
            # dist^2 = (xi-xj)^2 + (yi-yj)^2
            # Create difference matrices
            dx = xs[:, np.newaxis] - xs[np.newaxis, :]
            dy = ys[:, np.newaxis] - ys[np.newaxis, :]
            dist_sq = dx**2 + dy**2
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            
            r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
            
            # Only consider i < j
            # Mask for upper triangle
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Violation is max(0, r_sum - dist)
            viol = np.maximum(0, r_sum - dist)
            viol = viol[mask]
            
            penalty = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2) + np.sum(v5**2) + np.sum(viol**2)
            
            return obj + 100.0 * penalty # High penalty weight
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 0.5)) # r
            
        res = minimize(penalty_obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
        
        # Check if result is valid
        pts = res.x.reshape(N, 3)
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Verify constraints strictly
        valid = True
        # Boundary
        if np.any(xs < rs - 1e-5) or np.any(xs + rs > 1 + 1e-5) or \
           np.any(ys < rs - 1e-5) or np.any(ys + rs > 1 + 1e-5):
            valid = False
            
        # Pairwise
        if valid:
            dx = xs[:, np.newaxis] - xs[np.newaxis, :]
            dy = ys[:, np.newaxis] - ys[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
            # Check i < j
            for i in range(N):
                for j in range(i+1, N):
                    if dist[i, j] < r_sum[i, j] - 1e-5:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            return res.x, -res.fun # score is sum of radii
        else:
            # Try to repair or return None
            # Simple repair: shrink radii slightly
            scale = 0.9
            pts[:, 2] *= scale
            return pts.flatten(), np.sum(pts[:, 2]) # Lower score, but valid?
            # Actually, if invalid, better to return None and let other starts try.
            # Or return a valid sub-solution.
            # For now, return None if invalid.
            return None, -np.inf

    # Initial guesses
    initial_guesses = [hexagonal_initial()]
    # Add some random starts
    for _ in range(5):
        initial_guesses.append(random_initial())
        
    for x0 in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            x_opt, score = solve_one(x0)
            if x_opt is not None and score > best_score:
                best_score = score
                best_result = x_opt
        except Exception:
            pass

    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result.reshape(N, 3)
