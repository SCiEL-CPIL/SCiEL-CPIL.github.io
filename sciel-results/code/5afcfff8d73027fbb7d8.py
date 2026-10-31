import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    def unpack(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def pack(x, y, r):
        z = np.empty(3 * N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def objective(z):
        x, y, r = unpack(z)
        
        # Negative sum of radii (to be minimized)
        base_cost = -np.sum(r)
        
        # Overlap Penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                required = r[i] + r[j]
                if dist < required:
                    overlap_pen += (required - dist)**2
        
        # Boundary Penalty
        bound_pen = 0.0
        for i in range(N):
            # x bounds
            if x[i] < r[i]:
                bound_pen += (r[i] - x[i])**2
            if x[i] > 1 - r[i]:
                bound_pen += (x[i] - (1 - r[i]))**2
            # y bounds
            if y[i] < r[i]:
                bound_pen += (r[i] - y[i])**2
            if y[i] > 1 - r[i]:
                bound_pen += (y[i] - (1 - r[i]))**2
        
        # Weights
        w_overlap = 1000.0
        w_bound = 1000.0
        
        return base_cost + w_overlap * overlap_pen + w_bound * bound_pen

    def gradient(z):
        # Numerical gradient can be slow, but L-BFGS-B can use numerical.
        # However, for speed, let's provide analytical if possible, or rely on finite diff.
        # Given the complexity of max(0, x), numerical is safer for implementation correctness here.
        # But finite difference for 78 variables * 1000s of iterations might be slow.
        # Let's try to use a smooth approximation for the penalty to allow analytical grad?
        # No, let's stick to L-BFGS-B with numerical gradient for simplicity, but limit iterations.
        # Actually, let's implement a simple numerical gradient inside the optimizer if not provided?
        # scipy.optimize.minimize with method='L-BFGS-B' uses numerical gradient if jac is None.
        # This will be slow. Let's try to speed it up or use SLSQP with constraints?
        # SLSQP allows explicit constraints.
        
        # Let's switch strategy to SLSQP with constraints, which is more robust for this size.
        # But SLSQP is also slow.
        
        # Alternative: Use the penalty method with L-BFGS-B but optimize the penalty calculation.
        # Vectorized penalty calculation is key.
        pass

    def objective_fast(z):
        x, y, r = unpack(z)
        
        # Vectorized overlap penalty
        # This is O(N^2), but vectorized with numpy
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        req = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Only consider upper triangle to avoid double counting and self-overlap
        # We can use a mask
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_ij = req[iu]
        viol = np.maximum(0, r_ij - d_ij)
        overlap_pen = np.sum(viol**2)
        
        # Vectorized boundary penalty
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, x - (1 - r))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, y - (1 - r))
        
        bound_pen = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                    np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        base_cost = -np.sum(r)
        
        w_overlap = 1000.0
        w_bound = 1000.0
        
        return base_cost + w_overlap * overlap_pen + w_bound * bound_pen

    def generate_initial_solutions(num_starts):
        solutions = []
        # 1. Grid start
        side = int(np.ceil(np.sqrt(N)))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N: break
                xs[count] = (i + 0.5) / side
                ys[count] = (j + 0.5) / side
                rs[count] = 0.5 / side * 0.7
                count += 1
        # Add noise
        xs += np.random.uniform(-0.02, 0.02, N)
        ys += np.random.uniform(-0.02, 0.02, N)
        rs += np.random.uniform(-0.01, 0.01, N)
        # Clip
        xs = np.clip(xs, 0.05, 0.95)
        ys = np.clip(ys, 0.05, 0.95)
        rs = np.clip(rs, 0.01, 0.4)
        solutions.append(pack(xs, ys, rs))
        
        # 2. Random starts with simple relaxation
        for _ in range(num_starts - 1):
            xs = np.random.uniform(0.1, 0.9, N)
            ys = np.random.uniform(0.1, 0.9, N)
            rs = np.random.uniform(0.05, 0.15, N)
            # Simple relaxation to reduce initial overlaps
            for _ in range(50):
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        req = rs[i] + rs[j]
                        if dist < req and dist > 1e-6:
                            # Push apart
                            push = (req - dist) / 2
                            xs[i] += (dx/dist) * push
                            ys[i] += (dy/dist) * push
                            xs[j] -= (dx/dist) * push
                            ys[j] -= (dy/dist) * push
                # Boundary push
                for i in range(N):
                    if xs[i] < rs[i]: xs[i] = rs[i]
                    if xs[i] > 1-rs[i]: xs[i] = 1-rs[i]
                    if ys[i] < rs[i]: ys[i] = rs[i]
                    if ys[i] > 1-rs[i]: ys[i] = 1-rs[i]
            solutions.append(pack(xs, ys, rs))
            
        return solutions

    start_time = time.time()
    time_limit = 110
    
    best_score = -np.inf
    best_solution = None
    
    # Run multiple starts
    num_starts = 5
    inits = generate_initial_solutions(num_starts)
    
    bounds = []
    for _ in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    for z0 in inits:
        if time.time() - start_time > time_limit:
            break
            
        res = minimize(objective_fast, z0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if res.success:
            z_opt = res.x
            x, y, r = unpack(z_opt)
            
            # Check validity
            valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6:
                    valid = False
                    break
                if y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_solution = z_opt.copy()
        
    if best_solution is not None:
        x, y, r = unpack(best_solution)
        return np.column_stack((x, y, r))
    else:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N: break
                xs[count] = (i + 0.5) / side
                ys[count] = (j + 0.5) / side
                rs[count] = 0.5 / side * 0.7
                count += 1
        return np.column_stack((xs, ys, rs))
