import numpy as np
import time
from scipy.optimize import minimize

N = 32
TIME_LIMIT = 120.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Helper to evaluate and polish
    def polish_and_eval(vars):
        X = vars[0::3].copy()
        Y = vars[1::3].copy()
        R = vars[2::3].copy()
        
        # Polish radii: maximize each radius given fixed positions
        # Iterate a few times to allow radii to adjust to each other
        for _ in range(10):
            for i in range(N):
                min_dist = np.inf
                # Distance to boundaries
                min_dist = min(min_dist, X[i], 1.0 - X[i], Y[i], 1.0 - Y[i])
                # Distance to other circles
                for j in range(N):
                    if i == j: continue
                    d = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
                    min_dist = min(min_dist, d - R[j])
                R[i] = max(1e-6, min_dist * 0.999) # Safety margin
                
        # Calculate score
        score = np.sum(R)
        
        # Check validity
        valid = True
        for i in range(N):
            if X[i] - R[i] < -1e-5 or 1.0 - X[i] - R[i] < -1e-5 or Y[i] - R[i] < -1e-5 or 1.0 - Y[i] - R[i] < -1e-5:
                valid = False
            for j in range(i+1, N):
                d = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
                if d < R[i] + R[j] - 1e-5:
                    valid = False
        
        if not valid:
            score = -np.inf
            
        if score > best_score:
            return score, np.column_stack((X, Y, R))
        return score, None

    # Objective function with Soft Contact Graph
    def objective(vars, weight_ref):
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        obj = -np.sum(R)
        pen = 0.0
        weight = weight_ref[0]
        
        # Pairwise penalties with soft weighting
        # We want to penalize overlaps, but focus on close pairs
        # Using a simple distance-based weight to simulate "contact graph"
        for i in range(N):
            for j in range(i+1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    # Soft weight: higher weight for larger overlaps or closer pairs
                    # This helps the optimizer focus on critical constraints
                    w = 1.0 + 10.0 * overlap
                    pen += w * overlap**2
        
        # Boundary penalties
        for i in range(N):
            if X[i] - R[i] < 0: pen += 10.0 * (X[i] - R[i])**2
            if 1.0 - X[i] - R[i] < 0: pen += 10.0 * (1.0 - X[i] - R[i])**2
            if Y[i] - R[i] < 0: pen += 10.0 * (Y[i] - R[i])**2
            if 1.0 - Y[i] - R[i] < 0: pen += 10.0 * (1.0 - Y[i] - R[i])**2
            
        return obj + weight * pen

    def objective_grad(vars, weight_ref):
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        weight = weight_ref[0]
        
        for i in range(N):
            for j in range(i+1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10: continue
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    w = 1.0 + 10.0 * overlap
                    factor = 2.0 * w * overlap
                    # Gradient of (w * overlap^2) w.r.t positions
                    # Note: w depends on overlap, which depends on d
                    # Simplified gradient ignoring d(w)/d(overlap) for speed, 
                    # or approximating it. Let's use the simplified version first.
                    # Actually, let's just use the base penalty gradient scaled by w
                    grad[3*i]   += weight * factor * (-dx/d)
                    grad[3*i+1] += weight * factor * (-dy/d)
                    grad[3*j]   += weight * factor * (dx/d)
                    grad[3*j+1] += weight * factor * (dy/d)
                    grad[3*i+2] += weight * factor
                    grad[3*j+2] += weight * factor

        for i in range(N):
            val = X[i] - R[i]
            if val < 0:
                grad[3*i]   += weight * 20.0 * val
                grad[3*i+2] += weight * -20.0 * val
            
            val = 1.0 - X[i] - R[i]
            if val < 0:
                grad[3*i]   += weight * -20.0 * val
                grad[3*i+2] += weight * -20.0 * val
            
            val = Y[i] - R[i]
            if val < 0:
                grad[3*i+1] += weight * 20.0 * val
                grad[3*i+2] += weight * -20.0 * val
            
            val = 1.0 - Y[i] - R[i]
            if val < 0:
                grad[3*i+1] += weight * -20.0 * val
                grad[3*i+2] += weight * -20.0 * val

        return grad

    def run_optimization(init_pts, init_r):
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = init_pts[i, 0]
            x0[3*i+1] = init_pts[i, 1]
            x0[3*i+2] = init_r
            
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((1e-5, 0.5))
            
        weight_ref = [100.0]
        
        # Adaptive penalty: increase weight if objective is still high (violations)
        try:
            res = minimize(objective, x0, args=(weight_ref,), method='L-BFGS-B', 
                           jac=lambda v: objective_grad(v, weight_ref), 
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # If it didn't converge well, try again with higher weight
            if res.fun > -np.sum(res.x[2::3]) + 1e-4: # Check for residual penalty
                weight_ref[0] = 1000.0
                res = minimize(objective, res.x, args=(weight_ref,), method='L-BFGS-B', 
                               jac=lambda v: objective_grad(v, weight_ref), 
                               bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            return res.x
        except Exception as e:
            return x0

    # Initialization Strategies
    
    # 1. Hexagonal Lattice
    def hex_init():
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(init_pts[i] - init_pts[j])
                min_dist = min(min_dist, d)
        init_r = min_dist / 2 * 0.9
        return init_pts, init_r

    # 2. Grid with Jitter
    def grid_init():
        grid_size = 6
        pts = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) / grid_size + (np.random.rand() - 0.5) * 0.1
                y = (j + 0.5) / grid_size + (np.random.rand() - 0.5) * 0.1
                pts.append([x, y])
        pts = np.array(pts[:N])
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                min_dist = min(min_dist, d)
        init_r = min_dist / 2 * 0.9
        return pts, init_r

    # 3. Random
    def random_init():
        pts = np.random.rand(N, 2) * 0.8 + 0.1
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                min_dist = min(min_dist, d)
        init_r = min_dist / 2 * 0.9
        return pts, init_r

    inits = [
        ("hex", hex_init),
        ("grid", grid_init),
        ("random", random_init)
    ]

    for name, init_func in inits:
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
            
        init_pts, init_r = init_func()
        x_opt = run_optimization(init_pts, init_r)
        
        score, result = polish_and_eval(x_opt)
        if result is not None:
            best_result = result
            best_score = score

    # Final polish on the best result
    if best_result is not None:
        # One more pass of polishing
        X = best_result[:, 0].copy()
        Y = best_result[:, 1].copy()
        R = best_result[:, 2].copy()
        
        for _ in range(20):
            for i in range(N):
                min_dist = np.inf
                min_dist = min(min_dist, X[i], 1.0 - X[i], Y[i], 1.0 - Y[i])
                for j in range(N):
                    if i == j: continue
                    d = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
                    min_dist = min(min_dist, d - R[j])
                R[i] = max(1e-6, min_dist * 0.9999)
        
        best_result[:, 2] = R
        best_score = np.sum(R)

    if best_result is None:
        # Fallback
        best_result = np.zeros((N, 3))
        for i in range(N):
            best_result[i, 0] = (i % 6 + 0.5) / 6.0
            best_result[i, 1] = (i // 6 + 0.5) / 6.0
            best_result[i, 2] = 0.05

    return best_result
# EVOLVE-BLOCK-END
