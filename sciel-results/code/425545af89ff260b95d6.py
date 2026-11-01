import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_pos = None
    
    # Initial configurations
    inits = []
    
    # 1. Square Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Hexagonal Grid
    # Hex grid is denser
    # Row height = sqrt(3)/2 * side
    # We want to fit 50 circles.
    # Approx 7 cols, 7 rows? 7*7=49. 
    # Let's generate a hex grid and crop/center it.
    hex_points = []
    row = 0
    while len(hex_points) < N + 10:
        y = row * np.sqrt(3) / 2
        offset = 0.5 * (row % 2)
        col = 0
        while len(hex_points) < N + 10:
            x = col + offset
            hex_points.append((x, y))
            col += 1
        row += 1
    hex_points = np.array(hex_points)
    # Center and scale
    # We want side 1.
    # Current extent is roughly sqrt(N) * 1.
    # Let's just take first N and center them.
    hex_points = hex_points[:N]
    hex_points -= hex_points.mean(axis=0)
    # Scale to fit roughly in [-0.5, 0.5]
    # Max distance from center
    max_d = np.max(np.linalg.norm(hex_points, axis=1))
    if max_d > 0:
        hex_points *= 0.5 / max_d
    inits.append(hex_points)
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Optimization Function
    beta = 30.0 # Sharpness of the soft-min
    
    def objective_and_grad(v):
        pos = v.reshape(N, 2)
        
        # Distances to walls
        # d_wall = min(0.5 - |x|, 0.5 - |y|)
        # We need gradients for this too.
        # Let's compute pairwise and wall distances.
        
        # Pairwise
        # d_{ij} = ||c_i - c_j||
        # grad_{ij} w.r.t c_i is (c_i - c_j) / d_{ij}
        
        # Wall
        # d_{i,wall} = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # This is min of 4 linear functions.
        # The soft min will handle it.
        
        # To be efficient, we can compute all distances in a vectorized way?
        # N=50, N^2=2500. It's small.
        
        dists = []
        grads = [] # List of (index_i, index_j_or_wall, grad_vec_i, grad_vec_j)
        
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    dist = 1e-9
                    # Random gradient if coincident
                    g = np.random.randn(2)
                    g /= np.linalg.norm(g)
                else:
                    g = d / dist
                dists.append(dist)
                # Gradient of dist w.r.t pos[i] is g, w.r.t pos[j] is -g
                grads.append((i, j, g, -g))
        
        # Walls
        for i in range(N):
            # 4 constraints: x < 0.5, x > -0.5, y < 0.5, y > -0.5
            # dist = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
            # We add these as "virtual" points at infinity?
            # Or just add to the sum.
            # Let's treat them as distances to 4 lines.
            # d1 = 0.5 - x_i, grad = (-1, 0)
            # d2 = 0.5 + x_i, grad = (1, 0)
            # d3 = 0.5 - y_i, grad = (0, -1)
            # d4 = 0.5 + y_i, grad = (0, 1)
            
            # Note: If pos[i] is outside, these distances become negative.
            # exp(-beta * d) will be large.
            
            # We add these to the dists list and grads list.
            # For wall, it's only associated with index i.
            # We'll store as (i, -1, grad_i, None)
            
            d1 = 0.5 - pos[i, 0]
            dists.append(d1)
            grads.append((i, -1, np.array([-1.0, 0.0]), None))
            
            d2 = 0.5 + pos[i, 0]
            dists.append(d2)
            grads.append((i, -1, np.array([1.0, 0.0]), None))
            
            d3 = 0.5 - pos[i, 1]
            dists.append(d3)
            grads.append((i, -1, np.array([0.0, -1.0]), None))
            
            d4 = 0.5 + pos[i, 1]
            dists.append(d4)
            grads.append((i, -1, np.array([0.0, 1.0]), None))
            
        dists = np.array(dists)
        
        # Soft min calculation
        # To avoid overflow/underflow, subtract max dist?
        # We want exp(-beta * d). If d is large, exp is small.
        # If d is negative (overlap/outside), exp is large.
        # We want to minimize sum exp(-beta * d).
        # Let M = max(dists).
        # sum exp(-beta (d - M)) = exp(beta M) sum exp(-beta d).
        # log sum = beta M + log sum exp(-beta (d - M)).
        # f = -1/beta log S.
        
        # Actually, just compute directly. N is small.
        # But if d is very negative, -beta*d is large positive.
        # exp(100) is ~ 1e43. Sum might overflow?
        # Double max is ~ 1e308.
        # beta=30. If d=-10, -beta*d = 300. exp(300) ~ 1e130.
        # If 2500 terms, sum ~ 1e133. Log ~ 300.
        # It should be safe.
        
        exp_terms = np.exp(-beta * dists)
        S = np.sum(exp_terms)
        
        if S < 1e-300:
            S = 1e-300
        if S > 1e300:
            S = 1e300
            
        obj = -1.0 / beta * np.log(S)
        
        # Gradient
        # d obj / d x = (1/S) sum W_k (d d_k / d x)
        # W_k = exp_terms[k]
        
        grad_vec = np.zeros((N, 2))
        
        for k, (i, j, gi, gj) in enumerate(grads):
            w = exp_terms[k] / S
            grad_vec[i] += w * gi
            if j != -1:
                grad_vec[j] += w * gj
                
        # We want to MINIMIZE -obj.
        # So return -obj and -grad_vec
        return -obj, -grad_vec.flatten()

    # Run optimization for each init
    for init in inits:
        if time.time() > deadline:
            break
            
        v0 = init.flatten()
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        try:
            res = minimize(
                lambda v: objective_and_grad(v)[0],
                v0,
                jac=lambda v: objective_and_grad(v)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.success or res.fun < best_score: # Note: res.fun is -obj, so lower is better for -obj? 
                # Wait.
                # objective_and_grad returns -obj.
                # We minimize -obj.
                # So res.fun is the minimized value of -obj.
                # We want to maximize obj.
                # So we want to minimize res.fun.
                # best_score should track the MINIMUM of res.fun.
                # Let's rename best_val to min_neg_obj.
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_pos = res.x.reshape(N, 2)
                    
        except Exception:
            pass

    # If no optimization succeeded, return grid
    if best_pos is None:
        return inits[0]
        
    return best_pos
