"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Pre-calculate pairs for constraints
    pairs = list(itertools.combinations(range(N), 2))
    n_pairs = len(pairs)
    
    # Define constraints and objective for optimization
    # Variables: x = [c_1 (3), c_2 (3), ..., c_N (3), r]
    # Total dimensions: 3*N + 1
    
    def objective(x):
        return -x[-1]  # Maximize r => Minimize -r
    
    def constraints(x):
        c = x[:3*N].reshape(N, 3)
        r = x[-1]
        cons = []
        
        # Pairwise distance constraints: 2r - |c_i - c_j| <= 0
        for i, j in pairs:
            dist = np.linalg.norm(c[i] - c[j])
            cons.append(2*r - dist)
            
        # Boundary constraints: r + |c_ik| - 0.5 <= 0
        for i in range(N):
            for k in range(3):
                cons.append(r + abs(c[i, k]) - 0.5)
                
        return cons
    
    def make_constraints_func(x):
        # scipy expects a list of dicts or a function returning a list of floats
        # Let's use a function that returns a list of floats for 'ineq'
        c = x[:3*N].reshape(N, 3)
        r = x[-1]
        vals = []
        for i, j in pairs:
            dist = np.linalg.norm(c[i] - c[j])
            vals.append(2*r - dist)
        for i in range(N):
            for k in range(3):
                vals.append(r + abs(c[i, k]) - 0.5)
        return vals

    # Bounds for variables
    bounds = []
    for _ in range(3*N):
        bounds.append((-0.5, 0.5))
    bounds.append((0.0, 0.5))
    
    best_r = 0.0
    best_x = None
    
    # Initial configurations
    # 1. Jittered Grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Randomly select 50 points from the 125 grid points and add jitter
    np.random.seed(42)
    idx = np.random.choice(len(grid), N, replace=False)
    x0_grid = grid[idx].copy()
    x0_grid += np.random.uniform(-0.05, 0.05, size=(N, 3))
    # Clip to valid range
    x0_grid = np.clip(x0_grid, -0.45, 0.45)
    r0 = 0.1
    x0_1 = np.concatenate([x0_grid.flatten(), [r0]])
    
    # 2. Random Points
    np.random.seed(123)
    x0_rand = np.random.uniform(-0.4, 0.4, size=(N, 3))
    r0 = 0.1
    x0_2 = np.concatenate([x0_rand.flatten(), [r0]])
    
    # 3. FCC-like or another random
    np.random.seed(456)
    x0_rand2 = np.random.uniform(-0.4, 0.4, size=(N, 3))
    r0 = 0.1
    x0_3 = np.concatenate([x0_rand2.flatten(), [r0]])
    
    starts = [x0_1, x0_2, x0_3]
    
    for x0 in starts:
        if time.time() > deadline - 20:
            break
            
        try:
            # SLSQP is good for small N with constraints
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': make_constraints_func},
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success:
                c = res.x[:3*N].reshape(N, 3)
                r = res.x[-1]
                
                # Verify validity and calculate actual score
                # Check bounds
                if np.any(np.abs(c) > 0.5 + 1e-9):
                    continue
                # Check pairwise distances
                min_dist = 1.0
                for i, j in pairs:
                    d = np.linalg.norm(c[i] - c[j])
                    if d < min_dist:
                        min_dist = d
                # Check boundary distances
                min_boundary = 1.0
                for i in range(N):
                    for k in range(3):
                        d = 0.5 - abs(c[i, k])
                        if d < min_boundary:
                            min_boundary = d
                
                actual_r = min(min_dist / 2.0, min_boundary)
                
                if actual_r > best_r:
                    best_r = actual_r
                    best_x = c.copy()
                    
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to simple grid if optimization fails
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_x

# EVOLVE-BLOCK-END
