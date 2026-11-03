"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Initial grid configuration
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    x0 = np.array(cells[:N])
    
    # Flatten: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    def flatten(x):
        return x.flatten()
    
    def unflatten(z):
        return z.reshape(N, 3)
    
    def penalty(z, mu=1e4):
        """Penalty for constraint violations: circles out of bounds or overlapping."""
        x = unflatten(z)
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        
        # Boundary violations
        pen = 0.0
        for i in range(N):
            if xs[i] - rs[i] < 0:
                pen += mu * (xs[i] - rs[i])**2
            if xs[i] + rs[i] > 1:
                pen += mu * (xs[i] + rs[i] - 1)**2
            if ys[i] - rs[i] < 0:
                pen += mu * (ys[i] - rs[i])**2
            if ys[i] + rs[i] > 1:
                pen += mu * (ys[i] + rs[i] - 1)**2
        
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    pen += mu * (min_dist - dist)**2
        
        return pen
    
    def objective(z):
        """Maximize sum of radii, penalize violations."""
        x = unflatten(z)
        rs = x[:, 2]
        return -np.sum(rs) + penalty(z)
    
    # Optimize with L-BFGS-B
    z0 = flatten(x0)
    
    # Run optimization
    result = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    
    # Check deadline
    if time.time() > deadline - 5:
        # Return initial grid if running out of time
        return x0
    
    x_opt = unflatten(result.x)
    
    # Validate and repair if needed
    def is_valid(x, tol=1e-6):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol or ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True
    
    if is_valid(x_opt):
        return x_opt
    else:
        # Return the better of initial or optimized
        if -result.fun < -np.sum(x0[:, 2]):
            return x_opt
        else:
            return x0

# EVOLVE-BLOCK-END
