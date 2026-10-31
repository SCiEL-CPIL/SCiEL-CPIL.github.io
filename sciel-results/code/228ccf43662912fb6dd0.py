"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # We'll use a multi-start local optimization approach
    # Each circle has (x, y, r) as variables
    # Objective: maximize sum of radii
    # Constraints: circles don't overlap, stay in unit square
    
    best_solution = None
    best_score = -np.inf
    
    def compute_score(circles):
        """Compute the sum of radii, returning -inf if invalid."""
        n = len(circles)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check boundaries
        if np.any(r < 0) or np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or \
           np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
        
        # Check pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        
        return np.sum(r)
    
    def objective(params):
        """Negative sum of radii for minimization."""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        # Penalize violations heavily
        penalty = 0.0
        
        # Boundary violations
        penalty += np.sum(np.maximum(0, -x + r)) * 100
        penalty += np.sum(np.maximum(0, x - 1 + r)) * 100
        penalty += np.sum(np.maximum(0, -y + r)) * 100
        penalty += np.sum(np.maximum(0, y - 1 + r)) * 100
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap * 100
        
        return -np.sum(r) + penalty
    
    def make_circles_from_params(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        return np.column_stack([x, y, r])
    
    def grid_init():
        """Generate a grid-based initial solution."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                cr = 0.5 / side - 0.01
                cells.append([cx, cy, cr])
        return np.array(cells)
    
    def random_init():
        """Generate a random initial solution."""
        # Place circles with small random radii first
        circles = []
        for _ in range(N):
            r = random.uniform(0.01, 0.05)
            x = random.uniform(r, 1 - r)
            y = random.uniform(r, 1 - r)
            circles.append([x, y, r])
        return np.array(circles)
    
    def params_from_circles(circles):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        return np.concatenate([x, y, r])
    
    # Try multiple initializations
    inits = [grid_init()]
    for _ in range(5):
        inits.append(random_init())
    
    for init_circles in inits:
        if time.time() > deadline:
            break
        
        init_params = params_from_circles(init_circles)
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0, 0.5))  # r
        
        try:
            result = minimize(objective, init_params, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 500, 'maxfun': 5000})
            
            if result.success or result.fun < best_score:
                final_circles = make_circles_from_params(result.x)
                score = compute_score(final_circles)
                
                if score > best_score:
                    best_score = score
                    best_solution = final_circles
                    
        except Exception:
            pass
    
    # If no valid solution found, return the grid
    if best_solution is None:
        return grid_init()
    
    return best_solution
# EVOLVE-BLOCK-END
