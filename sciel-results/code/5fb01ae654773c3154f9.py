import numpy as np
from scipy.optimize import minimize
import itertools

N = 26

def construct():
    # We will try multiple restarts to find a good packing
    best_val = -np.inf
    best_sol = None
    
    # Number of restarts to try within time limit
    # Each optimization run should be fast enough to allow several restarts
    n_restarts = 10
    
    for k in range(n_restarts):
        # Initialize positions
        # Use a grid with some jitter to break symmetry
        side = int(np.ceil(np.sqrt(N)))
        x0 = np.zeros(3 * N)
        
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                # Center of cell
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                # Small jitter
                cx += np.random.uniform(-0.01, 0.01)
                cy += np.random.uniform(-0.01, 0.01)
                # Initial radius: small enough to not overlap in grid
                r_init = 0.4 / side
                
                x0[idx] = cx
                x0[idx+1] = cy
                x0[idx+2] = r_init
                idx += 3
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
            
        # Objective function
        def objective(p):
            # p is array of shape (3N,)
            coords = p[:2*N].reshape(N, 2)
            radii = p[2*N:]
            
            # 1. Reward for larger radii (we minimize negative sum)
            val = -np.sum(radii)
            
            # 2. Boundary penalties
            # x < r, y < r, x > 1-r, y > 1-r
            # Violation if x < r => r - x > 0
            pen_x_min = np.maximum(0, radii - coords[:, 0])
            pen_y_min = np.maximum(0, radii - coords[:, 1])
            pen_x_max = np.maximum(0, radii - (1 - coords[:, 0]))
            pen_y_max = np.maximum(0, radii - (1 - coords[:, 1]))
            
            boundary_pen = np.sum(pen_x_min**2) + np.sum(pen_y_min**2) + \
                           np.sum(pen_x_max**2) + np.sum(pen_y_max**2)
            
            # 3. Overlap penalties
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        overlap_pen += (min_dist - dist)**2
            
            # Weights for penalties
            # Large weights to ensure constraints are satisfied
            weight_boundary = 100.0
            weight_overlap = 100.0
            
            return val + weight_boundary * boundary_pen + weight_overlap * overlap_pen

        # Run optimization
        # Use L-BFGS-B as it handles bounds well
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.success or result.fun < best_val:
            # Extract solution
            sol = result.x
            # Check validity and calculate score
            coords = sol[:2*N].reshape(N, 2)
            radii = sol[2*N:]
            
            # Verify constraints strictly to avoid invalid submission
            valid = True
            for i in range(N):
                x, y, r = coords[i], coords[i][1] if False else sol[2*i+1], sol[2*i+2]
                # Correct extraction
                x = sol[2*i]
                y = sol[2*i+1]
                r = sol[2*i+2]
                
                if x < r - 1e-6 or y < r - 1e-6 or x > 1 - r + 1e-6 or y > 1 - r + 1e-6:
                    valid = False
                    break
                
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.linalg.norm(coords[i] - coords[j])
                        if dist < radii[i] + radii[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(radii)
                if score > best_val:
                    best_val = score
                    best_sol = sol.copy()
    
    if best_sol is None:
        # Fallback to seed if optimization failed completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Format output
    result_arr = best_sol.reshape(N, 3)
    # Ensure no tiny negative radii or out of bounds due to floating point errors
    result_arr[:, 2] = np.clip(result_arr[:, 2], 0, 0.5)
    result_arr[:, 0] = np.clip(result_arr[:, 0], 0, 1)
    result_arr[:, 1] = np.clip(result_arr[:, 1], 0, 1)
    
    return result_arr
