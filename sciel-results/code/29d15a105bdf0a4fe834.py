import numpy as np
from scipy.optimize import minimize
import time

N = 21

def soft_min(distances, k=100):
    # Soft min of a list of distances
    # Returns a scalar approximating min(distances)
    # Using log-sum-exp for stability
    d = np.array(distances)
    d_min = np.min(d)
    # Shift to avoid overflow
    exp_terms = np.exp(-k * (d - d_min))
    return d_min - (1.0 / k) * np.log(np.sum(exp_terms))

def objective(params, k=100):
    # params: [W, x1, y1, ..., xN, yN]
    W = params[0]
    H = 2.0 - W
    
    # Bounds for W: 0.5 to 1.5
    if W <= 0.5 or W >= 1.5:
        return 1e10
        
    centers = params[1:].reshape(N, 2)
    
    # Check if centers are within bounds (with margin)
    # If any center is out of bounds, return high penalty
    if np.any(centers[:, 0] < 0) or np.any(centers[:, 0] > W) or \
       np.any(centers[:, 1] < 0) or np.any(centers[:, 1] > H):
        return 1e10

    distances = []
    
    # Distances to walls
    for i in range(N):
        x, y = centers[i]
        distances.append(x)
        distances.append(y)
        distances.append(W - x)
        distances.append(H - y)
        
    # Distances to other circles (radius is half the distance)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist / 2.0)
            
    # We want to maximize the minimum distance
    # So we minimize the negative soft-min
    return -soft_min(distances, k)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_score = -np.inf
    best_params = None
    
    # Try different aspect ratios
    W_starts = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7]
    
    # Initial configurations
    def grid_init(W, H):
        # Approximate grid
        side = int(np.ceil(np.sqrt(N)))
        # Adjust grid to fit W, H
        # Simple grid
        xs = np.linspace(W / (side+1), W - W/(side+1), side)
        ys = np.linspace(H / (side+1), H - H/(side+1), side)
        # Create mesh
        points = []
        count = 0
        for y in ys:
            for x in xs:
                if count < N:
                    points.append([x, y])
                    count += 1
        # If not enough points, add some random
        while len(points) < N:
            points.append([np.random.uniform(0.1, W-0.1), np.random.uniform(0.1, H-0.1)])
        return np.array(points[:N])

    def random_init(W, H):
        return np.random.uniform(0.1, W-0.1, (N, 2))

    for W0 in W_starts:
        if time.time() > deadline:
            break
        H0 = 2.0 - W0
        
        # Try a few initializations
        inits = [
            grid_init(W0, H0),
            random_init(W0, H0),
            random_init(W0, H0)
        ]
        
        for init_centers in inits:
            if time.time() > deadline:
                break
            
            x0 = np.concatenate(([W0], init_centers.flatten()))
            
            # Bounds for W
            # Bounds for x, y are handled by penalty, but we can provide bounds to optimizer
            # L-BFGS-B supports bounds
            bounds = [(0.5, 1.5)]
            for i in range(N):
                bounds.append((0.0, W0)) # X bound depends on W, which is variable. 
                # This is tricky. L-BFGS-B bounds are static.
                # So we rely on penalty for X, Y bounds or use a transformation.
                # Given the complexity, penalty is safer for variable W.
                bounds.append((0.0, 2.0))
            
            # Actually, since W changes, the valid range for x changes.
            # The penalty function handles this.
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
                               bounds=bounds)
                
                if time.time() > deadline:
                    break

                if res.success or res.fun < best_score:
                    # Calculate actual min distance to verify
                    W_opt = res.x[0]
                    H_opt = 2.0 - W_opt
                    centers_opt = res.x[1:].reshape(N, 2)
                    
                    min_dist = np.inf
                    for i in range(N):
                        x, y = centers_opt[i]
                        min_dist = min(min_dist, x, y, W_opt - x, H_opt - y)
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.linalg.norm(centers_opt[i] - centers_opt[j]) / 2.0
                            min_dist = min(min_dist, d)
                    
                    # Score is the sum of radii. If we use equal radii = min_dist
                    # Sum = N * min_dist
                    # But the evaluator measures "raw value". 
                    # The task says "maximize the sum of their radii".
                    # If we set all radii to min_dist, sum = N * min_dist.
                    
                    if min_dist > best_score:
                        best_score = min_dist
                        best_params = (W_opt, centers_opt)
                        
            except Exception as e:
                pass

    if best_params is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    W_opt, centers_opt = best_params
    r = best_score
    
    # Construct output
    result = np.zeros((N, 3))
    result[:, 0] = centers_opt[:, 0]
    result[:, 1] = centers_opt[:, 1]
    result[:, 2] = r
    
    return result
