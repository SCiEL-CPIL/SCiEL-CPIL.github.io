import numpy as np
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 500.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Pairs of indices for distance calculations
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)
    
    def unpack(v):
        """Unpack variable vector into points and t"""
        points = v[:n*d].reshape(n, d)
        t = v[-1]
        return points, t

    def objective(v):
        """Objective: Minimize -t (Maximize t)"""
        _, t = unpack(v)
        return -t

    def objective_grad(v):
        """Gradient of objective"""
        grad = np.zeros_like(v)
        grad[-1] = -1.0
        return grad

    def constraints_defs():
        """
        Returns list of constraint dictionaries for scipy.optimize.minimize
        """
        cons = []
        
        # 1. t^2 <= d_ij^2 for all pairs
        # g_ij(v) = t^2 - ||x_i - x_j||^2 <= 0
        for k in range(num_pairs):
            i, j = i_idx[k], j_idx[k]
            
            def con_func(v, i=i, j=j):
                points, t = unpack(v)
                diff = points[i] - points[j]
                return t*t - np.dot(diff, diff)
                
            def con_grad(v, i=i, j=j):
                points, t = unpack(v)
                diff = points[i] - points[j]
                grad = np.zeros_like(v)
                grad[-1] = 2*t
                grad[i*d] = -2*diff[0]
                grad[i*d+1] = -2*diff[1]
                grad[j*d] = 2*diff[0]
                grad[j*d+1] = 2*diff[1]
                return grad
                
            cons.append({'type': 'ineq', 'fun': con_func, 'jac': con_grad})
            
        # 2. d_ij^2 <= 1 for all pairs (Fix scale to max dist 1)
        # h_ij(v) = ||x_i - x_j||^2 - 1 <= 0
        for k in range(num_pairs):
            i, j = i_idx[k], j_idx[k]
            
            def con_func2(v, i=i, j=j):
                points, _ = unpack(v)
                diff = points[i] - points[j]
                return np.dot(diff, diff) - 1.0
                
            def con_grad2(v, i=i, j=j):
                points, _ = unpack(v)
                diff = points[i] - points[j]
                grad = np.zeros_like(v)
                grad[i*d] = 2*diff[0]
                grad[i*d+1] = 2*diff[1]
                grad[j*d] = -2*diff[0]
                grad[j*d+1] = -2*diff[1]
                return grad
                
            cons.append({'type': 'ineq', 'fun': con_func2, 'jac': con_grad2})
            
        # 3. Centroid fixed to 0
        # Sum(x_i) = 0
        def centroid_func(v):
            points, _ = unpack(v)
            return np.sum(points, axis=0)
            
        def centroid_grad(v):
            points, _ = unpack(v)
            grad = np.zeros_like(v)
            # d(sum(x_i))/dx_i = 1
            for i in range(n):
                grad[i*d] = 1.0
                grad[i*d+1] = 1.0
            return grad
            
        cons.append({'type': 'eq', 'fun': centroid_func, 'jac': centroid_grad})
        
        return cons

    constraints = constraints_defs()
    
    # Initial Guesses
    initial_guesses = []
    
    # 1. 4x4 Grid
    x_grid = np.linspace(-1.5, 1.5, 4)
    y_grid = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x_grid, y_grid)
    pts_grid = np.column_stack((xx.ravel(), yy.ravel()))
    pts_grid -= np.mean(pts_grid, axis=0)
    # Normalize scale roughly
    max_d = np.max(np.linalg.norm(pts_grid[:, np.newaxis, :] - pts_grid[np.newaxis, :, :], axis=2))
    if max_d > 0:
        pts_grid /= max_d
    # Estimate t as min distance
    dists_grid = np.linalg.norm(pts_grid[:, np.newaxis, :] - pts_grid[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists_grid, np.inf)
    t_grid = np.min(dists_grid)
    v_grid = np.append(pts_grid.ravel(), t_grid)
    initial_guesses.append(v_grid)
    
    # 2. Hexagonal Packing
    pts_hex = []
    # Center
    pts_hex.append([0.0, 0.0])
    # Ring 1 (6 points)
    for k in range(6):
        angle = k * np.pi / 3
        pts_hex.append([np.cos(angle), np.sin(angle)])
    # Ring 2 (12 points, we need 15 more, so 9 from ring 2? 1+6+12=19. We need 16. So 1+6+9=16)
    # Let's just take first 15 points of a hex lattice
    # Standard hex lattice generation
    def hex_coord(q, r):
        x = 1.5 * q
        y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
        return [x, y]
        
    pts_hex = []
    # Generate enough hex points
    for q in range(-3, 4):
        for r in range(-3, 4):
            pts_hex.append(hex_coord(q, r))
    pts_hex = np.array(pts_hex[:16])
    pts_hex -= np.mean(pts_hex, axis=0)
    max_d_hex = np.max(np.linalg.norm(pts_hex[:, np.newaxis, :] - pts_hex[np.newaxis, :, :], axis=2))
    if max_d_hex > 0:
        pts_hex /= max_d_hex
    dists_hex = np.linalg.norm(pts_hex[:, np.newaxis, :] - pts_hex[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists_hex, np.inf)
    t_hex = np.min(dists_hex)
    v_hex = np.append(pts_hex.ravel(), t_hex)
    initial_guesses.append(v_hex)
    
    # 3. Random on Circle
    np.random.seed(42)
    for _ in range(3):
        angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        radii = np.random.uniform(0.5, 1.0, n)
        pts_rand = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        pts_rand -= np.mean(pts_rand, axis=0)
        max_d_rand = np.max(np.linalg.norm(pts_rand[:, np.newaxis, :] - pts_rand[np.newaxis, :, :], axis=2))
        if max_d_rand > 0:
            pts_rand /= max_d_rand
        dists_rand = np.linalg.norm(pts_rand[:, np.newaxis, :] - pts_rand[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists_rand, np.inf)
        t_rand = np.min(dists_rand)
        v_rand = np.append(pts_rand.ravel(), t_rand)
        initial_guesses.append(v_rand)

    # Optimization Loop
    for i, v0 in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                jac=objective_grad, 
                constraints=constraints, 
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < 0: # res.fun is -t, so lower is better
                v_best = res.x
                points, t = unpack(v_best)
                
                # Validate and calculate score
                dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dists, np.inf)
                d_min = np.min(dists)
                d_max = np.max(dists)
                
                if d_max > 0:
                    score = (d_min / d_max) ** 2
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
        except Exception as e:
            print(f"Optimization failed for guess {i}: {e}")
            continue

    if best_points is None:
        # Fallback to grid if optimization failed completely
        return initial_guesses[0][:n*d].reshape(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
