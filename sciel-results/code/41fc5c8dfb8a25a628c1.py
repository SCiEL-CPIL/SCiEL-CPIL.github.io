"""Pack 50 equal circles in a unit square using SLSQP constrained optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _init_hexagonal(n_x, n_y, spacing):
    """Create a hexagonal lattice of points, centered and scaled."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing
            y = j * dy
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    points = points[:N]
    # Center
    points -= np.mean(points, axis=0)
    # Scale to fit in [-0.45, 0.45]
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points *= (0.45 / max_ext)
    return points

def _local_search(centers, time_limit=2.0):
    """
    Simple local search to polish the initial guess before SLSQP.
    Moves points that are tight against walls or each other.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _radius(best_centers)
    step_size = 0.005
    
    while time.time() - start_time < time_limit:
        # Find current radius
        r = best_r
        
        # Identify tight points
        # Wall tight
        wall_tight = np.where(np.abs(0.5 - np.abs(best_centers)) < 1e-6)[0]
        # Pair tight
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_tight_mask = dist < 2 * r * 1.01
        pair_tight_indices = np.where(pair_tight_mask)[0]
        unique_tight = np.unique(np.concatenate([wall_tight, pair_tight_indices]))
        
        if len(unique_tight) == 0:
            break
            
        # Pick a random tight point
        idx = unique_tight[np.random.randint(len(unique_tight))]
        
        # Propose move away from constraints
        # Simple random perturbation
        move = np.random.normal(0, step_size, size=2)
        new_centers = best_centers.copy()
        new_centers[idx] += move
        
        # Check bounds strictly
        if np.any(np.abs(new_centers) > 0.5):
            continue
            
        new_r = _radius(new_centers)
        
        if new_r > best_r:
            best_centers = new_centers
            best_r = new_r
        else:
            # Small probability to accept worse to escape local minima
            if np.random.rand() < 0.05:
                best_centers = new_centers
                # best_r stays the same

    return best_centers

def construct():
    start_time = time.time()
    best_centers = None
    best_r = 0.0
    
    # 1. Initialize with a good hexagonal grid
    # 7x7 grid has 49 points, add 1 more? Or 8x7=56, take 50?
    # Let's try 8x7 hex grid and take first 50
    init = _init_hexagonal(8, 7, 0.13)
    
    # 2. Local search to polish
    if time.time() - start_time < DEADLINE - 30:
        init = _local_search(init, time_limit=min(3.0, DEADLINE - 30 - (time.time() - start_time)))
    
    # 3. SLSQP Optimization
    # Variables: x[0:N] = centers_x, x[N:2N] = centers_y, x[2N] = r
    # Objective: minimize -r
    # Constraints:
    #   g_i(x) = ||c_i - c_j||^2 - (2r)^2 >= 0  for all i < j
    #   h_k(x) = (0.5 - r)^2 - c_ik^2 >= 0 for all i, k
    
    x0 = np.zeros(2 * N + 1)
    x0[:N] = init[:, 0]
    x0[N:2*N] = init[:, 1]
    x0[2*N] = _radius(init)
    
    def objective(x):
        return -x[2*N]
    
    def objective_grad(x):
        g = np.zeros_like(x)
        g[2*N] = -1.0
        return g
        
    # Constraints
    # We generate constraints dynamically. To save time, we might only use active ones?
    # For N=50, number of pairs is 1225. Number of wall constraints is 100. Total ~1325 constraints.
    # SLSQP can handle this, but it might be slow.
    # Let's use a sparse approach or just pass the list.
    
    cons = []
    
    # Pairwise constraints
    # c_i = x[i], c_j = x[j]
    # dist^2 - 4r^2 >= 0
    for i in range(N):
        for j in range(i+1, N):
            # fun(x) = (x[i]-x[j])^2 + (x[N+i]-x[N+j])^2 - 4*x[2N]^2
            # grad = [0.. -2(x[i]-x[j]) .. 2(x[i]-x[j]) .. 0.. -2(x[N+i]-x[N+j]) .. 2(x[N+i]-x[N+j]) .. -8x[2N]]
            
            def fun(x, i=i, j=j):
                dx = x[i] - x[j]
                dy = x[N+i] - x[N+j]
                r = x[2*N]
                return dx*dx + dy*dy - 4*r*r
            
            def grad(x, i=i, j=j):
                g = np.zeros_like(x)
                dx = x[i] - x[j]
                dy = x[N+i] - x[N+j]
                r = x[2*N]
                g[i] = -2*dx
                g[j] = 2*dx
                g[N+i] = -2*dy
                g[N+j] = 2*dy
                g[2*N] = -8*r
                return g
            
            cons.append({'type': 'ineq', 'fun': fun, 'jac': grad})
            
    # Wall constraints
    # (0.5 - r)^2 - x[i]^2 >= 0
    for i in range(N):
        # x coord
        def fun_x(x, i=i):
            r = x[2*N]
            return (0.5 - r)**2 - x[i]**2
            
        def grad_x(x, i=i):
            g = np.zeros_like(x)
            r = x[2*N]
            g[i] = -2*x[i]
            g[2*N] = -2*(0.5 - r)
            return g
            
        cons.append({'type': 'ineq', 'fun': fun_x, 'jac': grad_x})
        
        # y coord
        def fun_y(x, i=i):
            r = x[2*N]
            return (0.5 - r)**2 - x[N+i]**2
            
        def grad_y(x, i=i):
            g = np.zeros_like(x)
            r = x[2*N]
            g[N+i] = -2*x[N+i]
            g[2*N] = -2*(0.5 - r)
            return g
            
        cons.append({'type': 'ineq', 'fun': fun_y, 'jac': grad_y})
        
    # Bounds
    # r > 0
    # centers in [-0.5, 0.5]
    bounds = [(None, None)] * (2*N) + [(0.0, 0.5)]
    
    # Run SLSQP
    # This might take a while. We need to monitor time.
    # SLSQP in scipy doesn't have a time limit callback directly, but we can check after.
    # If it takes too long, we return the best found so far? 
    # Actually, minimize returns the result. If it times out, we might not get a result.
    # To be safe, we can run it in a way that we can interrupt? 
    # Or just hope it finishes. With 101 vars and 1325 constraints, it should be okay if initialized well.
    
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad, bounds=bounds, constraints=cons, 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success:
            new_centers = np.column_stack((res.x[:N], res.x[N:2*N]))
            new_r = _radius(new_centers)
            
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            # Even if not successful, the result might be better
            new_centers = np.column_stack((res.x[:N], res.x[N:2*N]))
            new_r = _radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
                
    except Exception as e:
        pass
        
    # If we didn't get a valid result from SLSQP, fall back to the local search result
    if best_centers is None:
        best_centers = init
        best_r = _radius(best_centers)
        
    return best_centers

if __name__ == "__main__":
    c = construct()
    print(_radius(c))
