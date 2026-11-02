import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_sum = 0
    best_sol = None
    
    # Try several restarts
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Initialize
        # Grid-like initialization
        side = int(np.ceil(np.sqrt(N)))
        # Create grid points
        xs = np.linspace(0.2, 0.8, side)
        ys = np.linspace(0.2, 0.8, side)
        grid_points = []
        for x in xs:
            for y in ys:
                grid_points.append([x, y])
        
        # If we have more points than N, take first N. If fewer, add random?
        # 5x5 = 25. We need 26.
        # Let's just use random points for simplicity and robustness against symmetry traps.
        # But random is slower to converge.
        # Hybrid: Grid + Random perturbation.
        
        # Let's use a 5x5 grid + 1 random point.
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + 0.8*(i+0.5)/5, 0.1 + 0.8*(j+0.5)/5])
        # Add 26th point
        pts.append([0.5 + np.random.uniform(-0.1, 0.1), 0.5 + np.random.uniform(-0.1, 0.1)])
        
        # Shuffle to break symmetry?
        np.random.shuffle(pts)
        
        x0 = np.array(pts)
        # Initial radii: estimate based on distance to neighbors
        r0 = np.zeros(N)
        for i in range(N):
            min_dist = 1.0
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(x0[i] - x0[j])
                    min_dist = min(min_dist, d)
            # Distance to boundary
            min_dist = min(min_dist, x0[i][0], 1-x0[i][0], x0[i][1], 1-x0[i][1])
            r0[i] = min_dist * 0.49 # 0.49 to be safe
            
        # Optimize
        res = optimize_packing(x0, r0)
        
        # Check validity
        if res is not None:
            s = np.sum(res[:, 2])
            if s > best_sum:
                best_sum = s
                best_sol = res
                
    # Return best valid solution
    if best_sol is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_sol

def optimize_packing(x0, r0):
    # Combine x, y, r into a single vector
    # x: N, y: N, r: N -> 3N variables
    z0 = np.concatenate([x0[:,0], x0[:,1], r0])
    
    def objective(z):
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        
        # Penalty for boundaries
        pen_bound = 0
        for i in range(N):
            pen_bound += max(0, r[i] - x[i])**2
            pen_bound += max(0, r[i] - (1-x[i]))**2
            pen_bound += max(0, r[i] - y[i])**2
            pen_bound += max(0, r[i] - (1-y[i]))**2
            
        # Penalty for overlap
        pen_overlap = 0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                pen_overlap += max(0, r[i] + r[j] - dist)**2
                
        # Objective: Max sum r -> Min -sum r + Penalty
        return -np.sum(r) + 100.0 * (pen_bound + pen_overlap)
        
    # Use L-BFGS-B
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
    for i in range(N):
        bounds.append((0.0, 1.0)) # y
    for i in range(N):
        bounds.append((0.0, 0.5)) # r
        
    try:
        res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-6})
        if res.success or res.status == 0:
            z_opt = res.x
            x_opt = z_opt[0:N]
            y_opt = z_opt[N:2*N]
            r_opt = z_opt[2*N:3*N]
            
            # Check if solution is valid (constraints satisfied within tolerance)
            valid = True
            tol = 1e-5
            for i in range(N):
                if r_opt[i] > x_opt[i] + tol or r_opt[i] > 1-x_opt[i] + tol or \
                   r_opt[i] > y_opt[i] + tol or r_opt[i] > 1-y_opt[i] + tol:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                        if r_opt[i] + r_opt[j] > dist + tol:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                # Clip to be strictly inside if needed? 
                # The evaluator checks "disjoint interiors" and "inside".
                # Usually strict inequality is required or very close.
                # Let's shrink slightly to be safe?
                # Or just return as is if valid.
                return np.column_stack([x_opt, y_opt, r_opt])
            else:
                # If not strictly valid due to numerical issues, try to shrink?
                # For now, discard if not valid.
                return None
        else:
            return None
    except Exception:
        return None
