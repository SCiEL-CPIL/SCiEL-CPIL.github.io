import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    # Objective function to minimize
    # z is flattened [x1, y1, r1, x2, y2, r2, ...]
    def objective(z):
        # Unflatten
        z = z.reshape(N, 3)
        x = z[:, 0]
        y = z[:, 1]
        r = z[:, 2]
        
        # Sum of radii (we minimize negative sum)
        sum_r = np.sum(r)
        
        # Overlap penalty
        # Calculate pairwise distances
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        # We need upper triangle
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i = r[iu[0]]
        r_j = r[iu[1]]
        overlaps = np.maximum(0, r_i + r_j - d_ij)
        penalty_overlap = np.sum(overlaps**2)
        
        # Boundary penalty
        # r_i <= x_i, r_i <= 1-x_i, etc.
        pen_b = 0.0
        pen_b += np.sum(np.maximum(0, r - x)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - x))**2)
        pen_b += np.sum(np.maximum(0, r - y)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - y))**2)
        
        # Total penalty
        penalty = penalty_overlap + pen_b
        
        return -sum_r + 1000.0 * penalty  # Increased penalty weight to enforce validity

    def run_optimization(initial_z):
        # Bounds for each variable
        # x in [0,1], y in [0,1], r in [1e-6, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((1e-6, 0.5)) # r
            
        res = minimize(objective, initial_z, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
        return res

    def validate(z_opt):
        """Check if solution is valid and return sum of radii if so."""
        z = z_opt
        x, y, r = z[:,0], z[:,1], z[:,2]
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i, r_j = r[iu[0]], r[iu[1]]
        max_overlap = np.max(np.maximum(0, r_i + r_j - d_ij))
        
        pen_b = 0.0
        pen_b += np.max(np.maximum(0, r - x))
        pen_b += np.max(np.maximum(0, r - (1 - x)))
        pen_b += np.max(np.maximum(0, r - y))
        pen_b += np.max(np.maximum(0, r - (1 - y)))
        
        if max_overlap < 1e-6 and pen_b < 1e-6:
            return np.sum(r)
        return -np.inf

    inits = []
    
    # Init 1: Hexagonal packing approximation
    # Generate a hexagonal grid and select N points
    def generate_hex_init():
        points = []
        # Approximate side length for hex packing
        # Try a few different scales and offsets
        for scale in [0.15, 0.18, 0.20]:
            for offset_x in [0.0, 0.05, 0.1]:
                for offset_y in [0.0, 0.05, 0.1]:
                    pts = []
                    row = 0
                    while True:
                        y = row * scale * np.sqrt(3) / 2 + offset_y
                        if y > 1.0:
                            break
                        col = 0
                        while True:
                            x = col * scale + offset_x + (row % 2) * scale / 2
                            if x > 1.0:
                                break
                            pts.append((x, y))
                            col += 1
                        row += 1
                    if len(pts) >= N:
                        # Randomly select N points
                        np.random.shuffle(pts)
                        selected = pts[:N]
                        # Assign initial radius
                        r_init = 0.08
                        init = np.zeros((N, 3))
                        for i, (x, y) in enumerate(selected):
                            init[i, 0] = x
                            init[i, 1] = y
                            init[i, 2] = r_init
                        inits.append(init.flatten())
                        if len(inits) >= 10: # Limit hex inits
                            return
                    else:
                        continue
        return

    generate_hex_init()
    
    # Init 2: Random restarts
    for seed in range(20):
        if len(inits) >= 30: # Limit total inits
            break
        np.random.seed(seed)
        rx = np.random.rand(N)
        ry = np.random.rand(N)
        rr = np.full(N, 0.08)
        inits.append(np.column_stack((rx, ry, rr)).flatten())
    
    # Flatten inits and run
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        z0 = np.array(init).flatten()
        try:
            res = run_optimization(z0)
            z_opt = res.x.reshape(N, 3)
            
            # Post-processing: Small local search to fix violations or improve score
            # We can do a few iterations of a simple gradient step with smaller step size
            # Or just validate and update best
            
            score = validate(z_opt)
            if score > best_score:
                best_score = score
                best_solution = z_opt
        except Exception:
            pass

    if best_solution is None:
        # Fallback to trivial seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
