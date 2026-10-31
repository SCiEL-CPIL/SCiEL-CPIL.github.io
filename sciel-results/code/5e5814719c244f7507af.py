"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Objective: Minimize negative sum of radii
    def objective(v):
        # v is shape (3*N,)
        # Radii are at indices 2, 5, 8, ...
        radii = v[2::3]
        return -np.sum(radii)

    # Constraints: All must be >= 0
    def constraints_fun(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Boundary constraints
        c_boundaries = np.concatenate([
            x - r,       # x >= r
            y - r,       # y >= r
            1 - x - r,   # x <= 1 - r
            1 - y - r    # y <= 1 - r
        ])
        
        # Pairwise constraints
        # Calculate squared distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        
        # Calculate squared sum of radii
        dr = r[:, None] + r[None, :]
        rad_sum_sq = dr**2
        
        # Constraint: dist_sq >= rad_sum_sq  =>  dist_sq - rad_sum_sq >= 0
        # Mask out diagonal (self) and lower triangle (duplicates)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_pairs = (dist_sq - rad_sum_sq)[mask]
        
        return np.concatenate([c_boundaries, c_pairs])

    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N

    def get_hex_guess():
        # Generate a hexagonal grid
        # Spacing s
        # Rows: y = j * s * sqrt(3)/2
        # Cols: x = i * s + (j % 2) * s/2
        
        # We want to pack 26 circles.
        # Let's try to generate a large enough grid and pick the best 26.
        max_points = 100
        s = 0.2  # Initial spacing
        points = []
        for j in range(10):
            for i in range(10):
                x = i * s + (j % 2) * s / 2.0
                y = j * s * np.sqrt(3) / 2.0
                points.append((x, y))
        
        points = np.array(points)
        
        # Center the grid
        center = np.mean(points, axis=0)
        points = points - center
        
        # Scale to fit in [0.1, 0.9] roughly
        max_coord = np.max(np.abs(points))
        scale = 0.4 / max_coord if max_coord > 0 else 1.0
        points = points * scale
        
        # Translate to center of square (0.5, 0.5)
        points = points + np.array([0.5, 0.5])
        
        # Sort by distance from center to pick the "densest" central part
        dists = np.linalg.norm(points - np.array([0.5, 0.5]), axis=1)
        indices = np.argsort(dists)
        selected = points[indices[:N]]
        
        # Calculate initial radii
        # Find min distance between any pair
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(selected[i] - selected[j])
                if d < min_dist:
                    min_dist = d
        
        # Set radius to 0.4 * min_dist to ensure no initial overlap
        r0 = 0.4 * min_dist if min_dist > 0 else 0.05
        r0 = min(r0, 0.2) # Cap it
        
        v = np.zeros(3 * N)
        for i in range(N):
            v[3*i] = selected[i][0]
            v[3*i+1] = selected[i][1]
            v[3*i+2] = r0
            
        return v

    def get_random_guess():
        # Random valid configuration
        # Start with small radii
        r0 = 0.05
        points = np.zeros((N, 2))
        for i in range(N):
            # Rejection sampling
            for _ in range(100):
                x = np.random.uniform(r0, 1 - r0)
                y = np.random.uniform(r0, 1 - r0)
                points[i] = [x, y]
                
                # Check against previous points
                valid = True
                for j in range(i):
                    d = np.linalg.norm(points[i] - points[j])
                    if d < 2 * r0:
                        valid = False
                        break
                if valid:
                    break
            
            # If still invalid, place randomly (might overlap, optimizer will fix)
        
        v = np.zeros(3 * N)
        for i in range(N):
            v[3*i] = points[i][0]
            v[3*i+1] = points[i][1]
            v[3*i+2] = r0
        return v

    best_val = -np.inf
    best_result = None
    
    # Generate initial guesses
    guesses = [get_hex_guess()]
    
    # Add a few random guesses
    np.random.seed(42)
    for _ in range(10):
        guesses.append(get_random_guess())
        
    # Also add a grid guess
    side = int(np.ceil(np.sqrt(N)))
    grid_points = []
    for i in range(side):
        for j in range(side):
            if len(grid_points) < N:
                grid_points.append(((i + 0.5) / side, (j + 0.5) / side))
    grid_v = np.zeros(3 * N)
    for i in range(N):
        grid_v[3*i] = grid_points[i][0]
        grid_v[3*i+1] = grid_points[i][1]
        grid_v[3*i+2] = 0.05
    guesses.append(grid_v)

    # Shuffle guesses to ensure we try diverse starts early
    np.random.shuffle(guesses)

    # Run optimization
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective, 
                guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': constraints_fun},
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.success:
                val = -res.fun
                # Verify validity roughly
                v = res.x
                x, y, r = v[0::3], v[1::3], v[2::3]
                # Check if any constraint is significantly violated
                c = constraints_fun(v)
                if np.min(c) > -1e-5: # Tolerance for numerical error
                    if val > best_val:
                        best_val = val
                        best_result = v
        except Exception:
            pass

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Ensure return shape is (N, 3)
    return best_result.reshape(N, 3)

# EVOLVE-BLOCK-END
