"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 150.0  # wall-clock limit
    safety_margin = 5.0
    
    best_result = None
    best_value = -np.inf
    
    def get_value(circles):
        """Sum of radii, with penalty for violations."""
        n = len(circles)
        radii = circles[:, 2]
        return np.sum(radii)
    
    def check_valid(circles):
        """Check if all circles are valid (non-overlapping, inside unit square)."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
            for j in range(i + 1, n):
                xj, yj, rj = circles[j]
                dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                if dist < r + rj - 1e-9:
                    return False
        return True
    
    def penalty(circles, beta=50.0):
        """Smooth penalty for constraint violations."""
        n = len(circles)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary penalty: circles must stay in [0,1]^2
        boundary_pen = 0.0
        for i in range(n):
            if x[i] - r[i] < 0:
                boundary_pen += (x[i] - r[i])**2
            if x[i] + r[i] > 1:
                boundary_pen += (x[i] + r[i] - 1)**2
            if y[i] - r[i] < 0:
                boundary_pen += (y[i] - r[i])**2
            if y[i] + r[i] > 1:
                boundary_pen += (y[i] + r[i] - 1)**2
        
        # Overlap penalty: smooth approximation using log-sum-exp
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                # If dist < min_dist, penalty
                violation = min_dist - np.sqrt(dist_sq + 1e-12)
                if violation > 0:
                    overlap_pen += violation**2
        
        return boundary_pen + overlap_pen
    
    def objective(params):
        """Objective to minimize: negative sum of radii + penalty."""
        n = len(params) // 3
        circles = params.reshape(n, 3)
        # Ensure radii are positive
        circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
        val = -np.sum(circles[:, 2])
        pen = penalty(circles)
        return val + 100.0 * pen
    
    def generate_initial_config(method='grid', seed=None):
        """Generate an initial configuration."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        if method == 'grid':
            # Grid with some randomization
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                    y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
            # Shuffle and take N
            rng.shuffle(cells)
            return np.array(cells[:N])
        
        elif method == 'hex':
            # Hexagonal packing
            circles = []
            r = 0.08
            dx = 2 * r
            dy = dx * np.sqrt(3) / 2
            row = 0
            while len(circles) < N:
                col = 0
                while len(circles) < N:
                    x = r + col * dx + (row % 2) * dx / 2
                    y = r + row * dy
                    if x < 1 - r and y < 1 - r:
                        circles.append([x, y, r])
                    col += 1
                row += 1
            return np.array(circles[:N])
        
        elif method == 'random':
            # Random placement with small radii
            circles = []
            for _ in range(N):
                r = rng.uniform(0.03, 0.08)
                x = rng.uniform(r, 1 - r)
                y = rng.uniform(r, 1 - r)
                circles.append([x, y, r])
            return np.array(circles)
        
        else:
            # Fallback: grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    
    methods = ['grid', 'hex', 'random']
    seeds = [42, 123, 456, 789, 1000, 2000, 3000]
    
    for method in methods:
        for seed in seeds:
            if time.time() - start_time > deadline - safety_margin:
                break
            
            init_circles = generate_initial_config(method, seed)
            init_params = init_circles.flatten()
            
            # Bounds: x, y in [0,1], r in [0.001, 0.5]
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))  # x
                bounds.append((0.0, 1.0))  # y
                bounds.append((0.001, 0.5))  # r
            
            try:
                result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_value:
                    final_circles = result.x.reshape(N, 3)
                    final_circles[:, 2] = np.maximum(final_circles[:, 2], 1e-6)
                    
                    # Verify validity
                    if check_valid(final_circles):
                        val = get_value(final_circles)
                        if val > best_value:
                            best_value = val
                            best_result = final_circles.copy()
                            
            except Exception:
                pass
        
        if time.time() - start_time > deadline - safety_margin:
            break
    
    # If we found a valid result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid grid configuration
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = min(0.5 / side - 0.005, 0.5 - max(x - 0.5, 0.5 - x, y - 0.5, 0.5 - y))
            r = max(r, 0.01)
            cells.append([x, y, r])
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
