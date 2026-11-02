"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26

# EVOLVE-BLOCK-START

def _validate(x, y, r):
    """Check if circles are valid: inside [0,1]^2 and non-overlapping."""
    if np.any(r < 0):
        return False
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    # Check boundary containment
    if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
        return False
    # Check pairwise non-overlap
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-10:
                return False
    return True


def _sum_radii(x, y, r):
    return np.sum(r)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    best_value = 0.0
    best_result = None
    
    # Initial configuration: hexagonal packing in a grid
    def _hex_initial():
        # Try a few different initial arrangements
        configs = []
        
        # Grid arrangement
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-4
                    cells.append([x, y, r])
        configs.append(cells)
        
        # Hexagonal arrangement
        hex_configs = []
        for spacing in [0.18, 0.20, 0.22]:
            hex_cells = []
            row = 0
            while len(hex_cells) < N:
                offset = spacing * np.sqrt(3) / 2 if row % 2 == 1 else 0
                col = 0
                while len(hex_cells) < N:
                    x = spacing * (1 + col) + offset
                    y = spacing * np.sqrt(3) * (row + 0.5)
                    if 0 < x < 1 and 0 < y < 1:
                        r = spacing * 0.49
                        hex_cells.append([x, y, r])
                    col += 1
                row += 1
            if len(hex_cells) >= N:
                configs.append(hex_cells[:N])
        
        return configs
    
    def _optimize_from_initial(x0, y0, r0, time_limit):
        """Optimize starting from initial positions."""
        n = len(x0)
        # Variables: x, y, r for each circle
        x0_flat = np.concatenate([x0, y0, r0])
        
        def objective(vars):
            x = vars[:n]
            y = vars[n:2*n]
            r = vars[2*n:]
            # Penalty for constraint violations
            penalty = 0.0
            # Boundary constraints
            penalty += 100 * np.sum(np.maximum(0, r - x)**2)
            penalty += 100 * np.sum(np.maximum(0, r - y)**2)
            penalty += 100 * np.sum(np.maximum(0, r - (1-x))**2)
            penalty += 100 * np.sum(np.maximum(0, r - (1-y))**2)
            # Non-overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        penalty += 100 * overlap**2
            return -np.sum(r) + penalty
        
        # Bounds: 0 <= x,y <= 1, 0 <= r <= 0.5
        bounds = []
        for _ in range(n):
            bounds.append((0, 1))
        for _ in range(n):
            bounds.append((0, 1))
        for _ in range(n):
            bounds.append((0, 0.5))
        
        result = minimize(objective, x0_flat, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success:
            x = result.x[:n]
            y = result.x[n:2*n]
            r = result.x[2*n:]
            if _validate(x, y, r):
                return x, y, r, _sum_radii(x, y, r)
        
        return None, None, None, 0.0
    
    def _local_search(x, y, r, time_limit):
        """Local search to improve the packing."""
        n = len(x)
        current_value = _sum_radii(x, y, r)
        
        # Try random perturbations
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = current_value
        
        elapsed = time.time() - start_time
        while time.time() - start_time < deadline - 5.0:
            # Random perturbation
            scale = 0.02 * (1 - (time.time() - start_time) / (deadline - start_time))
            dx = np.random.uniform(-scale, scale, n)
            dy = np.random.uniform(-scale, scale, n)
            dr = np.random.uniform(-scale*0.5, scale*0.5, n)
            
            new_x = np.clip(x + dx, 0, 1)
            new_y = np.clip(y + dy, 0, 1)
            new_r = np.clip(r + dr, 0.01, 0.5)
            
            if _validate(new_x, new_y, new_r):
                new_val = _sum_radii(new_x, new_y, new_r)
                if new_val > best_val:
                    best_val = new_val
                    best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()
        
        return best_x, best_y, best_r, best_val
    
    # Try multiple initial configurations
    initial_configs = _hex_initial()
    
    for config in initial_configs:
        if time.time() > deadline - 20.0:
            break
        
        x0 = np.array([c[0] for c in config])
        y0 = np.array([c[1] for c in config])
        r0 = np.array([c[2] for c in config])
        
        # Optimize from this initial configuration
        x, y, r, val = _optimize_from_initial(x0, y0, r0, deadline - time.time() - 10)
        
        if x is not None and val > best_value:
            best_value = val
            best_result = (x, y, r)
        
        # Local search
        if x is not None:
            x2, y2, r2, val2 = _local_search(x, y, r, deadline - time.time())
            if val2 > best_value:
                best_value = val2
                best_result = (x2, y2, r2)
    
    # If we still have time, do a final polish
    if best_result is not None and time.time() < deadline - 10.0:
        x, y, r = best_result
        x, y, r, val = _local_search(x, y, r, deadline - time.time() - 5)
        if val > best_value:
            best_value = val
            best_result = (x, y, r)
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r = best_result
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
