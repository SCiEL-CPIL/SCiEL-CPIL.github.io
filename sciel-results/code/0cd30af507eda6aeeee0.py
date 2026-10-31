"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                positions.append([x, y])
                radii.append(r)
    
    positions = np.array(positions)
    radii = np.array(radii)
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    n = N
    def unpack(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        return x, y, r
    
    def pack(x, y, r):
        return np.concatenate([x, y, r])
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    def objective(v):
        x, y, r = unpack(v)
        return -np.sum(r)
    
    # Constraints: 
    # 1. Boundary: 0 <= x <= 1-r, 0 <= y <= 1-r
    # 2. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    
    # We'll use inequality constraints in the form g(v) >= 0
    
    # Boundary constraints
    def boundary_constraints(v):
        x, y, r = unpack(v)
        cons = []
        # x >= r
        cons.append(x - r)
        # 1-x >= r
        cons.append(1 - x - r)
        # y >= r
        cons.append(y - r)
        # 1-y >= r
        cons.append(1 - y - r)
        # r > 0 (small lower bound)
        cons.append(r - 1e-6)
        return cons
    
    # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    def overlap_constraints(v):
        x, y, r = unpack(v)
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                cons.append(dx*dx + dy*dy - dr*dr)
        return cons
    
    # Combine all constraints
    def all_constraints(v):
        cons = []
        # Boundary
        x, y, r = unpack(v)
        cons.extend(x - r)
        cons.extend(1 - x - r)
        cons.extend(y - r)
        cons.extend(1 - y - r)
        cons.extend(r - 1e-6)
        # Overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                cons.append(dx*dx + dy*dy - dr*dr)
        return cons
    
    # Initial point
    v0 = pack(positions, positions[:, 1] if positions.shape[1] > 1 else positions, radii)
    # Wait, positions is already 2D array with shape (N, 2)
    v0 = pack(positions[:, 0], positions[:, 1], radii)
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0))  # x
    for i in range(n):
        bounds.append((1e-6, 1.0))  # y
    for i in range(n):
        bounds.append((1e-6, 0.5))  # r
    
    best_v = v0.copy()
    best_obj = objective(v0)
    
    # Try multiple restarts with different initial configurations
    attempts = 0
    max_attempts = 10
    
    while time.time() < deadline:
        attempts += 1
        
        # Generate initial guess
        if attempts == 1:
            # Grid
            side = int(np.ceil(np.sqrt(N)))
            positions = []
            radii = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        r = 0.5 / side - 1e-4
                        positions.append([x, y])
                        radii.append(r)
            positions = np.array(positions)
            radii = np.array(radii)
        elif attempts == 2:
            # Random perturbation of grid
            side = int(np.ceil(np.sqrt(N)))
            positions = []
            radii = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side + np.random.uniform(-0.02, 0.02)
                        y = (j + 0.5) / side + np.random.uniform(-0.02, 0.02)
                        x = np.clip(x, 0.02, 0.98)
                        y = np.clip(y, 0.02, 0.98)
                        r = 0.5 / side - 1e-4
                        positions.append([x, y])
                        radii.append(r)
            positions = np.array(positions)
            radii = np.array(radii)
        else:
            # Random initial positions with small radii
            positions = np.random.uniform(0.05, 0.95, size=(N, 2))
            radii = np.full(N, 0.02)
        
        v0 = pack(positions[:, 0], positions[:, 1], radii)
        
        # Check if initial point is roughly feasible
        x, y, r = unpack(v0)
        feasible = True
        for i in range(n):
            if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                feasible = False
                break
        if feasible:
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    if dx*dx + dy*dy < dr*dr - 1e-6:
                        feasible = False
                        break
                if not feasible:
                    break
        
        if not feasible:
            # Make it feasible by reducing radii
            for i in range(n):
                v0[2*n+i] *= 0.5
            x, y, r = unpack(v0)
            # Check again
            feasible = True
            for i in range(n):
                if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                    feasible = False
                    break
            if feasible:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dr = r[i] + r[j]
                        if dx*dx + dy*dy < dr*dr - 1e-6:
                            feasible = False
                            break
                    if not feasible:
                        break
            if not feasible:
                continue
        
        # Optimize
        try:
            result = minimize(
                objective, 
                v0, 
                method='SLSQP',
                bounds=bounds,
                constraints=all_constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_obj:
                # Verify feasibility of result
                x, y, r = unpack(result.x)
                feasible = True
                for i in range(n):
                    if x[i] < r[i] - 1e-5 or x[i] > 1 - r[i] + 1e-5 or y[i] < r[i] - 1e-5 or y[i] > 1 - r[i] + 1e-5:
                        feasible = False
                        break
                if feasible:
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dr = r[i] + r[j]
                            if dx*dx + dy*dy < dr*dr - 1e-5:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible:
                    best_v = result.x.copy()
                    best_obj = result.fun
        except Exception as e:
            pass
        
        if time.time() > deadline - 10:
            break
    
    # Final result
    x, y, r = unpack(best_v)
    
    # Ensure strict feasibility with small margin
    margin = 1e-4
    for i in range(n):
        x[i] = np.clip(x[i], r[i] + margin, 1 - r[i] - margin)
        y[i] = np.clip(y[i], r[i] + margin, 1 - r[i] - margin)
        r[i] = max(r[i], 1e-6)
    
    # Check and fix overlaps by reducing radii if needed
    for iteration in range(100):
        overlap_found = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                if dist < sum_r + 1e-6:
                    # Reduce both radii slightly
                    reduction = (sum_r - dist + 1e-6) / 2 + 1e-5
                    r[i] = max(r[i] - reduction, 1e-6)
                    r[j] = max(r[j] - reduction, 1e-6)
                    overlap_found = True
        if not overlap_found:
            break
    
    # Final boundary check
    for i in range(n):
        r[i] = min(r[i], x[i] - margin, 1 - x[i] - margin, y[i] - margin, 1 - y[i] - margin)
        r[i] = max(r[i], 1e-6)
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
