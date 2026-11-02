import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Initialization: Hexagonal-like grid
    # A 5x5 grid is 25 points. We need 26.
    # Let's use a 5x5 grid and place the 26th circle in a "hole" or just use a slightly denser grid.
    # Better: 5 rows of 5, and the 26th one placed in the center of the square? No, that's occupied.
    # Let's use a 5x5 grid for the first 25, and the 26th at (0.5, 0.5) is bad.
    # Let's try a 6x5 grid (30 points) and pick 26? Or just 26 random points with small radii.
    # Random init with small radii is robust.
    
    np.random.seed(42)
    # Spread initial points somewhat
    # 5x5 grid + 1 random
    xs = []
    ys = []
    for i in range(5):
        for j in range(5):
            xs.append((i + 0.5) / 5.0)
            ys.append((j + 0.5) / 5.0)
    # 26th point: random
    xs.append(np.random.uniform(0.2, 0.8))
    ys.append(np.random.uniform(0.2, 0.8))
    
    xs = np.array(xs)
    ys = np.array(ys)
    rs = np.full(N, 0.05) # Small initial radius
    
    # State: [x0, y0, r0, x1, y1, r1, ...]
    state = np.zeros(3 * N)
    state[0::3] = xs
    state[1::3] = ys
    state[2::3] = rs

    def calc_energy(state):
        xs = state[0::3]
        ys = state[1::3]
        rs = state[2::3]
        
        # Penalty for boundary
        pen_bound = 0.0
        for i in range(N):
            min_dist = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
            if rs[i] > min_dist:
                pen_bound += (rs[i] - min_dist)**2
        
        # Penalty for overlap
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    pen_overlap += (min_dist - dist)**2
        
        # Objective: Maximize sum r -> Minimize -sum r
        # Energy = -sum r + lambda * (pen_bound + pen_overlap)
        lambda_pen = 10.0
        return -np.sum(rs) + lambda_pen * (pen_bound + pen_overlap)

    # 2. Simulated Annealing
    T = 1.0
    cooling = 0.999
    step_size_pos = 0.05
    step_size_rad = 0.05
    
    best_state = state.copy()
    best_energy = calc_energy(state)
    current_energy = best_energy
    
    while time.time() < deadline - 20.0: # Leave 20s for refinement
        # Propose move
        new_state = state.copy()
        i = np.random.randint(N)
        
        # Move position
        new_state[i*3] += np.random.uniform(-step_size_pos, step_size_pos)
        new_state[i*3+1] += np.random.uniform(-step_size_pos, step_size_pos)
        
        # Move radius
        new_state[i*3+2] += np.random.uniform(-step_size_rad, step_size_rad)
        
        # Clip to valid ranges (loose)
        new_state[i*3] = np.clip(new_state[i*3], 0, 1)
        new_state[i*3+1] = np.clip(new_state[i*3+1], 0, 1)
        new_state[i*3+2] = np.clip(new_state[i*3+2], 0.01, 0.5)
        
        new_energy = calc_energy(new_state)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            state = new_state
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = state.copy()
        
        T *= cooling
        if T < 1e-4:
            T = 1e-4

    # 3. Refinement with SLSQP
    # Objective: -sum r
    # Constraints:
    # 1. x_i - r_i >= 0
    # 2. 1 - x_i - r_i >= 0
    # 3. y_i - r_i >= 0
    # 4. 1 - y_i - r_i >= 0
    # 5. (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0

    def objective(v):
        rs = v[2::3]
        return -np.sum(rs)

    constraints = []
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3] - v[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i*3] - v[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3+1] - v[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i*3+1] - v[i*3+2]})
        
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i*3]-v[j*3])**2 + (v[i*3+1]-v[j*3+1])**2 - (v[i*3+2]+v[j*3+2])**2
            })

    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0.01, 0.5)) # r

    # SLSQP might be slow or fail. Try it.
    try:
        res = minimize(objective, best_state, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        if res.success and res.fun < best_energy:
            best_state = res.x
            best_energy = res.fun
    except Exception as e:
        pass

    # 4. Final check and cleanup
    xs = best_state[0::3].copy()
    ys = best_state[1::3].copy()
    rs = best_state[2::3].copy()
    
    # Ensure strict feasibility
    # If any constraint is violated, shrink radii slightly
    # This is a simple heuristic: check all constraints, if violated, reduce r_i by small amount
    for _ in range(10):
        valid = True
        # Check boundary
        for i in range(N):
            min_dist = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
            if rs[i] > min_dist:
                rs[i] = min_dist * 0.99
                valid = False
        # Check overlap
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    # Shrink both
                    # Reduce r_i and r_j such that r_i + r_j = dist * 0.99
                    # Keep ratio?
                    total_r = rs[i] + rs[j]
                    if total_r > 0:
                        scale = (dist * 0.99) / total_r
                        rs[i] *= scale
                        rs[j] *= scale
                    valid = False
        if valid:
            break

    result = np.column_stack((xs, ys, rs))
    return result
