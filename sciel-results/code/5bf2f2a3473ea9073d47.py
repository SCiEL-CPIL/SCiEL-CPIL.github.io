import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # 1. Initialization
    # Use a hexagonal-like grid for better initial packing density
    # 5x5 grid is 25. We need 26. 
    # Let's use a 5x5 grid and one extra, or just 26 random points in a grid structure.
    # Better: Generate a hexagonal lattice of ~30 points, pick 26.
    
    def init_hex():
        points = []
        # Generate hex points
        # Radius in hex units
        # Let's just use a jittered grid for simplicity and robustness
        # 5x5 grid = 25 points. 
        # We can use a 5x5 grid and add one more in the center of a cell? No, that's messy.
        # Let's use a 5x5 grid for the first 25, and the 26th near one of them?
        # Or just 26 points in a 5x5 grid with some overlap allowed initially?
        
        # Actually, a simple 5x5 grid with radius 0.1 is a valid start.
        # SA will improve it.
        xs = []
        ys = []
        rs = []
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                xs.append(x)
                ys.append(y)
                rs.append(0.1)
                idx += 1
        # If N < 25, this works. If N > 25, we need more.
        # N=26.
        # Add one more point.
        # Place it at (0.5, 0.5) with small radius? No, that's occupied.
        # Let's just use a 5x5 grid and one extra at (0.1, 0.1) with small radius?
        # Or better: Use a 6x5 grid and pick 26?
        
        # Let's try a different init: Random points in [0.1, 0.9] with small radius 0.05.
        # This avoids initial overlaps if we check, but random might overlap.
        # Let's stick to the grid but handle N=26 specifically.
        
        # 5x5 grid is 25.
        # Let's use 25 grid points and 1 point at (0.5, 0.5) is bad.
        # Let's use a 5x5 grid, and the 26th point at (0.5, 0.05) ?
        
        # Actually, let's just use 26 random points with small radius.
        # SA is robust to initialization.
        
        np.random.seed(42)
        xs = np.random.uniform(0.1, 0.9, N)
        ys = np.random.uniform(0.1, 0.9, N)
        rs = np.full(N, 0.05)
        return np.array([xs, ys, rs]).T

    # Better Init: Hexagonal
    def init_hexagonal():
        # Generate hex points
        pts = []
        # 4-5-5-4-4? 
        # Let's just generate a dense hex grid and sample 26
        # Hex grid spacing 0.2
        # x = i * 0.2, y = j * 0.2 * sqrt(3) + (i % 2) * 0.2 * sqrt(3) / 2
        # Scale to fit [0,1]
        
        # Simpler: 5x5 grid is fine. The 26th circle can start small.
        # Let's use the 5x5 grid for 25, and the 26th at (0.5, 0.5) with r=0.01?
        # No, (0.5,0.5) is the center of the grid.
        # Let's use 26 random points.
        
        pass

    # Let's use a "Jittered Grid" init
    # 5x5 grid, add jitter
    # For N=26, use 5x5 grid (25 pts) + 1 pt at (0.1, 0.9) with small r?
    # Or just 26 pts in a 5x5 grid with some rows having 6?
    
    # Let's use a generic random init with small radii.
    # It's safer and SA will expand them.
    
    def get_initial():
        np.random.seed(123)
        # Spread them out
        # 5x5 grid
        grid_size = 5
        xs = []
        ys = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(xs) < N:
                    xs.append((i + 0.5) / grid_size + np.random.uniform(-0.02, 0.02))
                    ys.append((j + 0.5) / grid_size + np.random.uniform(-0.02, 0.02))
        # We have 25. Need 1 more.
        # Add one at a random location, small radius
        xs.append(np.random.uniform(0.1, 0.9))
        ys.append(np.random.uniform(0.1, 0.9))
        
        xs = np.array(xs)
        ys = np.array(ys)
        rs = np.full(N, 0.05)
        return xs, ys, rs

    xs, ys, rs = get_initial()
    state = np.concatenate([xs, ys, rs])
    
    def calc_energy(state):
        xs, ys, rs = state[0:N], state[N:2*N], state[2*N:3*N]
        
        # Penalty for boundary
        pen_bound = 0
        for i in range(N):
            min_dist = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
            if rs[i] > min_dist:
                pen_bound += (rs[i] - min_dist)**2
        
        # Penalty for overlap
        pen_overlap = 0
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

    # SA
    T = 1.0
    cooling = 0.999
    step_size_pos = 0.05
    step_size_rad = 0.05
    
    best_state = state.copy()
    best_energy = calc_energy(state)
    current_energy = best_energy
    
    while time.time() < deadline - 10.0:
        # Propose move
        new_state = state.copy()
        i = np.random.randint(N)
        
        # Move position
        new_state[i] += np.random.uniform(-step_size_pos, step_size_pos)
        new_state[N+i] += np.random.uniform(-step_size_pos, step_size_pos)
        
        # Move radius
        new_state[2*N+i] += np.random.uniform(-step_size_rad, step_size_rad)
        
        # Clip to valid ranges (loose)
        new_state[i] = np.clip(new_state[i], 0, 1)
        new_state[N+i] = np.clip(new_state[N+i], 0, 1)
        new_state[2*N+i] = np.clip(new_state[2*N+i], 0.01, 0.5)
        
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

    # Refine with SLSQP
    # Objective: -sum r
    # Constraints:
    # 1. x_i - r_i >= 0
    # 2. 1 - x_i - r_i >= 0
    # 3. y_i - r_i >= 0
    # 4. 1 - y_i - r_i >= 0
    # 5. (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0

    def objective(v):
        rs = v[2*N:3*N]
        return -np.sum(rs)

    constraints = []
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[N+i] - v[2*N+i]})
        
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })

    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0.01, 0.5)) # r

    # SLSQP might be slow or fail. Try it.
    try:
        res = minimize(objective, best_state, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        if res.success and res.fun < best_energy:
            best_state = res.x
            best_energy = res.fun
    except Exception as e:
        pass

    # Final check and cleanup
    xs, ys, rs = best_state[0:N], best_state[N:2*N], best_state[2*N:3*N]
    
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
                    scale = (dist * 0.99) / total_r
                    rs[i] *= scale
                    rs[j] *= scale
                    valid = False
        if valid:
            break

    result = np.column_stack((xs, ys, rs))
    return result
