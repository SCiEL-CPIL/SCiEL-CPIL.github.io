import numpy as np
import time
import math

N = 26
DEADLINE = 145.0

def _md_step(positions, radii, velocities, dt, temp, growth_rate):
    """
    Perform one step of Langevin Molecular Dynamics for circle packing.
    
    Parameters:
    positions: (N, 2) array of centers
    radii: (N,) array of radii
    velocities: (N, 3) array of velocities (vx, vy, vr)
    dt: time step
    temp: temperature (damping/noise)
    growth_rate: force pushing radii to grow
    
    Returns:
    updated positions, radii, velocities
    """
    N = len(positions)
    forces = np.zeros((N, 3))
    
    # 1. Pairwise Repulsion
    # We use a stiff spring-like force for overlaps to ensure stability.
    # F = k * overlap * direction
    # We also add a term to push them apart if they are very close to prevent singularity.
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            
            min_dist = radii[i] + radii[j]
            overlap = min_dist - dist
            
            if overlap > 0:
                # Force magnitude
                # Using a quadratic potential derivative: F = 2 * k * overlap
                # k is stiffness. We scale it to be strong enough to hold positions.
                k = 100.0
                f_mag = 2.0 * k * overlap
                
                # Direction (unit vector from j to i)
                ux = dx / dist
                uy = dy / dist
                
                forces[i, 0] += f_mag * ux
                forces[i, 1] += f_mag * uy
                forces[j, 0] -= f_mag * ux
                forces[j, 1] -= f_mag * uy
                
    # 2. Wall Repulsion & Growth Force
    for i in range(N):
        r = radii[i]
        x, y = positions[i]
        
        # Wall forces (push inward if outside)
        # Left wall: x < r
        if x < r:
            forces[i, 0] += 2.0 * 100.0 * (r - x)
        # Right wall: x > 1 - r
        if x > 1.0 - r:
            forces[i, 0] -= 2.0 * 100.0 * (x - (1.0 - r))
        # Bottom wall: y < r
        if y < r:
            forces[i, 1] += 2.0 * 100.0 * (r - y)
        # Top wall: y > 1 - r
        if y > 1.0 - r:
            forces[i, 1] -= 2.0 * 100.0 * (y - (1.0 - r))
        
        # Growth Force: Constant push to increase radius
        # This is the "pressure" term.
        forces[i, 2] += growth_rate
        
    # 3. Update Velocities (Langevin Dynamics)
    # m * dv = F * dt - gamma * v * dt + sqrt(2 * gamma * T * dt) * noise
    # Assume mass = 1 for simplicity, or scale forces accordingly.
    # Let's use mass = 1.
    gamma = 1.0 # Friction coefficient
    
    # Brownian noise
    noise_pos = np.random.normal(0, 1, size=(N, 2))
    noise_rad = np.random.normal(0, 1, size=N)
    
    # Velocity update
    # v_new = v + (F * dt) - (gamma * v * dt) + noise
    # Note: gamma * dt should be small for stability, typically < 1.
    # We scale temperature to control the noise magnitude.
    noise_scale = math.sqrt(2.0 * gamma * temp * dt)
    
    velocities[:, :2] += forces[:, :2] * dt - gamma * velocities[:, :2] * dt
    velocities[:, 2] += forces[:, 2] * dt - gamma * velocities[:, 2] * dt
    
    velocities[:, :2] += noise_scale * noise_pos
    velocities[:, 2] += noise_scale * noise_rad
    
    # 4. Update Positions and Radii
    positions += velocities[:, :2] * dt
    radii += velocities[:, 2] * dt
    
    # 5. Constrain Radii
    # Radii must be positive and not exceed 0.5 (max possible in unit square)
    radii = np.clip(radii, 1e-6, 0.5)
    
    # If a radius hits the bound, zero its velocity in that direction to prevent jitter
    for i in range(N):
        if radii[i] <= 1e-6:
            velocities[i, 2] = 0
        elif radii[i] >= 0.5:
            velocities[i, 2] = 0
            
    return positions, radii, velocities

def _simulate_packing(seed, time_limit=10.0):
    """
    Run an MD simulation to find a good packing.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialization: Hexagonal grid is a good start
    # Generate a hex grid of points
    pts = []
    for i in range(10):
        for j in range(10):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            pts.append((x, y))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    max_coord = np.max(np.abs(pts))
    scale = 0.45 / max_coord
    pts *= scale
    pts += 0.5
    
    # Select N points closest to center (or random subset)
    dists = np.sum((pts - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    positions = pts[indices].copy()
    # Add small random perturbation to break symmetry
    positions += rng.uniform(-0.01, 0.01, size=(N, 2))
    positions = np.clip(positions, 0.01, 0.99)
    
    # Initial radii
    radii = np.full(N, 0.04)
    radii += rng.uniform(-0.005, 0.005, size=N)
    radii = np.clip(radii, 0.01, 0.1)
    
    velocities = np.zeros((N, 3))
    
    # Simulation parameters
    dt = 0.002
    # Start with high temperature to explore, then cool down
    initial_temp = 0.05
    final_temp = 0.0001
    initial_growth = 0.1
    final_growth = 0.01
    
    best_score = -np.inf
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    total_steps = 20000
    steps_per_check = 100
    
    for step in range(total_steps):
        if time.time() > start_time + time_limit:
            break
            
        # Annealing schedule
        progress = step / total_steps
        # Exponential cooling
        temp = initial_temp * (final_temp / initial_temp) ** progress
        # Growth rate decreases slightly as we pack tighter, or stays constant?
        # Keeping it constant or slightly decreasing helps find the maximum.
        growth = initial_growth * (0.5 + 0.5 * (1 - progress))
        
        positions, radii, velocities = _md_step(positions, radii, velocities, dt, temp, growth)
        
        # Check for validity and score periodically
        if step % steps_per_check == 0:
            # Check if circles are inside [0,1]^2
            # x, y, r
            # Valid if: r <= x <= 1-r and r <= y <= 1-r
            # And pairwise distances >= r_i + r_j
            
            # Fast validity check
            valid = True
            for i in range(N):
                if positions[i, 0] < radii[i] or positions[i, 0] > 1 - radii[i] or \
                   positions[i, 1] < radii[i] or positions[i, 1] > 1 - radii[i]:
                    valid = False
                    break
            
            if valid:
                # Check pairwise overlaps
                for i in range(N):
                    if not valid: break
                    for j in range(i + 1, N):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (radii[i] + radii[j])**2
                        if dist_sq < min_dist_sq - 1e-9: # Small tolerance
                            valid = False
                            break
                
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_positions = positions.copy()
                    best_radii = radii.copy()
                    
    # Final result
    result = np.zeros((N, 3))
    result[:, :2] = best_positions
    result[:, 2] = best_radii
    return result

def _refine_with_slsqp(p, time_limit=5.0):
    """
    Use SLSQP to polish the result from MD.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    # Flatten
    x0 = p.flatten()
    
    def objective(params):
        p = params.reshape(N, 3)
        return -np.sum(p[:, 2])
        
    def objective_grad(params):
        g = np.zeros_like(params)
        g[2::3] = -1.0
        return g
        
    cons = []
    
    # Wall constraints
    for i in range(N):
        # x >= r
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        # x <= 1 - r
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        # y >= r
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        # y <= 1 - r
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
        
    # Pair constraints
    iu = np.triu_indices(N, k=1)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        def pair_fun(p, i=i, j=j):
            dx = p[3*i] - p[3*j]
            dy = p[3*i+1] - p[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - p[3*i+2] - p[3*j+2]
        cons.append({'type': 'ineq', 'fun': pair_fun})
        
    lo = np.zeros(3*N)
    hi = np.zeros(3*N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
        
    bounds = list(zip(lo, hi))
    
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad,
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 50, 'ftol': 1e-12})
        if res.success:
            p_new = res.x.reshape(N, 3)
            # Verify validity
            valid = True
            x, y, r = p_new[:, 0], p_new[:, 1], p_new[:, 2]
            if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9) or \
               np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
                valid = False
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                    if not valid: break
            if valid:
                return p_new
    except Exception:
        pass
    return p

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_params = None
    best_score = -np.inf
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        # Run MD simulation
        p_md = _simulate_packing(seed, time_limit=5.0)
        score_md = np.sum(p_md[:, 2])
        
        if score_md > best_score:
            best_score = score_md
            best_params = p_md.copy()
            
        # Polish with SLSQP if time permits
        if time.time() < deadline - 15:
            p_refined = _refine_with_slsqp(best_params, time_limit=4.0)
            score_refined = np.sum(p_refined[:, 2])
            if score_refined > best_score:
                best_score = score_refined
                best_params = p_refined.copy()
                
    if best_params is None:
        # Fallback: Grid
        p = np.zeros((N, 3))
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            x = (i % side + 0.5) / side
            y = (i // side + 0.5) / side
            p[i] = [x, y, 0.04]
        best_params = p
        
    return best_params

