import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_energy_and_grad(z, w, h, beta=1.0, gamma=1.0):
    """
    Compute the objective function for minimization:
    F(z) = -sum(r) + beta * (overlap_penalties) + gamma * (wall_penalties)
    
    z: array of shape (3*N,) [x1, y1, r1, ...]
    """
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    grad = np.zeros_like(z)
    grad[2::3] = -1.0
    
    # Overlap penalties
    # Penalty = sum( max(0, r_i + r_j - d_ij)^2 )
    # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq + 1e-12)
            
            r_sum = r[i] + r[j]
            v = r_sum - d
            
            if v > 0:
                # Penalty v^2
                obj += beta * v * v
                # Gradient of v^2 w.r.t x_i, y_i, r_i, x_j, y_j, r_j
                # d(v^2)/dx_i = 2v * (-dx/d)
                # d(v^2)/dy_i = 2v * (-dy/d)
                # d(v^2)/dr_i = 2v * (1)
                # d(v^2)/dx_j = 2v * (dx/d)
                # d(v^2)/dy_j = 2v * (dy/d)
                # d(v^2)/dr_j = 2v * (1)
                
                coeff = 2 * beta * v / d
                
                grad[3*i + 0] -= coeff * dx
                grad[3*i + 1] -= coeff * dy
                grad[3*i + 2] += 2 * beta * v
                
                grad[3*j + 0] += coeff * dx
                grad[3*j + 1] += coeff * dy
                grad[3*j + 2] += 2 * beta * v
    
    # Wall penalties
    # Penalty = sum( max(0, r_i - x_i)^2 + max(0, r_i - (w-x_i))^2 + ... )
    
    for i in range(N):
        # Left wall: r_i - x_i
        v1 = r[i] - x[i]
        if v1 > 0:
            obj += gamma * v1 * v1
            grad[3*i + 0] -= 2 * gamma * v1
            grad[3*i + 2] += 2 * gamma * v1
            
        # Right wall: r_i - (w - x_i) = r_i + x_i - w
        v2 = r[i] + x[i] - w
        if v2 > 0:
            obj += gamma * v2 * v2
            grad[3*i + 0] += 2 * gamma * v2
            grad[3*i + 2] += 2 * gamma * v2
            
        # Bottom wall: r_i - y_i
        v3 = r[i] - y[i]
        if v3 > 0:
            obj += gamma * v3 * v3
            grad[3*i + 1] -= 2 * gamma * v3
            grad[3*i + 2] += 2 * gamma * v3
            
        # Top wall: r_i - (h - y_i) = r_i + y_i - h
        v4 = r[i] + y[i] - h
        if v4 > 0:
            obj += gamma * v4 * v4
            grad[3*i + 1] += 2 * gamma * v4
            grad[3*i + 2] += 2 * gamma * v4
            
    return obj, grad

def _is_valid(pos, radii, w, h, tol=1e-7):
    """Check validity of the configuration."""
    for i in range(N):
        if pos[i, 0] < -tol or pos[i, 0] > w + tol:
            return False
        if pos[i, 1] < -tol or pos[i, 1] > h + tol:
            return False
        if radii[i] < -tol:
            return False
        if radii[i] > pos[i, 0] + tol:
            return False
        if radii[i] > w - pos[i, 0] + tol:
            return False
        if radii[i] > pos[i, 1] + tol:
            return False
        if radii[i] > h - pos[i, 1] + tol:
            return False
            
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _simulated_annealing(w, h, start_time, deadline, initial_z):
    """
    Run Simulated Annealing to maximize sum of radii.
    We minimize E = -sum(r) + penalty.
    """
    z = initial_z.copy()
    T_start = 0.1
    T_end = 1e-6
    cooling = 0.999
    
    # Bounds for z
    r_max = min(w, h) / 2.0
    
    best_z = z.copy()
    best_score = np.sum(z[2::3])
    
    # Precompute indices for speed? N=21 is small, O(N^2) is fine.
    
    current_energy, _ = _compute_energy_and_grad(z, w, h, beta=10.0, gamma=10.0)
    
    T = T_start
    while T > T_end:
        if time.time() > deadline - 10:
            break
            
        # Random move: perturb one circle
        idx = np.random.randint(N)
        step_size = T * 0.5 + 0.01
        
        new_z = z.copy()
        # Perturb position
        new_z[3*idx] += np.random.normal(0, step_size)
        new_z[3*idx + 1] += np.random.normal(0, step_size)
        # Perturb radius
        new_z[3*idx + 2] += np.random.normal(0, step_size * 0.5)
        
        # Enforce bounds roughly to keep values sane
        new_z[3*idx] = np.clip(new_z[3*idx], 0.01, w - 0.01)
        new_z[3*idx + 1] = np.clip(new_z[3*idx + 1], 0.01, h - 0.01)
        new_z[3*idx + 2] = np.clip(new_z[3*idx + 2], 0.01, r_max)
        
        new_energy, _ = _compute_energy_and_grad(new_z, w, h, beta=10.0, gamma=10.0)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            z = new_z
            current_energy = new_energy
            
            score = np.sum(z[2::3])
            if score > best_score:
                best_score = score
                best_z = z.copy()
        
        T *= cooling
        
    return best_z

def _local_optimize(z0, w, h, deadline):
    """
    Perform local optimization using L-BFGS-B to polish the solution.
    """
    if time.time() > deadline - 5:
        return z0
        
    r_max = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.0, r_max))
        
    try:
        # Use high penalty weights to ensure validity
        res = minimize(_compute_energy_and_grad, z0, args=(w, h, 100.0, 100.0), 
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        return res.x
    except Exception:
        return z0

def _pack_for_aspect(w, h, start_time, deadline, num_sa_trials=5):
    """
    Pack circles in w x h rectangle.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    r_max = min(w, h) / 2.0
    
    for trial in range(num_sa_trials):
        if time.time() > deadline - 15:
            break
            
        # Initialize
        if trial == 0:
            # Grid init
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            if cols * rows < N:
                cols += 1
                rows = int(np.ceil(N / cols))
            
            z0 = np.zeros(3 * N)
            for i in range(N):
                col = i % cols
                row = i // cols
                z0[3*i + 0] = (col + 0.5) * w / cols
                z0[3*i + 1] = (row + 0.5) * h / rows
                cell_w = w / cols
                cell_h = h / rows
                z0[3*i + 2] = min(cell_w, cell_h) / 2.0 * 0.9
        else:
            # Random init
            z0 = np.zeros(3 * N)
            z0[0::3] = np.random.uniform(0.1, w - 0.1, N)
            z0[1::3] = np.random.uniform(0.1, h - 0.1, N)
            z0[2::3] = np.random.uniform(0.05, 0.2 * min(w, h), N)
            
        # Simulated Annealing
        z_sa = _simulated_annealing(w, h, start_time, deadline, z0)
        
        # Local Optimization
        z_opt = _local_optimize(z_sa, w, h, deadline)
        
        pos = np.column_stack((z_opt[0::3], z_opt[1::3]))
        radii = z_opt[2::3]
        
        if _is_valid(pos, radii, w, h):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    np.random.seed(42)
    
    best_global_score = 0
    best_global_result = None
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.45, 2.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # Reduce trials for extreme aspect ratios
        if ar in [1.0, 0.9, 1.1, 0.8, 1.25]:
            num_trials = 8
        else:
            num_trials = 4
            
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline, num_sa_trials=num_trials)
        
        if pos is None:
            continue
        
        if score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
    
    if best_global_result is None:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side * 0.8
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val
        best_global_result = np.column_stack([pos, rad])
    
    return best_global_result
