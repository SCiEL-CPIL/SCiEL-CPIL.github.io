"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _slsqp_objective(params):
    """
    Objective function for SLSQP: minimize -sum(radii)
    """
    r = params[2::3]
    return -np.sum(r)

def _slsqp_constraints(params):
    """
    Generate constraint functions for SLSQP.
    Constraints are of the form: g(params) >= 0
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    cons = []
    
    # Boundary constraints:
    # x - r >= 0
    # x + r <= 1  => 1 - x - r >= 0
    # y - r >= 0
    # y + r <= 1  => 1 - y - r >= 0
    # r >= 0
    
    for i in range(N):
        # Left wall
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2]
        })
        # Right wall
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]
        })
        # Bottom wall
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]
        })
        # Top wall
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]
        })
        # Radius non-negative
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+2]
        })
        
    # Non-overlap constraints:
    # dist(i,j) - r_i - r_j >= 0
    # Note: dist is non-smooth at 0, but SLSQP can often handle it if we avoid 0.
    # To be safe, we can use a smooth approximation or just rely on the solver's robustness.
    # A better approach for SLSQP is to use squared distance:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0 is NOT equivalent to dist >= r_i+r_j
    # because dist >= r_i+r_j is equivalent to dist^2 >= (r_i+r_j)^2 ONLY IF dist, r_i+r_j >= 0.
    # Since r_i, r_j >= 0, r_i+r_j >= 0. And dist >= 0.
    # So dist >= r_i+r_j <=> dist^2 >= (r_i+r_j)^2.
    # This formulation is smooth everywhere!
    
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2
            })
            
    return cons

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_config_force_directed(seed):
    """Generate initial config using a simple force-directed relaxation."""
    np.random.seed(seed)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.full(N, 0.1)
    
    # Simple relaxation: push apart overlaps
    for _ in range(100):
        moved = False
        for i in range(N):
            fx, fy = 0, 0
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] and dist > 1e-6:
                    # Repulsive force
                    f = (r[i] + r[j] - dist) / dist
                    fx += f * dx
                    fy += f * dy
            # Boundary forces
            if x[i] < r[i]: fx += (r[i] - x[i]) * 10
            if x[i] > 1 - r[i]: fx -= (x[i] - (1 - r[i])) * 10
            if y[i] < r[i]: fy += (r[i] - y[i]) * 10
            if y[i] > 1 - r[i]: fy -= (y[i] - (1 - r[i])) * 10
            
            x[i] += 0.01 * fx
            y[i] += 0.01 * fy
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            if np.abs(fx) > 1e-4 or np.abs(fy) > 1e-4:
                moved = True
        if not moved:
            break
            
    # Clip to valid bounds
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
        
    return np.column_stack([x, y, r])

def _optimize_slsqp(xyz, time_budget=30.0):
    """
    Optimize using SLSQP with explicit constraints.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Generate constraints
    constraints = _slsqp_constraints(params)
    
    # Phase 1: SLSQP optimization
    try:
        result = minimize(_slsqp_objective, best_params, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-10})
        
        if result.success or result.fun < _slsqp_objective(best_params):
            # Check if result is valid
            res_xyz = result.x.reshape(N, 3)
            if _validate(res_xyz):
                best_params = result.x
                best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget:
        return best_params.reshape(N, 3)
    
    # Phase 2: Polish with tighter tolerance
    try:
        constraints = _slsqp_constraints(best_params) # Regenerate to be safe, though functions are closures
        result = minimize(_slsqp_objective, best_params, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12})
        
        if result.success or result.fun < _slsqp_objective(best_params):
            res_xyz = result.x.reshape(N, 3)
            if _validate(res_xyz):
                best_params = result.x
                best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_initial_config_grid())
    initial_configs.append(_initial_config_hex())
    
    # Add several force-directed initializations with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initial_configs.append(_initial_config_force_directed(seed))
    
    # Add a few random ones
    np.random.seed(42)
    for trial in range(3):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
        
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time budget
        time_budget = min(20.0, remaining - 2.0)
        if time_budget < 2.0:
            break
            
        xyz_opt = _optimize_slsqp(xyz0, time_budget=time_budget)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
            
    # Final polish on the best configuration
    elapsed = time.time() - global_start
    if elapsed < deadline - 3.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 1.0:
            # Jitter and re-optimize to escape local minima
            np.random.seed(123)
            jittered = best_xyz + np.random.normal(0, 0.01, best_xyz.shape)
            # Ensure jittered is roughly valid
            for i in range(N):
                jittered[i, 0] = np.clip(jittered[i, 0], 0, 1)
                jittered[i, 1] = np.clip(jittered[i, 1], 0, 1)
                jittered[i, 2] = np.clip(jittered[i, 2], 0.01, 0.5)
                
            xyz_polished = _optimize_slsqp(jittered, time_budget=remaining)
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_polished
                
    # Final check
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
