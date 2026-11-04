"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _slsqp_objective(x, pos):
    """
    Objective: minimize -sum(radii)
    x: array of shape (N,) containing radii
    pos: array of shape (N, 2) fixed positions
    """
    radii = x
    return -np.sum(radii)

def _slsqp_constraints(pos):
    """
    Generate constraints for SLSQP.
    1. Boundary constraints: pos[i] - r[i] >= 0, pos[i] + r[i] <= 1
       => r[i] <= pos[i, 0], r[i] <= pos[i, 1], r[i] <= 1 - pos[i, 0], r[i] <= 1 - pos[i, 1]
       => r[i] - pos[i, 0] <= 0
       => r[i] - pos[i, 1] <= 0
       => r[i] - (1 - pos[i, 0]) <= 0
       => r[i] - (1 - pos[i, 1]) <= 0
    2. Pairwise constraints: ||pos[i] - pos[j]|| - r[i] - r[j] >= 0
       => r[i] + r[j] - ||pos[i] - pos[j]|| <= 0
    """
    cons = []
    
    # Boundary constraints (4 per circle)
    for i in range(N):
        # r_i <= x_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - pos[i, 0] <= 0})
        # r_i <= y_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - pos[i, 1] <= 0})
        # r_i <= 1 - x_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - (1.0 - pos[i, 0]) <= 0})
        # r_i <= 1 - y_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - (1.0 - pos[i, 1]) <= 0})
        
    # Pairwise constraints
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, d=dist: x[i] + x[j] - d <= 0})
            
    return cons

def _slsqp_optimize_radii(pos, deadline, initial_radii=None):
    """
    Optimize radii for fixed positions using SLSQP.
    """
    if initial_radii is None:
        # Initial guess: min distance to boundary
        initial_radii = np.array([
            min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1]) * 0.5 
            for i in range(N)
        ])
        # Ensure non-negative
        initial_radii = np.maximum(initial_radii, 1e-6)
        
    x0 = initial_radii
    bounds = [(1e-9, 1.0)] * N
    cons = _slsqp_constraints(pos)
    
    try:
        res = minimize(
            _slsqp_objective, 
            x0, 
            args=(pos,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )
        if res.success:
            return res.x
        else:
            # If SLSQP fails, return best feasible point found or initial
            # Check if initial is feasible
            return x0
    except Exception:
        return x0

def _repair_radii(pos, radii, deadline):
    """
    Post-processing: try to increase radii if there is slack.
    This handles cases where SLSQP might be conservative.
    We do a simple iterative pass: for each circle, try to increase radius
    until it hits a constraint.
    """
    radii = radii.copy()
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        changed = False
        for i in range(N):
            # Calculate max possible radius for circle i given current other radii
            max_r = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            for j in range(N):
                if i == j:
                    continue
                dist = np.linalg.norm(pos[i] - pos[j])
                max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] + 1e-9:
                # Increase to max_r
                radii[i] = max_r
                changed = True
            elif max_r < radii[i] - 1e-9:
                # Shrink if overlapping (shouldn't happen if SLSQP succeeded, but safe)
                radii[i] = max(0.0, max_r)
                changed = True
        if not changed:
            break
    return radii

def _validate_result(result):
    """Check if result is valid (disjoint circles in unit square)."""
    n = result.shape[0]
    for i in range(n):
        x, y, r = result[i]
        if x - r < -1e-9 or y - r < -1e-9:
            return False
        if x + r > 1.0 + 1e-9 or y + r > 1.0 + 1e-9:
            return False
        if r < 0:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = result[i,0] - result[j,0]
            dy = result[i,1] - result[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < result[i,2] + result[j,2] - 1e-9:
                return False
    return True

def _generate_initial_positions(seed):
    """
    Generate diverse initial positions.
    """
    rng = np.random.RandomState(seed)
    configs = []
    
    # 1. Grid 5x5 + 1
    pos = []
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            pos.append([x, y])
    pos.append([0.5, 0.5])
    pos = np.array(pos[:N])
    configs.append(pos)
    
    # 2. Hexagonal
    pos = []
    dx = 1.0 / 5.0
    dy = dx * np.sqrt(3) / 2
    for r in range(8):
        for c in range(6):
            if len(pos) >= N:
                break
            x = (c + 0.5) * dx
            y = (r + 0.5) * dy
            if r % 2 == 1:
                x += dx / 2.0
            if 0 <= x <= 1 and 0 <= y <= 1:
                pos.append([x, y])
    pos = np.array(pos[:N])
    configs.append(pos)
    
    # 3. Random with minimum distance
    for _ in range(5):
        pos = []
        min_dist = 0.15
        attempts = 0
        while len(pos) < N and attempts < 2000:
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            ok = True
            for px, py in pos:
                if (x - px)**2 + (y - py)**2 < min_dist**2:
                    ok = False
                    break
            if ok:
                pos.append([x, y])
            attempts += 1
        # If not enough, fill randomly
        while len(pos) < N:
            pos.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
        pos = np.array(pos[:N])
        configs.append(pos)
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    configs = _generate_initial_positions(42)
    
    for idx, pos in enumerate(configs):
        if time.time() > deadline - 20:
            break
            
        try:
            # Optimize radii for fixed positions
            radii = _slsqp_optimize_radii(pos, deadline)
            
            # Repair/Polish radii
            radii = _repair_radii(pos, radii, deadline)
            
            # Construct result
            result = np.zeros((N, 3))
            result[:, :2] = pos
            result[:, 2] = radii
            
            # Validate
            if _validate_result(result):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = result
            else:
                # If invalid, try to fix by shrinking radii slightly
                for i in range(N):
                    result[i, 2] *= 0.99
                if _validate_result(result):
                    score = np.sum(result[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = result
                        
        except Exception:
            continue
            
    if best_result is None:
        # Fallback: Grid with small radii
        result = np.zeros((N, 3))
        for i in range(N):
            result[i, 0] = (i % 5 + 0.5) / 5.0
            result[i, 1] = (i // 5 + 0.5) / 5.0
            result[i, 2] = 0.09
        return result
        
    return best_result

# EVOLVE-BLOCK-END
