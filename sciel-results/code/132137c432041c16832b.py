"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate a strictly feasible initial configuration
    # Use a grid that guarantees non-overlap and boundary containment
    side = 6
    # We have 26 circles. 6x6 grid has 36 spots. We use the first 26.
    # Spacing is 1/6. Max radius to stay inside and not overlap neighbors in grid is approx 1/(2*6) = 0.0833
    # Let's use r = 0.08 for safety.
    r_init = 0.08
    v0 = np.zeros(N*3)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            v0[3*idx] = (i + 0.5) / side
            v0[3*idx+1] = (j + 0.5) / side
            v0[3*idx+2] = r_init
            idx += 1
    
    # Optimization setup
    def objective(v):
        radii = v[2::3]
        return -np.sum(radii)
    
    def objective_grad(v):
        grad = np.zeros_like(v)
        grad[2::3] = -1.0
        return grad
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.array([1.0, 0.0, -1.0])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.array([0.0, 1.0, -1.0])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.array([-1.0, 0.0, -1.0])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.array([0.0, -1.0, -1.0])
        })
        
    # Pairwise distance constraints
    def _dist_jac(v, i, j):
        jac = np.zeros_like(v)
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        r_sum = v[3*i+2] + v[3*j+2]
        jac[3*i] = 2*dx
        jac[3*i+1] = 2*dy
        jac[3*j] = -2*dx
        jac[3*j+1] = -2*dy
        jac[3*i+2] = -2*r_sum
        jac[3*j+2] = -2*r_sum
        return jac

    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2,
                'jac': lambda v, i=i, j=j: _dist_jac(v, i, j)
            })

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-6, 0.5))
    
    best_result = v0.copy()
    best_value = np.sum(v0[2::3])
    
    # Run optimization
    # SLSQP can be slow. We limit iterations.
    try:
        res = minimize(objective, v0, method='SLSQP', jac=objective_grad,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
        
        if res.fun < -best_value:
            # Check if result is feasible enough
            # SLSQP might drift slightly out of bounds. We will repair it.
            candidate = res.x
            if _is_feasible(candidate, tol=1e-5):
                best_result = candidate
                best_value = -res.fun
            else:
                # Repair the result
                repaired = _repair(candidate)
                if _is_feasible(repaired, tol=1e-5):
                    val = np.sum(repaired[2::3])
                    if val > best_value:
                        best_result = repaired
                        best_value = val
    except Exception as e:
        pass
    
    # If time permits, try a second random start
    if time.time() < deadline - 20:
        np.random.seed(42)
        v1 = np.zeros(N*3)
        # Random feasible start
        for i in range(N):
            # Place in grid cells to avoid overlap initially
            row = i // 6
            col = i % 6
            v1[3*i] = (col + 0.5) / 6.0 + np.random.uniform(-0.02, 0.02)
            v1[3*i+1] = (row + 0.5) / 6.0 + np.random.uniform(-0.02, 0.02)
            v1[3*i+2] = 0.07
            
        try:
            res2 = minimize(objective, v1, method='SLSQP', jac=objective_grad,
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
            
            if res2.fun < -best_value:
                candidate = res2.x
                if _is_feasible(candidate, tol=1e-5):
                    best_result = candidate
                    best_value = -res2.fun
                else:
                    repaired = _repair(candidate)
                    if _is_feasible(repaired, tol=1e-5):
                        val = np.sum(repaired[2::3])
                        if val > best_value:
                            best_result = repaired
                            best_value = val
        except Exception as e:
            pass
            
    return _format_result(best_result)

def _is_feasible(v, tol=1e-6):
    for i in range(N):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        if x < r - tol or y < r - tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist_sq = dx*dx + dy*dy
            r_sum = v[3*i+2] + v[3*j+2]
            if dist_sq < (r_sum - tol)**2:
                return False
    return True

def _repair(v):
    # Simple repair: clip to bounds and shrink radii if overlapping
    v = v.copy()
    for _ in range(100):
        # Clip to square
        for i in range(N):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if x < r:
                x = r
            if y < r:
                y = r
            if x + r > 1:
                x = 1 - r
            if y + r > 1:
                y = 1 - r
            v[3*i] = x
            v[3*i+1] = y
            
        # Check overlaps and shrink
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = v[3*i+2] + v[3*j+2]
                if dist < r_sum:
                    # Shrink both radii
                    shrink = (r_sum - dist) / 2.0 + 1e-6
                    v[3*i+2] = max(1e-6, v[3*i+2] - shrink)
                    v[3*j+2] = max(1e-6, v[3*j+2] - shrink)
                    changed = True
        if not changed:
            break
    return v

def _format_result(v):
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = v[3*i]
        result[i, 1] = v[3*i+1]
        result[i, 2] = v[3*i+2]
    return result
# EVOLVE-BLOCK-END
