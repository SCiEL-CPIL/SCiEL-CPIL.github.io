"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def _violation(vars):
    """Compute constraint violation: sum of squared violations."""
    n = N
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    viol = 0.0
    # Boundary constraints
    viol += np.sum(np.maximum(0, x - r)**2)  # x - r < 0
    viol += np.sum(np.maximum(0, r - x)**2)  # x + r > 1 => r > 1-x => r - (1-x) > 0
    viol += np.sum(np.maximum(0, y - r)**2)
    viol += np.sum(np.maximum(0, r - y)**2)
    # Actually let me redo: x >= r, x <= 1-r, y >= r, y <= 1-r
    # So violations: r - x > 0, x + r - 1 > 0, r - y > 0, y + r - 1 > 0
    viol = 0.0
    viol += np.sum(np.maximum(0, r - x)**2)
    viol += np.sum(np.maximum(0, x + r - 1)**2)
    viol += np.sum(np.maximum(0, r - y)**2)
    viol += np.sum(np.maximum(0, y + r - 1)**2)
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist:
                viol += (min_dist - np.sqrt(dist_sq))**2
    return viol

def _objective(vars):
    """Negative sum of radii (to minimize)."""
    n = N
    r = vars[2*n:3*n]
    return -np.sum(r)

def _objective_with_penalty(vars, penalty_weight):
    """Objective with penalty for violations."""
    base = _objective(vars)
    viol = _violation(vars)
    return base + penalty_weight * viol

def _grad_penalty(vars, penalty_weight):
    """Gradient of penalized objective (numerical)."""
    eps = 1e-7
    grad = np.zeros_like(vars)
    f0 = _objective_with_penalty(vars, penalty_weight)
    for i in range(len(vars)):
        vars_p = vars.copy()
        vars_p[i] += eps
        f_p = _objective_with_penalty(vars_p, penalty_weight)
        grad[i] = (f_p - f0) / eps
    return grad

def _try_optimize(x0, time_limit, rng):
    """Try to optimize from initial point x0."""
    n = N
    start_time = time.time()
    
    # Start with small penalty, increase it
    penalty = 10.0
    best_vars = x0.copy()
    best_obj = _objective_with_penalty(x0, penalty)
    best_base_obj = _objective(x0)
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        try:
            result = minimize(
                _objective_with_penalty,
                best_vars,
                args=(penalty,),
                method='L-BFGS-B',
                jac=None,  # numerical gradient
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_obj:
                new_vars = result.x
                new_obj = result.fun
                new_base = _objective(new_vars)
                
                if new_base < best_base_obj:
                    best_vars = new_vars
                    best_base_obj = new_base
                    best_obj = new_obj
                
                # Check if constraints are satisfied
                viol = _violation(best_vars)
                if viol < 1e-10:
                    # Try to increase radii further by reducing penalty
                    penalty = max(1.0, penalty * 0.5)
                else:
                    penalty *= 2.0
                    if penalty > 1e6:
                        break
            else:
                penalty *= 1.5
                if penalty > 1e6:
                    break
        except Exception:
            break
    
    return best_vars, best_base_obj

def _local_refinement(vars, time_limit):
    """Polish the solution with tighter optimization."""
    n = N
    start_time = time.time()
    
    # Use SLSQP with explicit constraints for final polish
    # But SLSQP with many constraints can be slow, so use penalty with high weight
    
    penalty = 1e4
    best_vars = vars.copy()
    best_obj = _objective_with_penalty(vars, penalty)
    
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        try:
            result = minimize(
                _objective_with_penalty,
                best_vars,
                args=(penalty,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < best_obj:
                best_vars = result.x
                best_obj = result.fun
            else:
                break
        except Exception:
            break
    
    return best_vars

def construct():
    n = N
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    best_solution = None
    best_value = -np.inf
    
    def _is_valid(vars):
        """Check if solution is valid (all constraints satisfied)."""
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:3*n]
        
        # Check boundaries
        if np.any(r <= 0):
            return False
        if np.any(x < r - 1e-8):
            return False
        if np.any(x > 1 - r + 1e-8):
            return False
        if np.any(y < r - 1e-8):
            return False
        if np.any(y > 1 - r + 1e-8):
            return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist - 1e-8:
                    return False
        return True
    
    def _get_value(vars):
        if not _is_valid(vars):
            return -np.inf
        r = vars[2*n:3*n]
        return np.sum(r)
    
    # Generate multiple initial solutions
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Grid-like initial
            side = int(np.ceil(np.sqrt(n)))
            cells = []
            for i in range(side):
                for j in range(side):
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    r = 0.4 / side
                    cells.append((cx, cy, r))
            cells = cells[:n]
            x0 = np.zeros(3*n)
            for i, (cx, cy, r) in enumerate(cells):
                x0[i] = cx
                x0[n+i] = cy
                x0[2*n+i] = r
        elif restart == 1:
            # Random uniform with small radii
            x = np_rng.uniform(0.1, 0.9, n)
            y = np_rng.uniform(0.1, 0.9, n)
            r = np_rng.uniform(0.02, 0.06, n)
            x0 = np.concatenate([x, y, r])
        elif restart == 2:
            # Clustered
            centers = []
            for _ in range(4):
                cx = np_rng.uniform(0.2, 0.8)
                cy = np_rng.uniform(0.2, 0.8)
                centers.append((cx, cy))
            x = np.array([c[0] + np_rng.normal(0, 0.05) for c in centers for _ in range(n//4)])[:n]
            y = np.array([c[1] + np_rng.normal(0, 0.05) for c in centers for _ in range(n//4)])[:n]
            r = np_rng.uniform(0.03, 0.08, n)
            x0 = np.concatenate([x, y, r])
        else:
            # Random with varying radii
            x = np_rng.uniform(0.05, 0.95, n)
            y = np_rng.uniform(0.05, 0.95, n)
            r = np_rng.uniform(0.01, 0.1, n)
            x0 = np.concatenate([x, y, r])
        
        # Clamp to valid range
        x0[:n] = np.clip(x0[:n], 0.01, 0.99)
        x0[n:2*n] = np.clip(x0[n:2*n], 0.01, 0.99)
        x0[2*n:3*n] = np.clip(x0[2*n:3*n], 0.005, 0.3)
        
        # Time budget per restart
        remaining = deadline - time.time()
        if remaining < 2:
            break
        time_per_restart = min(remaining / (n_restarts - restart), 8.0)
        
        try:
            vars_sol, val = _try_optimize(x0, time_per_restart, rng)
            
            # Refine
            if time.time() < deadline - 2:
                vars_sol = _local_refinement(vars_sol, min(3.0, deadline - time.time() - 1))
            
            val = _get_value(vars_sol)
            if val > best_value:
                best_value = val
                best_solution = vars_sol.copy()
        except Exception:
            continue
    
    # If no valid solution found, use fallback
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.4 / side
                cells.append((cx, cy, r))
        cells = cells[:n]
        result = np.array(cells)
        return result
    
    x = best_solution[:n]
    y = best_solution[n:2*n]
    r = best_solution[2*n:3*n]
    
    result = np.column_stack([x, y, r])
    return result

# EVOLVE-BLOCK-END
