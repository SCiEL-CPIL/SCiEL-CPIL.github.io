"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _objective(v):
    """Maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[N * 2:])

def _wall_constraints(v):
    """Circle must be inside [0,1]^2: x >= r, y >= r, x+r <= 1, y+r <= 1."""
    n = len(v) // 3
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda vi=i, _x=x, _y=y, _r=r: _x[vi] - _r[vi]})
        cons.append({'type': 'ineq', 'fun': lambda vi=i, _x=x, _y=y, _r=r: _y[vi] - _r[vi]})
        cons.append({'type': 'ineq', 'fun': lambda vi=i, _x=x, _y=y, _r=r: 1.0 - _x[vi] - _r[vi]})
        cons.append({'type': 'ineq', 'fun': lambda vi=i, _x=x, _y=y, _r=r: 1.0 - _y[vi] - _r[vi]})
    return cons

def _pair_constraints(v):
    """No overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2."""
    n = len(v) // 3
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda vi=i, vj=j, _x=x, _y=y, _r=r:
                    (_x[vi] - _x[vj])**2 + (_y[vi] - _y[vj])**2 - (_r[vi] + _r[vj])**2
            })
    return cons

def _bounds(n):
    """Bounds for each variable: x,y in [0,1], r in [0, 0.5]."""
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-4, 0.5)) # r
    return bounds

def _is_valid(arr):
    """Check if the array represents valid non-overlapping circles in [0,1]^2."""
    n = len(arr)
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    # Check walls
    if np.any(x < r - 1e-8) or np.any(y < r - 1e-8):
        return False
    if np.any(x + r > 1.0 + 1e-8) or np.any(y + r > 1.0 + 1e-8):
        return False
    # Check pairs
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx*dx + dy*dy
            rsum2 = (r[i] + r[j])**2
            if dist2 < rsum2 - 1e-7:
                return False
    return True

def _fix_violations(arr, tol=1e-6):
    """Slightly shrink radii and adjust positions to fix minor violations."""
    arr = arr.copy()
    for _ in range(50):
        valid = True
        x = arr[:, 0]
        y = arr[:, 1]
        r = arr[:, 2]
        # Fix walls
        for i in range(len(arr)):
            if x[i] < r[i] - tol:
                x[i] = r[i] + tol
                valid = False
            if y[i] < r[i] - tol:
                y[i] = r[i] + tol
                valid = False
            if x[i] + r[i] > 1.0 + tol:
                x[i] = 1.0 - r[i] - tol
                valid = False
            if y[i] + r[i] > 1.0 + tol:
                y[i] = 1.0 - r[i] - tol
                valid = False
        arr[:, 0] = x
        arr[:, 1] = y
        # Fix overlaps by shrinking radii slightly
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                rsum = r[i] + r[j]
                if dist < rsum - tol:
                    # Shrink both radii proportionally
                    scale = dist / rsum * (1.0 - tol)
                    if scale < 1.0:
                        r[i] *= scale
                        r[j] *= scale
                        valid = False
        arr[:, 2] = r
        if valid:
            break
    return arr

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Initial grid layout
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    initial = np.array(cells[:N])
    
    # Reshape to vector [x0, y0, r0, x1, y1, r1, ...]
    v0 = np.zeros(N * 3)
    for i in range(N):
        v0[3*i] = initial[i, 0]
        v0[3*i+1] = initial[i, 1]
        v0[3*i+2] = initial[i, 2]
    
    best_v = v0.copy()
    best_score = np.sum(v0[2::3])
    
    # Build constraints
    constraints = _wall_constraints(v0) + _pair_constraints(v0)
    bounds = _bounds(N)
    
    # First SLSQP run
    try:
        res = minimize(_objective, v0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        if res.success or res.fun < -best_score:
            if res.fun < -best_score:
                best_score = -res.fun
                best_v = res.x.copy()
        else:
            # Even if not successful, check if it improved
            score = -res.fun
            if score > best_score:
                best_score = score
                best_v = res.x.copy()
    except Exception:
        pass
    
    # Second SLSQP run from the best point found
    if time.time() < deadline:
        try:
            # Rebuild constraints with updated values
            constraints2 = _wall_constraints(best_v) + _pair_constraints(best_v)
            res2 = minimize(_objective, best_v, method='SLSQP', bounds=bounds,
                           constraints=constraints2, options={'maxiter': 300, 'ftol': 1e-10})
            score2 = -res2.fun
            if score2 > best_score:
                best_score = score2
                best_v = res2.x.copy()
        except Exception:
            pass
    
    # Third SLSQP run for further refinement
    if time.time() < deadline - 10:
        try:
            constraints3 = _wall_constraints(best_v) + _pair_constraints(best_v)
            res3 = minimize(_objective, best_v, method='SLSQP', bounds=bounds,
                           constraints=constraints3, options={'maxiter': 500, 'ftol': 1e-12})
            score3 = -res3.fun
            if score3 > best_score:
                best_score = score3
                best_v = res3.x.copy()
        except Exception:
            pass
    
    # Convert back to array format
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_v[3*i]
        result[i, 1] = best_v[3*i+1]
        result[i, 2] = best_v[3*i+2]
    
    # Check validity and fix if needed
    if not _is_valid(result):
        result = _fix_violations(result)
        if not _is_valid(result):
            # Fall back to initial grid
            result = initial
    
    return result
# EVOLVE-BLOCK-END
