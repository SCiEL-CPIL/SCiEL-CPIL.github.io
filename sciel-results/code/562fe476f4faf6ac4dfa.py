"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26
import numpy as np
from scipy.optimize import minimize
import time

def _objective_and_grad(x):
    """x = [x1,y1,r1, x2,y2,r2, ...] length 3N. Minimize negative sum of radii."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:,0], pos[:,1], pos[:,2]
    # Objective: minimize -sum(r)
    obj = -np.sum(rs)
    return obj

def _constraints(x):
    """Return list of constraint dicts for scipy minimize."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:,0], pos[:,1], pos[:,2]
    cons = []
    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})          # x >= r
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1-x[3*i]) - x[3*i+2]})     # 1-x >= r
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})        # y >= r
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1-x[3*i+1]) - x[3*i+2]})    # 1-y >= r
    # Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                         (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2 - (x[3*i+2]+x[3*j+2])**2})
    # Radius positive
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2]})
    return cons

def _initial_ring_config(rng):
    """Generate a ring-based initial configuration."""
    pos = np.zeros((N, 3))
    # Central circle
    pos[0] = [0.5, 0.5, 0.15]
    # Ring 1: 6 circles
    idx = 1
    for k in range(6):
        angle = 2 * np.pi * k / 6
        r_ring = 0.30
        pos[idx, 0] = 0.5 + r_ring * np.cos(angle)
        pos[idx, 1] = 0.5 + r_ring * np.sin(angle)
        pos[idx, 2] = 0.08
        idx += 1
    # Ring 2: 12 circles
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi/12
        r_ring = 0.42
        pos[idx, 0] = 0.5 + r_ring * np.cos(angle)
        pos[idx, 1] = 0.5 + r_ring * np.sin(angle)
        pos[idx, 2] = 0.05
        idx += 1
    # Remaining 7 circles in corners/edges
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.08), (0.5, 0.92), (0.08, 0.5)]
    for k in range(7):
        pos[idx, 0] = corners[k][0]
        pos[idx, 1] = corners[k][1]
        pos[idx, 2] = 0.04
        idx += 1
    # Clip to valid range
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
        pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    return pos.flatten()

def _initial_grid_config(rng):
    """Generate a perturbed grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 1] = (row + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 2] = 0.19
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
        pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    return pos.flatten()

def _is_valid(x, tol=1e-7):
    """Check if configuration is valid."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:,0], pos[:,1], pos[:,2]
    if np.any(rs < 0):
        return False
    # Boundary
    if np.any(xs < -tol) or np.any(xs > 1+tol) or np.any(ys < -tol) or np.any(ys > 1+tol):
        return False
    if np.any(rs > xs + tol) or np.any(rs > 1-xs + tol) or np.any(rs > ys + tol) or np.any(rs > 1-ys + tol):
        return False
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            d2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
            if d2 < (rs[i]+rs[j])**2 - tol:
                return False
    return True

def _sum_radii(x):
    return np.sum(x.reshape(N, 3)[:, 2])

def construct():
    deadline = time.time() + 130  # 20s safety margin
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = -np.inf
    
    # Try multiple initial configurations and restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Choose initial config
        if restart % 2 == 0:
            x0 = _initial_ring_config(rng)
        else:
            x0 = _initial_grid_config(rng)
        
        # Add small random perturbation
        x0 = x0 + rng.uniform(-0.005, 0.005, size=3*N)
        
        # Ensure initial feasibility (clip radii down if needed)
        pos = x0.reshape(N, 3)
        for i in range(N):
            pos[i, 2] = min(pos[i, 2], pos[i, 0], 1-pos[i, 0], pos[i, 1], 1-pos[i, 1], 0.49)
        x0 = pos.flatten()
        
        # Run SLSQP optimization
        try:
            cons = _constraints(x0)
            bounds = [(0, 1), (0, 1), (0, 0.5)] * N
            res = minimize(_objective_and_grad, x0, method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 300, 'ftol': 1e-10})
            if res.success or res.fun < best_val:
                if _is_valid(res.x):
                    val = _sum_radii(res.x)
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
        except Exception:
            pass
        
        # Also try a quick local SA refinement from the best found
        if best_x is not None and time.time() < deadline - 15:
            _sa_refine(best_x, rng, deadline - 15)
            if _is_valid(best_x):
                val = _sum_radii(best_x)
                if val > best_val:
                    best_val = val
    
    if best_x is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_x.reshape(N, 3)

def _sa_refine(x, rng, deadline):
    """Simple simulated annealing refinement in-place."""
    pos = x.reshape(N, 3)
    current_val = _sum_radii(x)
    T = 0.01
    step = 0.01
    n_iter = 2000
    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Pick a random circle and dimension
        i = rng.integers(N)
        dim = rng.integers(3)
        old = pos[i, dim]
        pos[i, dim] += rng.normal(0, step)
        # Clip
        if dim == 0:
            pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        elif dim == 1:
            pos[i, 1] = np.clip(pos[i, 1], 0, 1)
        else:
            pos[i, 2] = np.clip(pos[i, 2], 0, 0.5)
        
        # Check validity
        if _is_valid(pos.flatten()):
            new_val = _sum_radii(pos.flatten())
            delta = new_val - current_val
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_val = new_val
            else:
                pos[i, dim] = old
        else:
            pos[i, dim] = old
        T *= 0.999
        step *= 0.9995
    x[:] = pos.flatten()
