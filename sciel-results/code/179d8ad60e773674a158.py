"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _valid(x):
    """x: (3N,) vector -> positions and radii. Check constraints."""
    n = len(x) // 3
    pts = x.reshape(n, 3)
    r = pts[:, 2]
    # radii positive
    if np.any(r <= 1e-12):
        return False
    # inside unit square
    if np.any(pts[:, 0] < r) or np.any(pts[:, 1] < r):
        return False
    if np.any(pts[:, 0] > 1 - r) or np.any(pts[:, 1] > 1 - r):
        return False
    # pairwise disjoint
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True

def _score(x):
    if not _valid(x):
        return -np.inf
    return float(np.sum(x[2::3]))

def _contact_graph(x, tol=1e-6):
    """Return set of active constraint indices (pairs + walls)."""
    n = len(x) // 3
    pts = x.reshape(n, 3)
    active = set()
    # pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < pts[i, 2] + pts[j, 2] + tol:
                active.add(('p', i, j))
    # walls
    for i in range(n):
        if pts[i, 0] < pts[i, 2] + tol:
            active.add(('w', i, 0))
        if pts[i, 1] < pts[i, 2] + tol:
            active.add(('w', i, 1))
        if pts[i, 0] > 1 - pts[i, 2] + tol:
            active.add(('w', i, 2))
        if pts[i, 1] > 1 - pts[i, 2] + tol:
            active.add(('w', i, 3))
    return active

def _make_initial_grid(n):
    """Create a reasonable initial packing."""
    side = int(np.ceil(np.sqrt(n)))
    # Use a slightly larger grid and pick best n
    best_x = None
    best_score = -np.inf
    for offset in [0.0, 0.1, 0.2, 0.3]:
        for scale in [0.9, 0.95, 1.0]:
            pts = []
            for i in range(side):
                for j in range(side):
                    cx = (i + 0.5 + offset) / (side + offset * 2)
                    cy = (j + 0.5 + offset) / (side + offset * 2)
                    r = (0.5 / (side + offset * 2)) * scale
                    if 0 < r < 0.5:
                        pts.append([cx, cy, r])
            if len(pts) >= n:
                x = np.array(pts[:n]).flatten()
                s = _score(x)
                if s > best_score:
                    best_score = s
                    best_x = x
    if best_x is None:
        # Fallback: simple grid
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        best_x = np.array(pts[:n]).flatten()
    return best_x

def _slsqp_optimize(x0, max_iter=200, tol=1e-12):
    """SLSQP optimization with all constraints."""
    n = len(x0) // 3
    cons = []
    # Wall constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
    # Pairwise: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
            })
    
    def objective(v):
        return -np.sum(v[2::3])
    
    res = minimize(objective, x0, method='SLSQP', bounds=[(1e-8, 0.5)] * (3*n),
                   constraints=cons, options={'maxiter': max_iter, 'ftol': tol})
    if res.success or res.fun < objective(x0):
        return res.x
    return x0

def _polish(x0, ftol=1e-14, max_iter=300):
    """High-precision polish pass."""
    n = len(x0) // 3
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
            })
    
    def objective(v):
        return -np.sum(v[2::3])
    
    res = minimize(objective, x0, method='SLSQP', bounds=[(1e-8, 0.5)] * (3*n),
                   constraints=cons, options={'maxiter': max_iter, 'ftol': ftol})
    if res.success or res.fun < objective(x0):
        return res.x
    return x0

def construct():
    global _START
    _START = time.time()
    
    # Phase 1: Generate initial configuration
    x0 = _make_initial_grid(N)
    
    # Phase 2: SLSQP optimization
    if _remaining() > 30:
        x1 = _slsqp_optimize(x0, max_iter=200, tol=1e-12)
        if _score(x1) > _score(x0):
            x0 = x1
    
    # Phase 3: Polish with tighter tolerance
    if _remaining() > 15:
        x2 = _polish(x0, ftol=1e-14, max_iter=300)
        if _score(x2) > _score(x0):
            x0 = x2
    
    # Final validity check and repair if needed
    if not _valid(x0):
        # Try to fix by reducing radii slightly
        pts = x0.reshape(N, 3).copy()
        for _ in range(100):
            if _valid(pts.flatten()):
                break
            pts[:, 2] *= 0.999
        x0 = pts.flatten()
    
    # Ensure final result is valid
    if not _valid(x0):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        x0 = np.array(pts[:N]).flatten()
    
    return x0.reshape(N, 3)
