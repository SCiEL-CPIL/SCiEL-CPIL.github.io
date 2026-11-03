"""Model-and-optimize circle packing in the unit square.
Uses a topology-based approach: maintain a contact graph, solve for
tangency constraints via iterative projection, and refine with local search.
"""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0
START = time.time()


def _elapsed():
    return time.time() - START


def _build_contact_graph(x, y, r, tol=1e-4):
    """Build contact graph: pairs (i,j) where circles i and j are close to touching."""
    n = len(r)
    contacts = []
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    # Contact if distance is within tolerance of sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dist[i, j] - need[i, j]) < tol:
                contacts.append((i, j))
    return contacts


def _solve_tangency_constraints(x, y, r, contacts, max_iter=1000, tol=1e-8):
    """Iteratively adjust positions to satisfy tangency constraints for given contacts.
    
    For each contact (i,j), we want |x_i - x_j| = r_i + r_j.
    We use a simple iterative projection method.
    """
    n = len(r)
    x = x.copy()
    y = y.copy()
    
    for iteration in range(max_iter):
        if _elapsed() > DEADLINE - 5.0:
            break
        max_change = 0.0
        
        for (i, j) in contacts:
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            target = r[i] + r[j]
            
            if dist < 1e-10:
                continue
            
            # Error: how much do we need to adjust?
            error = dist - target
            if abs(error) < tol:
                continue
            
            # Move each circle half the error along the line connecting them
            # Weight by inverse radius (smaller circles move more)
            w_i = r[j] / (r[i] + r[j])
            w_j = r[i] / (r[i] + r[j])
            
            unit_x = dx / dist
            unit_y = dy / dist
            
            # Move i in direction away from j if too close, toward j if too far
            adjust = error * 0.5
            x[i] += w_i * adjust * unit_x
            y[i] += w_i * adjust * unit_y
            x[j] -= w_j * adjust * unit_x
            y[j] -= w_j * adjust * unit_y
            
            max_change = max(max_change, abs(adjust))
        
        # Clip to boundary
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        if max_change < tol:
            break
    
    return x, y


def _local_search(x, y, r, contacts, max_iter=2000):
    """Local search: try small perturbations and keep if they maintain feasibility
    and increase sum of radii."""
    n = len(r)
    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    step = 0.01
    for iteration in range(max_iter):
        if _elapsed() > DEADLINE - 5.0:
            break
        
        # Try a random perturbation
        i = np.random.randint(n)
        dx = np.random.uniform(-step, step)
        dy = np.random.uniform(-step, step)
        
        new_x = best_x.copy()
        new_y = best_y.copy()
        new_x[i] += dx
        new_y[i] += dy
        
        # Clip to boundary
        new_x[i] = np.clip(new_x[i], best_r[i], 1 - best_r[i])
        new_y[i] = np.clip(new_y[i], best_r[i], 1 - best_r[i])
        
        # Check if still feasible
        if _is_feasible(new_x, new_y, best_r):
            # Try to increase radius of circle i
            max_r = min(new_x[i], 1 - new_x[i], new_y[i], 1 - new_y[i])
            # Also check against all other circles
            for j in range(n):
                if j == i:
                    continue
                dx_j = new_x[i] - new_x[j]
                dy_j = new_y[i] - new_y[j]
                dist_j = np.sqrt(dx_j * dx_j + dy_j * dy_j)
                max_r = min(max_r, dist_j - best_r[j])
            
            if max_r > best_r[i] + 1e-8:
                best_r[i] = max_r
                best_x, best_y = new_x, new_y
                best_sum = np.sum(best_r)
        
        # Decay step size
        if iteration % 100 == 0:
            step *= 0.95
    
    return best_x, best_y, best_r, best_sum


def _is_feasible(x, y, r, eps=1e-7):
    """Check if configuration is feasible (no overlaps, all inside square)."""
    n = len(r)
    # Boundary check
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    
    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    
    return True


def _grid_init(rng, r0=0.18):
    """5x5 grid + 1 extra circle."""
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    # Add one extra
    pts.append((0.5, 0.5 + 0.5 / cols))
    rng.shuffle(pts)
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, r0)
    return x, y, r


def _fix_to_boundary(x, y, r):
    """Clip radii so each circle fits inside the square."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return np.minimum(r, np.maximum(d, 0.0))


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)
    
    # Multiple initial configurations
    inits = [
        _grid_init(rng, 0.18),
        _grid_init(rng, 0.20),
        _grid_init(rng, 0.16),
    ]
    
    for x0, y0, r0 in inits:
        if _elapsed() > DEADLINE - 10.0:
            break
        
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        r = _fix_to_boundary(x, y, r)
        
        # Build initial contact graph
        contacts = _build_contact_graph(x, y, r, tol=1e-3)
        
        # Solve tangency constraints
        x, y = _solve_tangency_constraints(x, y, r, contacts, max_iter=500)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # Local search to increase radii
        x, y, r, val = _local_search(x, y, r, contacts, max_iter=1000)
        
        if _is_feasible(x, y, r):
            if val > best_val:
                best_val = val
                best = np.column_stack([x, y, r])
    
    if best is not None:
        return best
    
    # Fallback: grid of 25 + 1 tiny
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
