"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26


def _valid(x, y, r):
    """Check if all circles are inside [0,1]^2 and have disjoint interiors."""
    if len(r) != N:
        return False
    if np.any(r <= 0):
        return False
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return False
    if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < -1e-9) or np.any(r > 0.5 + 1e-9):
        return False
    # Check pairwise distances
    for i in range(N):
        if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9:
            return False
        if y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
            if d2 < (r[i] + r[j])**2 - 1e-9:
                return False
    return True


def _score(x, y, r):
    """Sum of radii, only if valid."""
    if _valid(x, y, r):
        return np.sum(r)
    return 0.0


def _init_grid(n):
    """Grid initialization with small jitter."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.5 / side - 0.002
            cells.append([cx, cy, cr])
    return np.array(cells[:n])


def _force_directed_relax(x, y, r, n_iter=300, dt=0.005, repulse_exp=2, wall_repulse=0.5):
    """Phase-based force-directed relaxation with grow-then-pack."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = len(r)
    
    for it in range(n_iter):
        # Decay step size
        step = dt * (1.0 - it / n_iter)
        if step < 1e-6:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Pairwise repulsion (nonlinear)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d * 1.5:  # Only interact when close
                    if d < 1e-12:
                        dx = np.random.randn() * 1e-6
                        dy = np.random.randn() * 1e-6
                        d = np.sqrt(dx*dx + dy*dy)
                    overlap = min_d - d
                    if overlap > 0:
                        # Strong repulsion when overlapping
                        force = overlap ** repulse_exp * 10.0
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
                        # Slightly shrink both to resolve overlap
                        fr[i] -= overlap * 0.1
                        fr[j] -= overlap * 0.1
        
        # Wall constraints
        for i in range(n):
            # Left wall
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * wall_repulse
                fr[i] -= (r[i] - x[i]) * 0.5
            # Right wall
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * wall_repulse
                fr[i] -= (x[i] - (1 - r[i])) * 0.5
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * wall_repulse
                fr[i] -= (r[i] - y[i]) * 0.5
            # Top wall
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * wall_repulse
                fr[i] -= (y[i] - (1 - r[i])) * 0.5
        
        # Update positions
        x += step * fx
        y += step * fy
        r += step * fr * 0.3  # Radii change slower
        
        # Enforce boundaries
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.clip(r, 0.001, 0.5)
    
    return x, y, r


def _slsqp_polish(x, y, r, maxiter=200):
    """SLSQP optimization to maximize sum of radii."""
    n = len(r)
    
    def objective(params):
        px = params[:n]
        py = params[n:2*n]
        pr = params[2*n:]
        return -np.sum(pr)
    
    # Inequality constraints: g_i >= 0
    constraints = []
    
    # Circle i inside square
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i], 'jac': lambda p, i=i: np.array([1.0 if k == i else 0.0 for k in range(3*n)] - [0.0]*n - [0.0]*n + [1.0 if k == 2*n+i else 0.0 for k in range(n)])})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i], 'jac': lambda p, i=i: np.array([-1.0 if k == i else 0.0 for k in range(3*n)] - [0.0]*n - [0.0]*n + [-1.0 if k == 2*n+i else 0.0 for k in range(n)])})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i], 'jac': lambda p, i=i: np.array([0.0]*n + [1.0 if k == n+i else 0.0 for k in range(n)] + [0.0]*n + [1.0 if k == 2*n+i else 0.0 for k in range(n)])})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i], 'jac': lambda p, i=i: np.array([0.0]*n + [-1.0 if k == n+i else 0.0 for k in range(n)] + [0.0]*n + [-1.0 if k == 2*n+i else 0.0 for k in range(n)])})
    
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 
                'fun': lambda p, i=i, j=j: (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2,
                'jac': lambda p, i=i, j=j: np.array([2*(p[i]-p[j]) if k==i else -2*(p[i]-p[j]) if k==j else 0.0 for k in range(n)] + 
                    [2*(p[n+i]-p[n+j]) if k==n+i else -2*(p[n+i]-p[n+j]) if k==n+j else 0.0 for k in range(n)] +
                    [-2*(p[2*n+i]+p[2*n+j]) if k==2*n+i else -2*(p[2*n+i]+p[2*n+j]) if k==2*n+j else 0.0 for k in range(n)])})
    
    x0 = np.concatenate([x, y, r])
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0.001, 0.5)] * n
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-12})
    
    if result.success or result.fun < objective(x0):
        px = result.x[:n]
        py = result.x[n:2*n]
        pr = result.x[2*n:]
        return px, py, pr
    
    return x, y, r


def _nelder_mead_radius_polish(x, y, r, maxiter=500):
    """Nelder-Mead on radii only, keeping positions fixed."""
    n = len(r)
    
    def objective(pr):
        # Check validity with fixed positions
        for i in range(n):
            if x[i] < pr[i] - 1e-10 or x[i] > 1 - pr[i] + 1e-10:
                return 1e10
            if y[i] < pr[i] - 1e-10 or y[i] > 1 - pr[i] + 1e-10:
                return 1e10
        for i in range(n):
            for j in range(i+1, n):
                d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                if d2 < (pr[i]+pr[j])**2 - 1e-10:
                    return 1e10
        return -np.sum(pr)
    
    bounds = [(0.001, min(x[i], 1-x[i], y[i], 1-y[i])) for i in range(n)]
    
    # Clip initial radii to bounds
    r0 = np.array([min(max(r[i], b[0]), b[1]) for i, b in enumerate(bounds)])
    
    result = minimize(objective, r0, method='Nelder-Mead', 
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
    
    if result.fun < 1e9:
        return x, y, result.x
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    # Phase 1: Grid initialization
    x, y, r = _init_grid(N)
    best_score = _score(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Phase 2: Force-directed relaxation
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        x, y, r = _force_directed_relax(x, y, r, n_iter=500, dt=0.008)
        s = _score(x, y, r)
        if s > best_score:
            best_score = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Phase 3: SLSQP polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        x, y, r = _slsqp_polish(best_x, best_y, best_r, maxiter=150)
        s = _score(x, y, r)
        if s > best_score:
            best_score = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Phase 4: Nelder-Mead radius polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        x, y, r = _nelder_mead_radius_polish(best_x, best_y, best_r, maxiter=300)
        s = _score(x, y, r)
        if s > best_score:
            best_score = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Final validity check
    if not _valid(best_x, best_y, best_r):
        # Fallback to grid
        x, y, r = _init_grid(N)
    
    return np.column_stack([best_x, best_y, best_r])
