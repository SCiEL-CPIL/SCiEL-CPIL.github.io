"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9

def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - r[j])
        mr[i] = max(0.0, dmin)
    return mr

def _valid_check(pos, r, tol=1e-7):
    """Check validity: in square, no overlaps."""
    n = len(r)
    for i in range(n):
        if r[i] < -tol:
            return False
        if pos[i, 0] < r[i] - tol or pos[i, 0] > 1.0 - r[i] + tol:
            return False
        if pos[i, 1] < r[i] - tol or pos[i, 1] > 1.0 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _delaunay_edges(pos):
    """Build Delaunay triangulation of centers, return edge list (i, j) with i < j."""
    n = len(pos)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _build_constraints(pos, r, edges):
    """Build constraint functions for SLSQP."""
    n = len(r)
    # Wall constraints: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    # We'll encode as: pos[i,0] - r[i] >= 0, etc.
    # SLSQP constraints: fun >= 0
    cons = []
    for i in range(n):
        # x_i - r_i >= 0
        ii = i
        cons.append({'type': 'ineq', 'fun': lambda p, ii=ii: p[2*ii] - p[2*n+ii],
                     'jac': lambda p, ii=ii: np.array([0.0, 0.0, 1.0, 0.0])[:2*ii+2] + np.array([0.0, 0.0])[:0]})
    # This lambda approach is tricky. Let's build them properly.
    
    # Better: define constraint functions explicitly
    cons = []
    # Wall constraints
    for i in range(n):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i] - p[2*n+i],
                     'jac': lambda p, i=i: np.zeros(3*n).copy() + np.array([1.0, 0.0])[:2] * 0 + np.array([0.0, 0.0])})
        # Actually, let's just do it numerically or build properly.
    
    # Let me rebuild this more carefully.
    # Variables: [x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}, r_0, r_1, ..., r_{n-1}]
    # Index of x_i: 2*i, y_i: 2*i+1, r_i: 2*n+i
    
    cons = []
    # Wall constraints: 4 per circle
    for i in range(n):
        xi = 2*i
        yi = 2*i+1
        ri = 2*n+i
        # x_i - r_i >= 0
        def c1(p, xi=xi, ri=ri):
            return p[xi] - p[ri]
        def j1(p, xi=xi, ri=ri):
            v = np.zeros(3*n)
            v[xi] = 1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c1, 'jac': j1})
        
        # 1 - x_i - r_i >= 0
        def c2(p, xi=xi, ri=ri):
            return 1.0 - p[xi] - p[ri]
        def j2(p, xi=xi, ri=ri):
            v = np.zeros(3*n)
            v[xi] = -1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c2, 'jac': j2})
        
        # y_i - r_i >= 0
        def c3(p, yi=yi, ri=ri):
            return p[yi] - p[ri]
        def j3(p, yi=yi, ri=ri):
            v = np.zeros(3*n)
            v[yi] = 1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c3, 'jac': j3})
        
        # 1 - y_i - r_i >= 0
        def c4(p, yi=yi, ri=ri):
            return 1.0 - p[yi] - p[ri]
        def j4(p, yi=yi, ri=ri):
            v = np.zeros(3*n)
            v[yi] = -1.0
            v[ri] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c4, 'jac': j4})
        
        # r_i >= 0 (small positive)
        def c5(p, ri=ri):
            return p[ri]
        def j5(p, ri=ri):
            v = np.zeros(3*n)
            v[ri] = 1.0
            return v
        cons.append({'type': 'ineq', 'fun': c5, 'jac': j5})
    
    # Non-overlap constraints for Delaunay edges: dist(i,j) - r_i - r_j >= 0
    for (i, j) in edges:
        xi, yi = 2*i, 2*i+1
        xj, yj = 2*j, 2*j+1
        ri, rj = 2*n+i, 2*n+j
        
        def c_overlap(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]
            dy = p[yi] - p[yj]
            d = np.hypot(dx, dy)
            return d - p[ri] - p[rj]
        
        def j_overlap(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]
            dy = p[yi] - p[yj]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                d = 1e-12
            v = np.zeros(3*n)
            v[xi] = dx / d
            v[yi] = dy / d
            v[xj] = -dx / d
            v[yj] = -dy / d
            v[ri] = -1.0
            v[rj] = -1.0
            return v
        
        cons.append({'type': 'ineq', 'fun': c_overlap, 'jac': j_overlap})
    
    return cons

def _slsqp_step(pos, r, edges, time_budget):
    """One SLSQP step to optimize positions and radii."""
    n = len(r)
    x0 = np.concatenate([pos.flatten(), r])
    
    def objective(p):
        return -np.sum(p[2*n:])  # minimize -sum(r)
    
    cons = _build_constraints(pos, r, edges)
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 50, 'ftol': 1e-8})
        if res.success or res.fun < objective(x0):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_r = res.x[2*n:]
            if _valid_check(new_pos, new_r):
                return new_pos, new_r
    except Exception:
        pass
    
    return pos, r

def _grow_and_relax(pos, r, edges, t_start, deadline):
    """Alternating growth and relaxation."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    for iteration in range(30):
        if time.time() > t_start + deadline:
            break
        
        # Step 1: Grow radii (fix positions, maximize radii)
        for i in range(n):
            if time.time() > t_start + deadline:
                break
            # Max radius for circle i given others
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)
        
        # Step 2: SLSQP relaxation with current edges
        new_pos, new_r = _slsqp_step(pos, r, edges, 5.0)
        pos = new_pos
        r = new_r
        
        # Update edges
        edges = _delaunay_edges(pos)
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()
        
        # Check if converged
        if np.sum(r) - best_score < 1e-10:
            break
    
    return best_pos, best_r

def _polish(pos, r, t_start, deadline):
    """Final polish with multiple SLSQP iterations."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    for _ in range(5):
        if time.time() > t_start + deadline:
            break
        
        edges = _delaunay_edges(pos)
        new_pos, new_r = _slsqp_step(pos, r, edges, 10.0)
        
        if np.sum(new_r) > best_score:
            best_score = np.sum(new_r)
            best_pos = new_pos
            best_r = new_r
        
        pos = best_pos
        r = best_r
    
    return best_pos, best_r

def construct():
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit
    
    best_score = -1.0
    best_result = None
    
    # Multiple restarts with different initial configurations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > t_start + DEADLINE - 20.0:
            break
        
        rng = np.random.default_rng(1000 + restart)
        
        # Initial configuration: grid + random perturbation
        if restart == 0:
            # Pure grid
            side = int(np.ceil(np.sqrt(N)))
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side
                pos[i, 1] = (row + 0.5) / side
                r[i] = 0.4 / side
        elif restart == 1:
            # Grid with jitter
            side = int(np.ceil(np.sqrt(N)))
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side + rng.normal(0, 0.02)
                pos[i, 1] = (row + 0.5) / side + rng.normal(0, 0.02)
                r[i] = 0.35 / side
            pos = np.clip(pos, 0.05, 0.95)
        else:
            # Random positions with small radii
            pos = rng.random((N, 2)) * 0.8 + 0.1
            r = np.full(N, 0.02)
        
        # Ensure initial validity
        if not _valid_check(pos, r):
            # Shrink radii until valid
            scale = 0.5
            while not _valid_check(pos, r * scale) and scale > 0.01:
                scale *= 0.7
            r = r * scale
        
        # Build initial edges
        edges = _delaunay_edges(pos)
        
        # Grow and relax
        pos, r = _grow_and_relax(pos, r, edges, t_start, DEADLINE - 10.0)
        
        # Polish
        pos, r = _polish(pos, r, t_start, DEADLINE - 5.0)
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, r])
        
        # Early exit if very good
        if best_score > 2.8:
            break
    
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            r[i] = 0.4 / side
        best_result = np.column_stack([pos, r])
    
    return best_result

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
