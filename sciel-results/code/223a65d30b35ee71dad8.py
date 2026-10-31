"""Circle packing in a perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize
import random

N = 21
DEADLINE = 115.0  # Safety margin below 120s limit

def _rect_from_width(W):
    H = 2.0 - W
    return W, H

def _objective(x, W, H):
    """
    x is shape (3*N,): [x1..xn, y1..yn, r1..rn]
    Returns sum of radii.
    """
    n = N
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]
    return np.sum(rs)

def _objective_grad(x, W, H):
    n = N
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]
    grad = np.zeros(3*n)
    grad[2*n:3*n] = 1.0
    return grad

def _constraints(x, W, H):
    """
    Returns list of constraint dicts.
    Each constraint: {'type': 'ineq', 'fun': f, 'jac': g}
    f(x) >= 0
    """
    n = N
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]
    
    cons = []
    
    # Boundary constraints:
    # x_i >= r_i, x_i <= W - r_i, y_i >= r_i, y_i <= H - r_i
    # We can combine these into 4 sets of N constraints.
    
    # 1. x_i - r_i >= 0
    def f_bx1(x):
        return x[:n] - x[2*n:3*n]
    def g_bx1(x):
        g = np.zeros(3*n)
        g[:n] = 1.0
        g[2*n:3*n] = -1.0
        return g
    cons.append({'type': 'ineq', 'fun': f_bx1, 'jac': g_bx1})
    
    # 2. W - x_i - r_i >= 0
    def f_bx2(x):
        return W - x[:n] - x[2*n:3*n]
    def g_bx2(x):
        g = np.zeros(3*n)
        g[:n] = -1.0
        g[2*n:3*n] = -1.0
        return g
    cons.append({'type': 'ineq', 'fun': f_bx2, 'jac': g_bx2})
    
    # 3. y_i - r_i >= 0
    def f_by1(x):
        return x[n:2*n] - x[2*n:3*n]
    def g_by1(x):
        g = np.zeros(3*n)
        g[n:2*n] = 1.0
        g[2*n:3*n] = -1.0
        return g
    cons.append({'type': 'ineq', 'fun': f_by1, 'jac': g_by1})
    
    # 4. H - y_i - r_i >= 0
    def f_by2(x):
        return H - x[n:2*n] - x[2*n:3*n]
    def g_by2(x):
        g = np.zeros(3*n)
        g[n:2*n] = -1.0
        g[2*n:3*n] = -1.0
        return g
    cons.append({'type': 'ineq', 'fun': f_by2, 'jac': g_by2})
    
    # Non-overlap constraints:
    # dist(i,j)^2 - (r_i + r_j)^2 >= 0
    # This is non-convex, but SLSQP can handle it if initialized well.
    # We will add these one by one or in a vectorized way?
    # SLSQP expects a list of dicts. We can create a single constraint function 
    # that returns a vector of all pair violations? 
    # No, standard SLSQP takes a list of constraint dicts. 
    # For efficiency, we can define a single constraint function that returns 
    # a vector of all pair distances squared minus radius sums squared.
    
    def f_pairs(x):
        xs = x[:n]
        ys = x[n:2*n]
        rs = x[2*n:3*n]
        vals = []
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx*dx + dy*dy
                rsum = rs[i] + rs[j]
                vals.append(d2 - rsum*rsum)
        return np.array(vals)
        
    def g_pairs(x):
        xs = x[:n]
        ys = x[n:2*n]
        rs = x[2*n:3*n]
        g = np.zeros((n*(n-1)//2, 3*n))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx*dx + dy*dy
                # Gradient of d2 - (ri+rj)^2
                # d(d2)/dxi = 2*dx, d(d2)/dxj = -2*dx
                # d((ri+rj)^2)/dri = 2*(ri+rj), d/d rj = 2*(ri+rj)
                # Note: if d2 is 0, gradient is 0, which is problematic. 
                # But constraint is d2 >= (ri+rj)^2.
                g[idx, i] = 2.0 * dx
                g[idx, n+i] = 2.0 * dy
                g[idx, 2*n+i] = -2.0 * (rs[i] + rs[j])
                
                g[idx, j] = -2.0 * dx
                g[idx, n+j] = -2.0 * dy
                g[idx, 2*n+j] = -2.0 * (rs[i] + rs[j])
                idx += 1
        return g
        
    cons.append({'type': 'ineq', 'fun': f_pairs, 'jac': g_pairs})
    
    # Radius positivity: r_i >= 0
    def f_pos(x):
        return x[2*n:3*n]
    def g_pos(x):
        g = np.zeros((n, 3*n))
        g[:, 2*n:3*n] = np.eye(n)
        return g
    cons.append({'type': 'ineq', 'fun': f_pos, 'jac': g_pos})
    
    return cons

def _is_valid(circles, w, h, tol=1e-6):
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    if np.any(rs < -tol):
        return False
    if np.any(xs < -rs - tol) or np.any(xs > w + rs + tol):
        return False
    if np.any(ys < -rs - tol) or np.any(ys > h + rs + tol):
        return False

    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy < (rs[i] + rs[j])**2 - tol:
                return False
    return True

def _grid_init(w, h, n_cols, n_rows):
    xs = (np.arange(n_cols) + 0.5) * (w / n_cols)
    ys = (np.arange(n_rows) + 0.5) * (h / n_rows)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X = X.flatten()
    Y = Y.flatten()
    X = X[:N]
    Y = Y[:N]
    r0 = min(w / n_cols, h / n_rows) / 2.0 * 0.9
    R = np.full(N, r0)
    return np.concatenate([X, Y, R])

def _random_init(w, h):
    xs = np.random.uniform(0.1, w-0.1, N)
    ys = np.random.uniform(0.1, h-0.1, N)
    rs = np.full(N, 0.1)
    return np.concatenate([xs, ys, rs])

def _force_init(w, h, iterations=100):
    """Simple force-directed initialization to spread circles."""
    xs = np.random.uniform(0.1, w-0.1, N)
    ys = np.random.uniform(0.1, h-0.1, N)
    rs = np.full(N, 0.1)
    
    for _ in range(iterations):
        # Repulsion between circles
        for i in range(N):
            fx = 0
            fy = 0
            for j in range(N):
                if i == j: continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-5:
                    fx += np.random.uniform(-1, 1)
                    fy += np.random.uniform(-1, 1)
                    continue
                # Force proportional to 1/d
                f = 1.0 / d
                fx += f * dx
                fy += f * dy
            
            # Attraction to center (weak) to keep them in box
            fx += 0.01 * (w/2 - xs[i])
            fy += 0.01 * (h/2 - ys[i])
            
            # Boundary repulsion
            if xs[i] < rs[i]: fx -= 1.0
            if xs[i] > w - rs[i]: fx += 1.0
            if ys[i] < rs[i]: fy -= 1.0
            if ys[i] > h - rs[i]: fy += 1.0
            
            xs[i] += 0.05 * fx
            ys[i] += 0.05 * fy
            
            # Keep in bounds
            xs[i] = np.clip(xs[i], rs[i], w - rs[i])
            ys[i] = np.clip(ys[i], rs[i], h - rs[i])
            
    return np.concatenate([xs, ys, rs])

def construct():
    t0 = time.time()
    best_circles = None
    best_val = -np.inf
    
    # Aspect ratios to try
    # w + h = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.1, 0.9]
    
    for ar in aspect_ratios:
        if time.time() - t0 > DEADLINE:
            break
        
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        # Generate multiple initializations
        inits = []
        
        # Grids
        grid_configs = [(5, 5), (6, 4), (4, 6), (7, 4), (4, 7)]
        for nc, nr in grid_configs:
            if nc * nr >= N:
                inits.append(_grid_init(w, h, nc, nr))
                
        # Random
        inits.append(_random_init(w, h))
        inits.append(_random_init(w, h))
        
        # Force directed
        inits.append(_force_init(w, h))
        
        for x0 in inits:
            if time.time() - t0 > DEADLINE:
                break
            
            # Optimize
            try:
                # SLSQP is slower but handles constraints better than L-BFGS-B with penalty
                # We use analytic gradients
                cons = _constraints(x0, w, h)
                
                res = minimize(
                    _objective, x0,
                    args=(w, h),
                    method='SLSQP',
                    jac=_objective_grad,
                    bounds=[(0, w)] * N + [(0, h)] * N + [(0, 1.0)] * N,
                    constraints=cons,
                    options={'maxiter': 100, 'ftol': 1e-8, 'disp': 0}
                )
                
                if res.success or res.status == 0:
                    circles = res.x.reshape(N, 3)
                    # Clip to valid bounds
                    circles[:, 0] = np.clip(circles[:, 0], 0, w)
                    circles[:, 1] = np.clip(circles[:, 1], 0, h)
                    circles[:, 2] = np.maximum(0.0, circles[:, 2])
                    
                    # Ensure strict validity (SLSQP might leave tiny violations)
                    # Re-clip radii to boundary distances
                    for i in range(N):
                        max_r = min(circles[i, 0], circles[i, 1], w - circles[i, 0], h - circles[i, 1])
                        circles[i, 2] = min(circles[i, 2], max_r)
                    
                    if _is_valid(circles, w, h):
                        val = np.sum(circles[:, 2])
                        if val > best_val:
                            best_val = val
                            best_circles = circles.copy()
                            
            except Exception:
                continue
                
    # Fallback: simple grid
    if best_circles is None:
        w = h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        
    return best_circles
