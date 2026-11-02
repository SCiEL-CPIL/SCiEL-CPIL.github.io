"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    def is_valid(x, w, h, tol=1e-6):
        """Check if configuration is strictly valid with tolerance."""
        n = len(x) // 3
        # Check bounds
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9:
                return False
            if xi - ri < -tol or xi + ri > w + tol or yi - ri < -tol or yi + ri > h + tol:
                return False
        
        # Check disjointness
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True

    def get_delaunay_pairs(centers):
        """Get Delaunay triangulation edges as a set of pairs."""
        try:
            tri = Delaunay(centers)
            pairs = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        a, b = sorted([simplex[i], simplex[j]])
                        pairs.add((a, b))
            return pairs
        except Exception:
            # Fallback: all pairs
            return set((i, j) for i in range(N) for j in range(i+1, N))

    def softplus_penalty(v, beta=50.0):
        """Smooth approximation of max(0, v)."""
        return np.logaddexp(0, beta * v) / beta

    def objective_and_grad(params, w, h, pairs):
        """Objective: minimize penalty - sum(radii). Returns (value, gradient)."""
        xs = params[0::3].copy()
        ys = params[1::3].copy()
        rs = params[2::3].copy()
        
        # Ensure positive radii
        rs = np.maximum(rs, 1e-8)
        
        val = -np.sum(rs)
        grad = np.zeros(3*N)
        
        # Boundary penalties
        for i in range(N):
            # Left boundary
            v = -(xs[i] - rs[i])
            if v > 0:
                p = softplus_penalty(v)
                val += p
                # d/dxs: -beta*sigmoid(beta*v) * (-1) = beta*sigmoid(beta*v)
                # d/drs: -beta*sigmoid(beta*v) * (1) = -beta*sigmoid(beta*v)
                sig = 1.0 / (1.0 + np.exp(-50.0 * v))
                grad[3*i] += 50.0 * sig
                grad[3*i+2] -= 50.0 * sig
            
            # Right boundary
            v = (xs[i] + rs[i]) - w
            if v > 0:
                p = softplus_penalty(v)
                val += p
                sig = 1.0 / (1.0 + np.exp(-50.0 * v))
                grad[3*i] += 50.0 * sig
                grad[3*i+2] += 50.0 * sig
            
            # Bottom boundary
            v = -(ys[i] - rs[i])
            if v > 0:
                p = softplus_penalty(v)
                val += p
                sig = 1.0 / (1.0 + np.exp(-50.0 * v))
                grad[3*i+1] += 50.0 * sig
                grad[3*i+2] -= 50.0 * sig
            
            # Top boundary
            v = (ys[i] + rs[i]) - h
            if v > 0:
                p = softplus_penalty(v)
                val += p
                sig = 1.0 / (1.0 + np.exp(-50.0 * v))
                grad[3*i+1] += 50.0 * sig
                grad[3*i+2] += 50.0 * sig
        
        # Overlap penalties (only for Delaunay pairs + nearby circles)
        # First, get current Delaunay pairs, but also include any pairs that are actually overlapping
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) not in pairs:
                    # Check if they might be overlapping (cheap check)
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    if dist_sq >= (min_dist * 1.01)**2:
                        continue  # Definitely not overlapping and not in Delaunay
        
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                min_dist = rs[i] + rs[j]
                
                v = min_dist - dist
                if v > 0:
                    p = softplus_penalty(v)
                    val += p
                    sig = 1.0 / (1.0 + np.exp(-50.0 * v))
                    force = 50.0 * sig
                    
                    # Gradient w.r.t. positions: force * (dx/dist, dy/dist)
                    # If circles overlap, push them apart
                    if dist > 1e-8:
                        grad[3*i] += force * dx / dist
                        grad[3*i+1] += force * dy / dist
                        grad[3*j] -= force * dx / dist
                        grad[3*j+1] -= force * dy / dist
                    
                    # Gradient w.r.t. radii: force for each
                    grad[3*i+2] += force
                    grad[3*j+2] += force
        
        # Gradient for radii from objective: -1 for each radius
        for i in range(N):
            grad[3*i+2] -= 1.0
        
        return val, grad

    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_w, best_h
        if time.time() > deadline:
            return
        if is_valid(x, w, h, tol=1e-5):
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()
                best_w = w
                best_h = h

    # Loop over aspect ratios
    widths = np.linspace(0.7, 1.5, 12)
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize with a grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        centers = np.zeros((N, 2))
        for i in range(N):
            col = i % a
            row = i // a
            centers[i][0] = (col + 0.5) * w / a
            centers[i][1] = (row + 0.5) * h / b
            
        r_est = 0.5 * min(w/a, h/b) * 0.85
        r_est = min(r_est, w/2, h/2)
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = centers[i][0]
            x0[3*i+1] = centers[i][1]
            x0[3*i+2] = r_est
        
        # Get initial Delaunay pairs
        pairs = get_delaunay_pairs(centers)
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        
        try:
            # Use custom objective with gradient
            def obj(params):
                v, _ = objective_and_grad(params, w, h, pairs)
                return v
            
            def jac(params):
                _, g = objective_and_grad(params, w, h, pairs)
                return g
            
            res = minimize(obj, x0, jac=jac, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxcor': 10})
            
            if time.time() > deadline:
                break
            
            # Update Delaunay pairs and do a second optimization pass
            if res.success or res.fun < 1.0:
                centers_new = np.column_stack([res.x[0::3], res.x[1::3]])
                pairs_new = get_delaunay_pairs(centers_new)
                
                # Second pass with updated pairs
                def obj2(params):
                    v, _ = objective_and_grad(params, w, h, pairs_new)
                    return v
                
                def jac2(params):
                    _, g = objective_and_grad(params, w, h, pairs_new)
                    return g
                
                res2 = minimize(obj2, res.x, jac=jac2, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxcor': 10})
                
                if time.time() > deadline:
                    break
                
                # Use the better result
                if res2.fun < res.fun:
                    res = res2
                
                check_and_update(res.x, w, h)
            else:
                check_and_update(res.x, w, h)
                
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.09
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
        best_w = w
        best_h = h

    # Final strict validation and correction
    if not is_valid(best_result, best_w, best_h, tol=1e-7):
        for i in range(N):
            best_result[3*i+2] *= 0.99
        
        if not is_valid(best_result, best_w, best_h, tol=1e-7):
            a, b = 7, 3
            w, h = 1.4, 0.6
            r = 0.05
            best_result = np.zeros(3*N)
            for i in range(a):
                for j in range(b):
                    idx = i*b + j
                    best_result[3*idx] = (i + 0.5) * w / a
                    best_result[3*idx+1] = (j + 0.5) * h / b
                    best_result[3*idx+2] = r
            best_w = w
            best_h = h

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
