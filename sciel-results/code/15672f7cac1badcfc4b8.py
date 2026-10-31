"""Circle packing in perimeter-4 rectangle: Fixed-center inflation + SLSQP polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 21
START = time.time()
DEADLINE = START + 145.0

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    if W <= 0.1 or H <= 0.1:
        return None
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _inflation_solve(cx, cy, W, H):
    """
    Solve for maximum radii given fixed centers.
    This is a convex optimization problem:
    Maximize sum(r_i)
    Subject to:
      r_i + r_j <= dist(i,j)
      r_i <= min(cx_i, W-cx_i, cy_i, H-cy_i)
      r_i >= 0
    """
    n = len(cx)
    # Build adjacency or just use all pairs for simplicity since N=21 is small
    # Variables: r (n,)
    # Objective: -sum(r)
    
    # Constraints for SLSQP:
    # 1. Pairwise: dist(i,j) - r_i - r_j >= 0
    # 2. Boundary: min_dist_to_wall_i - r_i >= 0
    
    cons = []
    bounds = []
    
    # Precompute distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
            dists[i,j] = d
            dists[j,i] = d
    
    # Precompute wall distances
    wall_dists = np.minimum(np.minimum(cx, cy), np.minimum(W-cx, H-cy))
    
    # Bounds
    for i in range(n):
        bounds.append((1e-6, wall_dists[i]))
        
    # Constraints
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            if dists[i,j] < 1e-6:
                continue
            di, dj = i, j
            def constraint(x, di=di, dj=dj):
                return dists[di, dj] - x[di] - x[dj]
            cons.append({'type': 'ineq', 'fun': constraint})
            
    # Wall constraints
    for i in range(n):
        wi = i
        def wall_constraint(x, wi=wi):
            return wall_dists[wi] - x[wi]
        cons.append({'type': 'ineq', 'fun': wall_constraint})
        
    # Initial guess
    r0 = np.ones(n) * 0.05
    r0 = np.minimum(r0, wall_dists)
    
    try:
        res = minimize(
            lambda r: -np.sum(r),
            r0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        if res.success:
            r_opt = np.maximum(res.x, 1e-6)
            # Verify
            valid = True
            for i in range(n):
                if r_opt[i] > wall_dists[i] + 1e-6:
                    valid = False
                    break
                for j in range(i+1, n):
                    if r_opt[i] + r_opt[j] > dists[i,j] + 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return r_opt
    except Exception:
        pass
        
    # Fallback: simple iterative clipping
    r = np.ones(n) * 0.1
    r = np.minimum(r, wall_dists)
    for _ in range(100):
        for i in range(n):
            # Clip against walls
            r[i] = min(r[i], wall_dists[i])
            # Clip against neighbors
            for j in range(n):
                if i != j:
                    max_r = dists[i,j] - r[j]
                    if r[i] > max_r:
                        r[i] = max(max_r, 0)
    r = np.maximum(r, 1e-6)
    return r

def _generate_grid_centers(W, H, kx, ky):
    """Generate a kx by ky grid of centers."""
    xs = np.linspace(W * 0.1, W * 0.9, kx) if kx > 1 else np.array([W*0.5])
    ys = np.linspace(H * 0.1, H * 0.9, ky) if ky > 1 else np.array([H*0.5])
    
    cx = []
    cy = []
    for y in ys:
        for x in xs:
            cx.append(x)
            cy.append(y)
            
    # If we have too many, subsample
    if len(cx) > N:
        # Take first N or random sample
        idx = np.random.choice(len(cx), N, replace=False)
        cx = np.array(cx)[idx]
        cy = np.array(cy)[idx]
    elif len(cx) < N:
        # Pad with random points
        while len(cx) < N:
            cx.append(random.uniform(0.1*W, 0.9*W))
            cy.append(random.uniform(0.1*H, 0.9*H))
            
    return np.array(cx), np.array(cy)

def _force_relax_unified(x0, W, H, max_iter=500):
    """
    Force-based relaxation that moves centers and adjusts radii simultaneously.
    """
    x = x0.copy()
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    
    best_sum = _evaluate(x, W, H) or 0.0
    best_x = x.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        # Compute forces
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                if d < 1e-6:
                    dx = random.uniform(-1e-3, 1e-3)
                    dy = random.uniform(-1e-3, 1e-3)
                    d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req:
                    # Repulsive force
                    strength = (req - d) * 0.5
                    Fx[i] += strength * dx / d
                    Fy[i] += strength * dy / d
                    Fx[j] -= strength * dx / d
                    Fy[j] -= strength * dy / d
                    
        # Wall forces
        for i in range(N):
            if cx[i] < r[i]:
                Fx[i] += (r[i] - cx[i]) * 0.5
            if W - cx[i] < r[i]:
                Fx[i] -= (W - cx[i] - r[i]) * 0.5
            if cy[i] < r[i]:
                Fy[i] += (r[i] - cy[i]) * 0.5
            if H - cy[i] < r[i]:
                Fy[i] -= (H - cy[i] - r[i]) * 0.5
                
        # Apply forces
        step = 0.01 * (1 - it / max_iter)
        cx += step * Fx
        cy += step * Fy
        
        # Clamp
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        
        # Re-solve radii for these centers
        r_new = _inflation_solve(cx, cy, W, H)
        
        x_new = np.column_stack([cx, cy, r_new])
        s_new = _evaluate(x_new, W, H)
        
        if s_new is not None and s_new > best_sum:
            best_sum = s_new
            best_x = x_new.copy()
            x = x_new.copy()
            r = r_new.copy()
        else:
            # Try smaller step
            x = np.column_stack([cx, cy, r])
            
    return best_x, best_sum

def _slsqp_full(x0, W0, H0, maxiter=300):
    """
    Full SLSQP optimization over centers, radii, and W.
    """
    n = N
    # Vector: [W, cx_0..cx_n, cy_0..cy_n, r_0..r_n]
    x = np.zeros(1 + 3*n)
    x[0] = W0
    x[1:1+n] = x0[:, 0]
    x[1+n:1+2*n] = x0[:, 1]
    x[1+2*n:1+3*n] = x0[:, 2]
    
    def objective(x):
        return -np.sum(x[1+2*n:1+3*n])
        
    cons = []
    # W bounds
    cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.3})
    cons.append({'type': 'ineq', 'fun': lambda x: 1.7 - x[0]})
    
    # Boundary constraints
    for i in range(n):
        cx_i = 1 + i
        cy_i = 1 + n + i
        r_i = 1 + 2*n + i
        
        # cx >= r
        cons.append({'type': 'ineq', 'fun': lambda x, cx=cx_i, r=r_i: x[cx] - x[r]})
        # W - cx >= r
        cons.append({'type': 'ineq', 'fun': lambda x, w=0, cx=cx_i, r=r_i: x[w] - x[cx] - x[r]})
        # cy >= r
        cons.append({'type': 'ineq', 'fun': lambda x, cy=cy_i, r=r_i: x[cy] - x[r]})
        # H - cy >= r  => (2-W) - cy - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, w=0, cy=cy_i, r=r_i: (2.0 - x[w]) - x[cy] - x[r]})
        
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            cxi = 1 + i
            cxj = 1 + j
            cyi = 1 + n + i
            cyj = 1 + n + j
            ri = 1 + 2*n + i
            rj = 1 + 2*n + j
            
            def non_overlap(x, cxi=cxi, cxj=cxj, cyi=cyi, cyj=cyj, ri=ri, rj=rj):
                dx = x[cxi] - x[cxj]
                dy = x[cyi] - x[cyj]
                return dx*dx + dy*dy - (x[ri] + x[rj])**2
                
            cons.append({'type': 'ineq', 'fun': non_overlap})
            
    bounds = [(0.3, 1.7)]
    for i in range(n):
        bounds.append((0.0, 2.0)) # cx
    for i in range(n):
        bounds.append((0.0, 2.0)) # cy
    for i in range(n):
        bounds.append((1e-4, 0.9)) # r
        
    try:
        res = minimize(
            objective,
            x,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-8, 'disp': False}
        )
        x_opt = res.x
        W_opt = x_opt[0]
        H_opt = 2.0 - W_opt
        cx = x_opt[1:1+n]
        cy = x_opt[1+n:1+2*n]
        r = x_opt[1+2*n:1+3*n]
        
        out = np.column_stack([cx, cy, r])
        s = _evaluate(out, W_opt, H_opt)
        if s is not None:
            return out, W_opt, H_opt, s
    except Exception:
        pass
        
    return None, W0, H0, 0.0

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE - 5
        
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Grid-based Inflation
    # Try various aspect ratios and grid configurations
    Ws = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    grid_configs = [(3,3), (4,3), (3,4), (5,4), (4,5), (5,5), (6,4), (4,6)]
    
    seeds = []
    for W in Ws:
        H = 2.0 - W
        if H < 0.3 or W < 0.3:
            continue
        for kx, ky in grid_configs:
            if kx * ky < N:
                continue
            cx, cy = _generate_grid_centers(W, H, kx, ky)
            r = _inflation_solve(cx, cy, W, H)
            x = np.column_stack([cx, cy, r])
            s = _evaluate(x, W, H)
            if s is not None:
                seeds.append((s, W, H, x))
                
    # Sort seeds
    seeds.sort(key=lambda t: -t[0])
    
    # Phase 2: Force Relaxation on top seeds
    top_k = min(15, len(seeds))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, x0 = seeds[idx]
        x_new, s_new = _force_relax_unified(x0, W, H, max_iter=200)
        if s_new > best_sum:
            best_sum = s_new
            best_arr = x_new
            best_W = W
            best_H = H
            
    # Phase 3: SLSQP Polish on best
    if time_ok() and best_arr is not None:
        x_slsqp, W_slsqp, H_slsqp, s_slsqp = _slsqp_full(best_arr, best_W, best_H, maxiter=200)
        if s_slsqp > best_sum:
            best_sum = s_slsqp
            best_arr = x_slsqp
            best_W = W_slsqp
            best_H = H_slsqp
            
    # Phase 4: More Force Relaxation on best (with perturbation)
    if time_ok() and best_arr is not None:
        for _ in range(3):
            if not time_ok():
                break
            x_perturbed = best_arr.copy()
            x_perturbed[:, 0] += np.random.randn(N) * 0.02
            x_perturbed[:, 1] += np.random.randn(N) * 0.02
            x_perturbed[:, 0] = np.clip(x_perturbed[:, 0], 0.0, best_W)
            x_perturbed[:, 1] = np.clip(x_perturbed[:, 1], 0.0, best_H)
            
            x_new, s_new = _force_relax_unified(x_perturbed, best_W, best_H, max_iter=100)
            if s_new > best_sum:
                best_sum = s_new
                best_arr = x_new
                
    # Final SLSQP polish
    if time_ok() and best_arr is not None:
        x_slsqp, W_slsqp, H_slsqp, s_slsqp = _slsqp_full(best_arr, best_W, best_H, maxiter=100)
        if s_slsqp > best_sum:
            best_sum = s_slsqp
            best_arr = x_slsqp
            best_W = W_slsqp
            best_H = H_slsqp
            
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        cx, cy = _generate_grid_centers(W, H, 5, 5)
        r = _inflation_solve(cx, cy, W, H)
        best_arr = np.column_stack([cx, cy, r])
        best_W, best_H = W, H
        
    # Final clamping
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Emergency fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return out

