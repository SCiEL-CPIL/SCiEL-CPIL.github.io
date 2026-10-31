"""
Circle packing in perimeter-4 rectangle.

Combines explicit SLSQP constraints (from inspirations) for robustness against 
overlaps with force-directed relaxation (from current program) for initialization 
and repair. This avoids the pitfalls of penalty methods where constraint 
violations might be tolerated to increase radii.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # Safety margin for 150s limit

def _force_relax(cx, cy, r, W, H, max_iter=200, dt=0.002):
    """
    Fast force-directed relaxation to remove overlaps and wall violations.
    Returns updated cx, cy.
    """
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    
    if W <= 0 or H <= 0:
        return cx, cy
        
    # Ensure initial validity roughly
    cx = np.clip(cx, r, W - r)
    cy = np.clip(cy, r, H - r)
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces (strong)
        # Left
        viol = r - cx
        mask = viol > 0
        fx[mask] += viol[mask] * 50.0
        # Right
        viol = cx + r - W
        mask = viol > 0
        fx[mask] -= viol[mask] * 50.0
        # Bottom
        viol = r - cy
        mask = viol > 0
        fy[mask] += viol[mask] * 50.0
        # Top
        viol = cy + r - H
        mask = viol > 0
        fy[mask] -= viol[mask] * 50.0
        
        # Pairwise overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d2 = dx*dx + dy*dy
                min_d = r[i] + r[j]
                if d2 < min_d*min_d:
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-6
                    overlap = min_d - d
                    ux = dx / d
                    uy = dy / d
                    # Repulsive force
                    force = overlap * 10.0
                    fx[i] += ux * force
                    fy[i] += uy * force
                    fx[j] -= ux * force
                    fy[j] -= uy * force
        
        cx += dt * fx
        cy += dt * fy
        
        # Clip to valid region to prevent explosion
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
    return cx, cy

def _is_feasible(W, H, cx, cy, r, tol=1e-5):
    """Check if the configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if np.any(r < -tol):
        return False
    # Walls
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    # Overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < (r[i] + r[j] - tol)**2:
                return False
    return True

def _generate_seed(W, H, layout='grid'):
    """
    Generate initial x vector: [r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]
    """
    x = np.zeros(3 * N)
    
    if layout == 'hex':
        # Hexagonal packing approximation
        # Try to fit a hex grid
        # Spacing s, radius r = s/2
        # Width = (cols - 0.5) * s * sqrt(3) ? No, hex is complex. 
        # Let's stick to a dense grid for simplicity and reliability, 
        # but jittered to break symmetry.
        pass
        
    # Grid layout with jitter
    cols = int(np.ceil(np.sqrt(N * (W/H))))
    rows = int(np.ceil(N / cols))
    
    # Adjust cols/rows to better fit aspect ratio
    if W > H:
        # Wider
        cols = min(N, max(1, int(np.sqrt(N * W / H))))
    else:
        cols = min(N, max(1, int(np.sqrt(N * W / H))))
        
    rows = int(np.ceil(N / cols))
    
    cw = W / cols
    ch = H / rows
    
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    idx = 0
    for i in range(cols):
        for j in range(rows):
            if idx >= N:
                break
            cx[idx] = (i + 0.5) * cw
            cy[idx] = (j + 0.5) * ch
            # Radius is half the min cell size, shrunk slightly
            r[idx] = 0.5 * min(cw, ch) * 0.95
            idx += 1
            
    # If we didn't fill all, or if cols*rows < N (shouldn't happen with ceil), 
    # ensure we have N points.
    if idx < N:
        # Fallback: just place them in a line if grid failed? 
        # Unlikely with the above logic.
        pass

    x[:N] = r
    x[N:2*N] = cx
    x[2*N:3*N] = cy
    
    return x

def _make_constraints(W, H):
    """
    Create SLSQP constraints.
    x layout: [r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]
    """
    cons = []
    
    # 1. Radii must be positive
    # r_i >= 1e-4
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i] - 1e-4
        })
        
    # 2. Wall constraints
    # cx_i - r_i >= 0
    # W - cx_i - r_i >= 0
    # cy_i - r_i >= 0
    # H - cy_i - r_i >= 0
    for i in range(N):
        # Left
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[N+i] - x[i]
        })
        # Right
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (W - x[N+i]) - x[i]
        })
        # Bottom
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2*N+i] - x[i]
        })
        # Top
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (H - x[2*N+i]) - x[i]
        })
        
    # 3. Non-overlap constraints
    # (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= (r_i + r_j)^2
    # => (cx_i - cx_j)^2 + (cy_i - cy_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[N+i] - x[N+j])**2 + 
                    (x[2*N+i] - x[2*N+j])**2 - 
                    (x[i] + x[j])**2
                )
            })
            
    return cons

def _objective(x):
    """
    Maximize sum of radii => Minimize negative sum of radii.
    x layout: [r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]
    """
    r = x[:N]
    return -np.sum(r)

def construct():
    start_time = time.time()
    best_val = -float('inf')
    best_result = None
    
    # We will vary W (width). H = 2.0 - W.
    # Due to symmetry, we only need to check W in [0.5, 1.5] or just [0.5, 1.0] 
    # and swap if needed, but the problem is symmetric in W/H for the value.
    # Let's sample several aspect ratios.
    
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1, 0.6, 1.4]
    
    for W in aspect_ratios:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        H = 2.0 - W
        if H <= 0.1 or W <= 0.1:
            continue
            
        # 1. Generate seed
        x0 = _generate_seed(W, H)
        
        # 2. Pre-relax to ensure feasibility for SLSQP
        r0 = x0[:N].copy()
        cx0 = x0[N:2*N].copy()
        cy0 = x0[2*N:3*N].copy()
        
        # Relax positions only, keeping radii fixed initially to find a valid layout
        cx_rel, cy_rel = _force_relax(cx0, cy0, r0, W, H, max_iter=300)
        
        x0[N:2*N] = cx_rel
        x0[2*N:3*N] = cy_rel
        
        # 3. Define constraints and bounds
        cons = _make_constraints(W, H)
        
        # Bounds:
        # r_i in [1e-4, min(W,H)/2]
        # cx_i in [0, W]
        # cy_i in [0, H]
        bounds = []
        max_r = min(W, H) / 2.0
        for i in range(N):
            bounds.append((1e-4, max_r))
        for i in range(N):
            bounds.append((0.0, W))
        for i in range(N):
            bounds.append((0.0, H))
            
        # 4. Optimize
        try:
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            x_opt = res.x
            r_opt = x_opt[:N]
            cx_opt = x_opt[N:2*N]
            cy_opt = x_opt[2*N:3*N]
            
            # Check feasibility with tight tolerance
            if _is_feasible(W, H, cx_opt, cy_opt, r_opt, tol=1e-4):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_result = (cx_opt.copy(), cy_opt.copy(), r_opt.copy())
            else:
                # If SLSQP result is slightly infeasible, try to repair with relaxation
                # Note: Relaxation moves centers, might change feasibility relative to radii
                cx_rep, cy_rep = _force_relax(cx_opt, cy_opt, r_opt, W, H, max_iter=100)
                if _is_feasible(W, H, cx_rep, cy_rep, r_opt, tol=1e-4):
                    val = np.sum(r_opt)
                    if val > best_val:
                        best_val = val
                        best_result = (cx_rep.copy(), cy_rep.copy(), r_opt.copy())
                        
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        # Simple grid in 1x1 square (perimeter 4)
        # 21 circles in 1x1. 
        # 5x5 grid = 25 spots. Use 21.
        # Side length 1. Cell size 0.2. Radius 0.1.
        # Sum radii = 21 * 0.1 = 2.1.
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3 # Slight shrink
                cells.append((cx, cy, r))
        return np.array(cells[:N])
        
    cx, cy, r = best_result
    return np.column_stack([cx, cy, r])
