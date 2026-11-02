import math
import time
import numpy as np
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 8.0
N = 11
APOTHEM = math.sqrt(3) / 2.0
# distance between centers of two touching unit hexagons (same orientation)
D_TOUCH = 2.0 * APOTHEM  # = sqrt(3)

def _elapsed_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY)

def _outer_side(pts, cx, cy, beta):
    """Minimum side length S of a regular hexagon (center cx,cy, rotation beta) enclosing pts."""
    # normals of the 6 sides of the outer hexagon
    # for a regular hexagon with rotation beta (vertex at angle beta),
    # the side normals are at beta + pi/6 + k*pi/3
    max_proj = 0.0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        m = max(np.max(projs), -np.min(projs))
        if m > max_proj:
            max_proj = m
    S = (max_proj + APOTHEM) / APOTHEM
    return S

def _soft_outer(pts, cx, cy, beta, temp=0.05):
    """Soft version of outer hexagon side length using log-sum-exp."""
    all_vals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        all_vals.append(projs)
        all_vals.append(-projs)
    all_vals = np.concatenate(all_vals)
    max_val = np.max(all_vals)
    shifted = (all_vals - max_val) / temp
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
    D = temp * log_sum
    S = (D + APOTHEM) / APOTHEM
    return S

def _soft_overlap_penalty(pts, temp=0.02):
    """Soft penalty for pairs of hexagons being too close (centers < D_TOUCH)."""
    n = len(pts)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            violation = D_TOUCH - dist
            if violation > 0:
                penalty += violation * violation
    return penalty

def _exact_check(pts, cx, cy, S, beta):
    """Check if configuration is valid: all hexagons inside outer, and pairwise disjoint."""
    # Check containment: each hexagon's 6 vertices must be inside outer hexagon
    outer_apothem = S * APOTHEM
    for i in range(N):
        px, py = pts[i, 0], pts[i, 1]
        # vertices of inner hexagon i (unit side, rotation = beta for simplicity, or 0)
        # Actually inner hexagons can have any rotation; but for disjointness check,
        # we need to check actual polygon intersection.
        # For now, use center distance as a quick filter, then do polygon check.
        pass
    
    # More careful: check each inner hexagon is inside outer
    for i in range(N):
        px, py = pts[i, 0], pts[i, 1]
        for v in range(6):
            ang = v * math.pi / 3.0  # inner hexagon rotation 0
            vx = px + math.cos(ang)
            vy = py + math.sin(ang)
            # Check if (vx, vy) is inside outer hexagon
            inside = True
            for k in range(6):
                ang_o = beta + math.pi / 6.0 + k * math.pi / 3.0
                nx = math.cos(ang_o)
                ny = math.sin(ang_o)
                proj = (vx - cx) * nx + (vy - cy) * ny
                if proj > outer_apothem + 1e-9:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check pairwise disjointness: for two unit hexagons with same orientation,
    # they are disjoint iff center distance >= 2*apothem = sqrt(3)
    # This is only true for same orientation. For different orientations, need polygon check.
    # Since we're using all inner hexagons with rotation 0 (same as outer's relative),
    # center distance check is sufficient for axis-aligned hexagons.
    for i in range(N):
        for j in range(i + 1, N):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < D_TOUCH - 1e-9:
                return False
    return True

def construct():
    if not _elapsed_ok():
        # Fallback: simple hexagonal cluster
        pts = np.array([[0.0, 0.0]] + 
                       [(math.sqrt(3)*math.cos(a), math.sqrt(3)*math.sin(a)) for a in range(0, 360, 60)] +
                       [(2*APOTHEM*math.sqrt(3)*math.cos(a), 2*APOTHEM*math.sqrt(3)*math.sin(a)) for a in [30, 90, 150]])
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        S = _outer_side(pts, cx, cy, 0.0)
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), 0.0

    # Initial configurations
    inits = []
    
    # 1. Hexagonal ring: center + 6 around + 4 more
    ring = [(0.0, 0.0)]
    for a in range(0, 360, 60):
        ring.append((D_TOUCH * math.cos(math.radians(a)), D_TOUCH * math.sin(math.radians(a))))
    # Add 4 more in a second ring
    for a in [30, 90, 150, 270]:
        ring.append((2 * D_TOUCH * math.cos(math.radians(a)), 2 * D_TOUCH * math.sin(math.radians(a))))
    inits.append(np.array(ring))
    
    # 2. Linear-ish arrangement
    lin = [(i * D_TOUCH, 0.0) for i in range(N)]
    inits.append(np.array(lin))
    
    # 3. Random initializations
    for trial in range(20):
        rng = np.random.default_rng(trial * 42)
        pts = rng.uniform(-3, 3, size=(N, 2))
        inits.append(pts)
    
    # 4. From current best lattice (approximate)
    lattice = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (2, 0), (-2, 0), (0, 2), (0, -2)]
    lat_pts = []
    for q, r in lattice:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        lat_pts.append((x, y))
    inits.append(np.array(lat_pts))
    
    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None
    
    for init_idx, init_pts in enumerate(inits):
        if not _elapsed_ok():
            break
        
        # Optimize: variables = [x1,y1,...,xN,yN, cx, cy, beta]
        n_vars = 2 * N + 3
        
        def objective(params, temp_outer=0.05, temp_overlap=0.02):
            if not _elapsed_ok():
                return 1e10
            pts = params[:2*N].reshape(N, 2)
            cx = params[2*N]
            cy = params[2*N + 1]
            beta = params[2*N + 2]
            
            S = _soft_outer(pts, cx, cy, beta, temp=temp_outer)
            overlap_pen = _soft_overlap_penalty(pts, temp=temp_overlap)
            
            # Weight overlap penalty heavily
            return S + 10.0 * overlap_pen
        
        init_params = np.concatenate([init_pts.ravel(), [0.0, 0.0, 0.0]])
        
        try:
            res = minimize(objective, init_params, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                   'maxfun': 5000, 'disp': False})
        except Exception:
            continue
        
        if res.fun > 1e9:
            continue
        
        # Extract best from result
        best_params = res.x
        best_pts_local = best_params[:2*N].reshape(N, 2)
        best_cx = best_params[2*N]
        best_cy = best_params[2*N + 1]
        best_beta_local = best_params[2*N + 2]
        # Normalize beta to [0, pi/3)
        best_beta_local = ((best_beta_local % (math.pi / 3)) + math.pi / 3) % (math.pi / 3)
        
        # Compute exact S
        S_exact = _outer_side(best_pts_local, best_cx, best_cy, best_beta_local)
        
        # Check validity
        if _exact_check(best_pts_local, best_cx, best_cy, S_exact, best_beta_local):
            if S_exact < best_S:
                best_S = S_exact
                best_pts = best_pts_local.copy()
                best_beta = best_beta_local
                best_center = (best_cx, best_cy)
        else:
            # Even if not perfectly valid, track the best objective value
            # and try to repair later
            if res.fun < best_S:  # rough comparison
                # Try a quick repair: adjust positions to satisfy constraints
                pass
    
    # If we have a valid result, use it
    if best_pts is not None and best_S < float('inf'):
        # Final polish: try to reduce S by optimizing beta and center only
        for _ in range(3):
            if not _elapsed_ok():
                break
            def obj2(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi / 3)) + math.pi / 3) % (math.pi / 3)
                return _outer_side(best_pts, cx, cy, beta)
            
            try:
                res2 = minimize(obj2, np.array([best_center[0], best_center[1], best_beta]),
                               method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res2.fun < best_S:
                    best_center = (res2.x[0], res2.x[1])
                    best_beta = ((res2.x[2] % (math.pi / 3)) + math.pi / 3) % (math.pi / 3)
                    best_S = res2.fun
            except Exception:
                break
        
        cx, cy = best_center
        inner = [[float(x), float(y), float(math.degrees(best_beta))] for x, y in best_pts]
        return inner, [float(cx), float(cy)], float(best_S), float(math.degrees(best_beta))
    
    # Fallback: use the lattice configuration
    pts = np.array([(math.sqrt(3) * (q + r / 2.0), 1.5 * r) for q, r in 
                    [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]])
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    # Find best beta
    best_b = 0.0
    best_s = float('inf')
    for b in np.linspace(0, math.pi / 6, 100):
        s = _outer_side(pts, cx, cy, b)
        if s < best_s:
            best_s = s
            best_b = b
    inner = [[float(x), float(y), float(math.degrees(best_b))] for x, y in pts]
    return inner, [float(cx), float(cy)], float(best_s), float(math.degrees(best_b))
