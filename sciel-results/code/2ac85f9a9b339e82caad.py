"""Circle packing in perimeter-4 rectangle: vectorized force-directed + SLSQP with diverse seeds."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0

def _evaluate(cx, cy, r, W, H, tol=1e-7):
    """Return sum of radii if valid, else None."""
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + tol < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _vectorized_forces(cx, cy, r, W, H, wall_k=30.0, pair_k=15.0):
    """Compute repulsion forces vectorized."""
    n = len(cx)
    fx = np.zeros(n)
    fy = np.zeros(n)
    
    # Wall forces
    fx += np.where(cx - r < 0, (cx - r) * wall_k, 0.0)
    fx += np.where(cx + r > W, -(cx + r - W) * wall_k, 0.0)
    fy += np.where(cy - r < 0, (cy - r) * wall_k, 0.0)
    fy += np.where(cy + r > H, -(cy + r - H) * wall_k, 0.0)
    
    # Pairwise forces
    for i in range(n):
        dx = cx[i] - cx
        dy = cy[i] - cy
        d = np.sqrt(dx * dx + dy * dy)
        d[i] = 1e12
        overlap = r[i] + r - d
        mask = (overlap > 0) & (d > 1e-12)
        if np.any(mask):
            fx[i] += np.sum(dx[mask] / d[mask] * overlap[mask] * pair_k)
            fy[i] += np.sum(dy[mask] / d[mask] * overlap[mask] * pair_k)
            # Subtract from others
            for j in np.where(mask)[0]:
                fx[j] -= dx[j] / d[j] * overlap[j] * pair_k
                fy[j] -= dy[j] / d[j] * overlap[j] * pair_k
    
    return fx, fy

def _force_relax(cx, cy, r, W, H, max_iter=200, dt=0.004):
    """Force-directed relaxation."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    n = len(cx)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        
        fx, fy = _vectorized_forces(cx, cy, r, W, H)
        
        # Adaptive step size
        max_f = np.max(np.abs(fx)) + np.max(np.abs(fy))
        if max_f < 1e-8:
            break
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r, np.maximum(lo, 1e-6))
    
    return cx, cy, r

def _slsqp_optimize(cx, cy, r, W, H, max_iter=150):
    """SLSQP optimization for a fixed rectangle."""
    n = N
    x0 = np.concatenate([cx, cy, r])
    
    def objective(z):
        x = z[:n]
        y = z[n:2*n]
        rad = z[2*n:3*n]
        rad = np.maximum(rad, 1e-6)
        
        obj = -np.sum(rad)
        
        # Pairwise overlap penalty
        for i in range(n):
            dx = x[i] - x
            dy = y[i] - y
            d = np.sqrt(dx*dx + dy*dy)
            d[i] = 1e12
            overlap = rad[i] + rad - d
            mask = overlap > 0
            if np.any(mask):
                obj += 200.0 * np.sum(overlap[mask] ** 2)
        
        # Wall penalty
        for i in range(n):
            if x[i] < rad[i]:
                obj += 200.0 * (rad[i] - x[i]) ** 2
            elif x[i] > W - rad[i]:
                obj += 200.0 * (x[i] - (W - rad[i])) ** 2
            if y[i] < rad[i]:
                obj += 200.0 * (rad[i] - y[i]) ** 2
            elif y[i] > H - rad[i]:
                obj += 200.0 * (y[i] - (H - rad[i])) ** 2
        
        return obj
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))  # x
        bounds.append((0.0, H))  # y
        bounds.append((1e-6, min(W, H)/2.0))  # r
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'maxls': 50})
        
        x_opt = result.x[:n]
        y_opt = result.x[n:2*n]
        r_opt = result.x[2*n:3*n]
        
        # Clamp
        x_opt = np.clip(x_opt, 0.0, W)
        y_opt = np.clip(y_opt, 0.0, H)
        r_opt = np.maximum(r_opt, 1e-6)
        lo = np.minimum(np.minimum(x_opt, y_opt), np.minimum(W - x_opt, H - y_opt))
        r_opt = np.minimum(r_opt, lo)
        
        s = _evaluate(x_opt, y_opt, r_opt, W, H)
        if s is not None:
            return x_opt, y_opt, r_opt, s
    except Exception:
        pass
    
    return None

def _radial_layout(W, H, n_rings=4):
    """Radial/hexagonal layout."""
    cx = []
    cy = []
    r_list = []
    
    # Center
    cx.append(W/2.0)
    cy.append(H/2.0)
    base_r = min(W, H) * 0.25
    r_list.append(base_r * 0.8)
    
    # Rings
    for ring in range(1, n_rings + 1):
        angle_step = 2 * np.pi / (ring * 6)
        ring_r = min(W, H) * 0.12 * ring
        for k in range(ring * 6):
            angle = k * angle_step
            x = W/2.0 + ring_r * np.cos(angle)
            y = H/2.0 + ring_r * np.sin(angle)
            if 0.1 < x < W - 0.1 and 0.1 < y < H - 0.1:
                cx.append(x)
                cy.append(y)
                r_list.append(min(x, y, W-x, H-y) * 0.7)
        
        if len(cx) >= N:
            break
    
    # Pad if needed
    while len(cx) < N:
        cx.append(random.uniform(0.1, W-0.1))
        cy.append(random.uniform(0.1, H-0.1))
        r_list.append(0.02)
    
    return np.array(cx[:N]), np.array(cy[:N]), np.array(r_list[:N])

def _grid_layout(kx, ky, W, H, shrink=0.9):
    """Grid layout."""
    cx = np.linspace(W/(2*kx), W*(2*kx-1)/(2*kx), kx)
    cy = np.linspace(H/(2*ky), H*(2*ky-1)/(2*ky), ky)
    
    pts_cx = []
    pts_cy = []
    pts_r = []
    
    for y in cy:
        for x in cx:
            pts_cx.append(x)
            pts_cy.append(y)
            pts_r.append(min(x, y, W-x, H-y) * shrink)
    
    # Pad
    while len(pts_cx) < N:
        pts_cx.append(random.uniform(0.05*W, 0.95*W))
        pts_cy.append(random.uniform(0.05*H, 0.95*H))
        pts_r.append(0.01)
    
    return np.array(pts_cx[:N]), np.array(pts_cy[:N]), np.array(pts_r[:N])

def _hex_layout(W, H):
    """Hexagonal packing layout."""
    cx = []
    cy = []
    r_list = []
    
    base_r = min(W, H) / 10.0
    row_spacing = base_r * 1.732
    col_spacing = base_r * 1.5
    
    row = 0
    while len(cx) < N:
        y = row * row_spacing
        if y > H - base_r:
            row += 1
            continue
        
        offset = col_spacing / 2 if row % 2 else 0
        col = 0
        while len(cx) < N:
            x = col * col_spacing + offset
            if x > W - base_r:
                break
            cx.append(x)
            cy.append(y)
            r_list.append(min(x, y, W-x, H-y) * 0.85)
            col += 1
        row += 1
    
    return np.array(cx[:N]), np.array(cy[:N]), np.array(r_list[:N])

def _random_layout(W, H):
    """Random feasible layout."""
    cx = np.random.uniform(0.1*W, 0.9*W, N)
    cy = np.random.uniform(0.1*H, 0.9*H, N)
    r_list = np.minimum(np.minimum(cx, cy), np.minimum(W-cx, H-cy)) * 0.4
    r_list = np.maximum(r_list, 0.005)
    return cx, cy, r_list

def construct():
    best_sum = 0.0
    best_result = None
    
    def time_ok():
        return time.time() < DEADLINE - 5
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse starting configurations
    candidates = []
    
    Ws = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for W in Ws:
        H = 2.0 - W
        if H < 0.3 or W < 0.3:
            continue
        
        # Grid layouts
        for kx, ky in [(3,4), (4,3), (3,3), (4,4), (5,3), (3,5), (4,5), (5,4)]:
            if kx * ky < N:
                continue
            cx, cy, r = _grid_layout(kx, ky, W, H)
            s = _evaluate(cx, cy, r, W, H)
            if s is not None:
                candidates.append((s, cx, cy, r, W, H))
        
        # Hex layout
        cx, cy, r = _hex_layout(W, H)
        s = _evaluate(cx, cy, r, W, H)
        if s is not None:
            candidates.append((s, cx, cy, r, W, H))
        
        # Radial layout
        cx, cy, r = _radial_layout(W, H)
        s = _evaluate(cx, cy, r, W, H)
        if s is not None:
            candidates.append((s, cx, cy, r, W, H))
        
        # Random layouts
        for _ in range(3):
            cx, cy, r = _random_layout(W, H)
            s = _evaluate(cx, cy, r, W, H)
            if s is not None:
                candidates.append((s, cx, cy, r, W, H))
    
    # Sort by initial score
    candidates.sort(key=lambda t: -t[0])
    
    # Phase 1: Force relaxation + SLSQP on top candidates
    top_k = min(12, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        
        s0, cx0, cy0, r0, W, H = candidates[idx]
        
        # Force relaxation
        cx_r, cy_r, r_r = _force_relax(cx0, cy0, r0, W, H, max_iter=150)
        
        # SLSQP optimization
        result = _slsqp_optimize(cx_r, cy_r, r_r, W, H, max_iter=100)
        if result is not None:
            cx_opt, cy_opt, r_opt, s_opt = result
            if s_opt > best_sum:
                best_sum = s_opt
                best_result = (cx_opt, cy_opt, r_opt, W, H)
    
    # Phase 2: If we have a good result, try to improve with more iterations
    if best_result is not None and time_ok():
        cx0, cy0, r0, W, H = best_result
        # More SLSQP iterations
        result = _slsqp_optimize(cx0, cy0, r0, W, H, max_iter=200)
        if result is not None:
            cx_opt, cy_opt, r_opt, s_opt = result
            if s_opt > best_sum:
                best_sum = s_opt
                best_result = (cx_opt, cy_opt, r_opt, W, H)
    
    # Phase 3: Polish with force-directed radius growth
    if best_result is not None and time_ok():
        cx0, cy0, r0, W, H = best_result
        cx_p, cy_p, r_p = cx0.copy(), cy0.copy(), r0.copy()
        
        # Try to grow radii iteratively
        for _ in range(30):
            if not time_ok():
                break
            improved = False
            for i in range(N):
                # Try growing radius
                dr = 0.0005
                new_r = r_p[i] + dr
                lo = min(cx_p[i], cy_p[i], W - cx_p[i], H - cy_p[i])
                if new_r > lo:
                    new_r = lo
                if new_r <= r_p[i]:
                    continue
                
                # Check overlaps
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(cx_p[i] - cx_p[j], cy_p[i] - cy_p[j])
                    if d < new_r + r_p[j] - 1e-8:
                        valid = False
                        break
                
                if valid:
                    r_p[i] = new_r
                    improved = True
            
            if not improved:
                break
        
        s_polished = _evaluate(cx_p, cy_p, r_p, W, H)
        if s_polished is not None and s_polished > best_sum:
            best_sum = s_polished
            best_result = (cx_p, cy_p, r_p, W, H)
    
    # Fallback
    if best_result is None:
        W, H = 1.0, 1.0
        side = 5
        cx = []
        cy = []
        r_list = []
        for i in range(N):
            row, col = divmod(i, side)
            cx.append((col + 0.5) / side)
            cy.append((row + 0.5) / side)
            r_list.append(0.5 / side - 1e-4)
        best_result = (np.array(cx), np.array(cy), np.array(r_list), W, H)
    
    cx, cy, r, W, H = best_result
    
    # Final clamping
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    out = np.column_stack([cx, cy, r])
    
    # Final validation
    s_final = _evaluate(out[:, 0], out[:, 1], out[:, 2], W, H)
    if s_final is None:
        # Emergency fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(5) for j in range(5)]
        return np.array(cells[:N])
    
    return out
