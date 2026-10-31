"""Circle packing in perimeter-4 rectangle: hex-grid init + vectorized penalty + SLSQP polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check all constraints are satisfied."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _hex_grid_positions(n_circles, W, H):
    """Generate hexagonal lattice positions clipped to rectangle."""
    # Hex packing: rows offset by half spacing
    # Try to fit n_circles in a hex pattern
    positions = []
    # Estimate spacing
    # For hex packing, area per circle ~ (2*pi*r^2)/(pi*r^2/sqrt(3)) ... 
    # Let's just generate a hex grid and take first n_circles
    max_rows = int(np.sqrt(n_circles * 2 / np.sqrt(3))) + 3
    # Start with a reasonable spacing
    base_spacing = min(W, H) / max(3, int(np.sqrt(n_circles)))
    
    for attempt in range(5):
        spacing = base_spacing * (1.0 + 0.3 * attempt)
        row_h = spacing * np.sqrt(3) / 2
        positions = []
        row = 0
        while len(positions) < n_circles and row < max_rows + 5:
            offset = (spacing / 2) if (row % 2 == 1) else 0.0
            col = 0
            while len(positions) < n_circles:
                x = offset + col * spacing
                y = row * row_h
                if x < W and y < H:
                    positions.append((x, y))
                else:
                    break
                col += 1
            row += 1
        if len(positions) >= n_circles:
            break
    
    if len(positions) < n_circles:
        # Fallback: grid
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
        positions = []
        for j in range(rows):
            for i in range(cols):
                if len(positions) >= n_circles:
                    break
                positions.append((W * (i + 0.5) / cols, H * (j + 0.5) / rows))
    
    positions = positions[:n_circles]
    cx = np.array([p[0] for p in positions])
    cy = np.array([p[1] for p in positions])
    return cx, cy


def _vectorized_penalty(x, W, H):
    """Vectorized penalty objective for L-BFGS-B."""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    # Wall violations (vectorized)
    pen = 0.0
    # Left wall
    viol = np.maximum(0, r - cx)
    pen += np.sum(viol ** 2)
    # Right wall
    viol = np.maximum(0, cx + r - W)
    pen += np.sum(viol ** 2)
    # Bottom wall
    viol = np.maximum(0, r - cy)
    pen += np.sum(viol ** 2)
    # Top wall
    viol = np.maximum(0, cy + r - H)
    pen += np.sum(viol ** 2)
    
    # Pairwise overlap (vectorized with broadcasting)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    min_dist = r[:, None] + r[None, :]
    # Only upper triangle
    iu = np.triu_indices(N, 1)
    overlap = np.maximum(0, min_dist[iu] - dist[iu])
    pen += np.sum(overlap ** 2)
    
    return -np.sum(r) + 50.0 * pen


def _force_relax(cx, cy, r, W, H, max_iter=300, dt=0.002):
    """Force-directed relaxation to resolve overlaps and wall violations."""
    n = len(cx)
    cx = cx.astype(float).copy()
    cy = cy.astype(float).copy()
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 20
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 20
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 20
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 20
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    
    return cx, cy


def _polish_slsqp(cx, cy, r, W, H, max_iter=200):
    """SLSQP polish with explicit constraints."""
    n = N
    
    def objective(x):
        cx_l = x[:n]
        cy_l = x[n:2*n]
        r_l = x[2*n:]
        return -np.sum(r_l)
    
    # Inequality constraints: g(x) >= 0
    constraints = []
    
    # Wall constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (W - x[i]) - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (H - x[n+i]) - x[2*n+i]})
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[n+i]-x[n+j])**2) - x[2*n+i] - x[2*n+j]
            })
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
    for i in range(n):
        bounds.append((1e-6, H - 1e-6))
    for i in range(n):
        bounds.append((1e-4, min(W, H) / 2))
    
    x0 = np.concatenate([cx, cy, r])
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-10})
        if res.success:
            cx_new = res.x[:n]
            cy_new = res.x[n:2*n]
            r_new = res.x[2*n:]
            if _is_valid(cx_new, cy_new, r_new, W, H, tol=1e-6):
                return cx_new, cy_new, r_new
    except:
        pass
    
    return None, None, None


def _try_aspect(W, H, deadline):
    """Try to optimize packing for given W, H."""
    best_val = -np.inf
    best_result = None
    
    # Hex grid initialization
    cx, cy = _hex_grid_positions(N, W, H)
    # Initial radius: based on local spacing
    # Use a conservative estimate
    r = np.full(N, min(W, H) / (2 * int(np.ceil(np.sqrt(N)))) * 0.85)
    
    # Force relax to fix initial issues
    cx, cy = _force_relax(cx, cy, r, W, H, max_iter=200)
    
    # L-BFGS-B with penalty (fast)
    x0 = np.concatenate([cx, cy, r])
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
    for i in range(N):
        bounds.append((1e-6, H - 1e-6))
    for i in range(N):
        bounds.append((1e-4, min(W, H) / 2))
    
    try:
        res = minimize(_vectorized_penalty, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-12})
        if res.fun < _vectorized_penalty(x0):
            cx_l = res.x[:N]
            cy_l = res.x[N:2*N]
            r_l = res.x[2*N:]
            
            # Check if valid or nearly valid
            if _is_valid(cx_l, cy_l, r_l, W, H, tol=1e-5):
                val = np.sum(r_l)
                if val > best_val:
                    best_val = val
                    best_result = (cx_l.copy(), cy_l.copy(), r_l.copy(), W, H)
            else:
                # Try SLSQP polish
                cx_p, cy_p, r_p = _polish_slsqp(cx_l, cy_l, r_l, W, H, max_iter=300)
                if cx_p is not None:
                    val = np.sum(r_p)
                    if val > best_val:
                        best_val = val
                        best_result = (cx_p, cy_p, r_p, W, H)
    except:
        pass
    
    # Also try with slightly different initial radii
    if time.time() < deadline:
        r2 = np.full(N, min(W, H) / (2 * int(np.ceil(np.sqrt(N)))) * 0.95)
        cx2, cy2 = _force_relax(cx.copy(), cy.copy(), r2, W, H, max_iter=200)
        x0b = np.concatenate([cx2, cy2, r2])
        try:
            res2 = minimize(_vectorized_penalty, x0b, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-12})
            if res2.fun < _vectorized_penalty(x0b):
                cx_l = res2.x[:N]
                cy_l = res2.x[N:2*N]
                r_l = res2.x[2*N:]
                if _is_valid(cx_l, cy_l, r_l, W, H, tol=1e-5):
                    val = np.sum(r_l)
                    if val > best_val:
                        best_val = val
                        best_result = (cx_l.copy(), cy_l.copy(), r_l.copy(), W, H)
                else:
                    cx_p, cy_p, r_p = _polish_slsqp(cx_l, cy_l, r_l, W, H, max_iter=200)
                    if cx_p is not None:
                        val = np.sum(r_p)
                        if val > best_val:
                            best_val = val
                            best_result = (cx_p, cy_p, r_p, W, H)
        except:
            pass
    
    return best_result, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_val = -np.inf
    best_result = None
    
    # Aspect ratios to try: W + H = 2
    # Square is often optimal for circle packing, but try variations
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.43, 0.65, 1.5, 0.6, 1.6, 0.55, 1.8, 0.5]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.01:
            continue
        result, val = _try_aspect(W, H, deadline)
        if result is not None and val > best_val:
            best_val = val
            best_result = result
    
    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for j in range(side):
            for i in range(side):
                if len(cells) >= N:
                    break
                r = 0.5 / side - 1e-3
                cells.append(((i + 0.5) / side, (j + 0.5) / side, r))
        return np.array(cells)
    
    cx, cy, r, W, H = best_result
    return np.column_stack([cx, cy, r])
