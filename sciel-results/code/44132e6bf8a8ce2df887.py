"""Circle packing in perimeter-4 rectangle: unified SLSQP with explicit constraints + force-directed polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # safety margin before 150s kill

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
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
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _force_relax(x, W, H, max_iter=300, dt=0.003):
    """
    Force-directed relaxation to resolve overlaps in a packing.
    x: (n, 3) array with [cx, cy, r]
    Returns relaxed x.
    """
    n = len(x)
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 20
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 20
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20
        
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
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Clamp positions
        for i in range(n):
            cx[i] = np.clip(cx[i], 0.0, W)
            cy[i] = np.clip(cy[i], 0.0, H)
            lo = min(cx[i], cy[i], W - cx[i], H - cy[i])
            r[i] = min(r[i], max(lo, 1e-6))
    
    return np.column_stack([cx, cy, r])

def _slsqp_unified(z, max_iter=300):
    """
    SLSQP optimization in unified parameter space with explicit inequality constraints.
    z: flat vector [x1, y1, r1, ..., xN, yN, rN, w]
    Returns (z_opt, value) or (None, 0.0).
    """
    def objective(z_vec):
        x = z_vec[0::3][:N]
        y = z_vec[1::3][:N]
        r = z_vec[2::3][:N]
        w = z_vec[-1]
        h = 2.0 - w
        
        if w < 0.3 or h < 0.3:
            return 1e6
        
        return -np.sum(r)
    
    def check_validity(z_vec, tol=1e-8):
        x = z_vec[0::3][:N]
        y = z_vec[1::3][:N]
        r = z_vec[2::3][:N]
        w = z_vec[-1]
        h = 2.0 - w
        
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        
        val = float(np.sum(r))
        return True, val
    
    # Build inequality constraints: g(z) >= 0
    # For SLSQP: type='ineq' means g(z) >= 0
    
    def make_constraints(z_init):
        cons = []
        
        # Wall constraints: for each circle, distance to each wall >= r_i
        # Left wall: x_i - r_i >= 0
        # Right wall: w - x_i - r_i >= 0
        # Bottom wall: y_i - r_i >= 0
        # Top wall: h - y_i - r_i >= 0
        
        for i in range(N):
            # Left
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[3*i] - z[3*i+2],
                'jac': lambda z, i=i: np.array([0.0 if k != 3*i else 1.0 for k in range(3*N+1)] + [0.0])
            })
            # Right: w - x_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[-1] - z[3*i] - z[3*i+2],
                'jac': lambda z, i=i: np.array([0.0 if k != 3*i else -1.0 for k in range(3*N+1)] + [-1.0])
            })
            # Bottom
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[3*i+1] - z[3*i+2],
                'jac': lambda z, i=i: np.array([0.0 if k != 3*i+1 else 1.0 for k in range(3*N+1)] + [0.0])
            })
            # Top: h - y_i - r_i = (2-w) - y_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i: (2.0 - z[-1]) - z[3*i+1] - z[3*i+2],
                'jac': lambda z, i=i: np.array([0.0 if k != 3*i+1 else -1.0 for k in range(3*N+1)] + [-1.0])
            })
        
        # Overlap constraints: d_ij - r_i - r_j >= 0
        # Use smooth approximation for differentiability
        for i in range(N):
            for j in range(i + 1, N):
                def overlap_fun(z, i=i, j=j):
                    xi = z[3*i]
                    yi = z[3*i+1]
                    ri = z[3*i+2]
                    xj = z[3*j]
                    yj = z[3*j+1]
                    rj = z[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    return d - ri - rj
                
                def overlap_jac(z, i=i, j=j):
                    xi = z[3*i]
                    yi = z[3*i+1]
                    ri = z[3*i+2]
                    xj = z[3*j]
                    yj = z[3*j+1]
                    rj = z[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    jac = np.zeros(3*N+1)
                    jac[3*i] = dx / d
                    jac[3*i+1] = dy / d
                    jac[3*i+2] = -1.0
                    jac[3*j] = -dx / d
                    jac[3*j+1] = -dy / d
                    jac[3*j+2] = -1.0
                    return jac
                
                cons.append({'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac})
        
        # Width constraint: 0.3 <= w <= 1.7
        cons.append({'type': 'ineq', 'fun': lambda z: z[-1] - 0.3, 'jac': lambda z: np.array([0.0]*(3*N) + [1.0])})
        cons.append({'type': 'ineq', 'fun': lambda z: 1.7 - z[-1], 'jac': lambda z: np.array([0.0]*(3*N) + [-1.0])})
        
        return cons
    
    try:
        constraints = make_constraints(z)
        
        result = minimize(
            objective, z, method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'maxls': 50}
        )
        
        z_opt = result.x
        valid, val = check_validity(z_opt)
        
        if valid and val > 0:
            return z_opt, val
        else:
            # Try repair: shrink radii slightly
            z_repair = z_opt.copy()
            r_idx = np.arange(2, len(z_repair) - 1, 3)
            z_repair[r_idx] *= 0.95
            valid, val = check_validity(z_repair)
            if valid and val > 0:
                return z_repair, val
    except Exception:
        pass
    
    return None, 0.0

def _polish_result(z, time_budget):
    """
    Polish a valid packing with force-directed radius growth.
    z: flat vector [x1, y1, r1, ..., xN, yN, rN, w]
    Returns polished z or None.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    dt = 0.001
    max_iter = 500
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10
        
        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Clamp positions
        for i in range(n):
            x[i] = np.clip(x[i], 0.0, w)
            y[i] = np.clip(y[i], 0.0, h)
            lo = min(x[i], y[i], w - x[i], h - y[i])
            r[i] = min(r[i], max(lo, 1e-6))
    
    # Try to grow radii
    for _ in range(50):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(n):
            # Try growing radius slightly
            dr = 0.001
            new_r = r[i] + dr
            lo = min(x[i], y[i], w - x[i], h - y[i])
            if new_r > lo:
                new_r = lo
            if new_r <= r[i]:
                continue
            
            # Check if this growth causes overlaps
            valid = True
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < new_r + r[j]:
                    valid = False
                    break
            
            if valid:
                r[i] = new_r
                improved = True
        
        if not improved:
            break
    
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    
    return z

def _grid_layout(kx, ky, W, H, shrink=0.95):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _random_layout(W, H):
    """Random feasible-ish layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.1 * W, 0.9 * W)
        cy = random.uniform(0.1 * H, 0.9 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.4
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def _honeycomb_layout(W, H):
    """
    Generate a hexagonal/honeycomb-like layout for better packing density.
    """
    pts = []
    base_r = min(W, H) / 8.0
    
    rows = int(H / (base_r * 1.732)) + 2
    cols = int(W / (base_r * 1.5)) + 2
    
    for row in range(rows):
        y = row * base_r * 1.732
        if y > H:
            break
        offset = (base_r * 1.5) if (row % 2 == 1) else 0
        for col in range(cols):
            x = col * base_r * 1.5 + offset
            if x > W:
                break
            r = min(x, y, W - x, H - y) * 0.9
            r = max(r, 1e-4)
            pts.append([x, y, r])
    
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def construct():
    best_sum = 0.0
    best_z = None  # unified parameter vector
    
    def time_ok():
        return time.time() < DEADLINE - 3
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse starting candidates in unified space
    candidates = []
    
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]
    
    for W in Ws:
        H = 2.0 - W
        if H < 0.3 or W < 0.3:
            continue
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.95)
            s = _evaluate(g, W, H)
            if s is not None:
                # Convert to unified space
                z = np.zeros(3 * N + 1)
                z[0::3][:N] = g[:, 0]
                z[1::3][:N] = g[:, 1]
                z[2::3][:N] = g[:, 2]
                z[-1] = W
                candidates.append((s, z))
        
        # Honeycomb layouts
        h = _honeycomb_layout(W, H)
        s_h = _evaluate(h, W, H)
        if s_h is not None:
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = h[:, 0]
            z[1::3][:N] = h[:, 1]
            z[2::3][:N] = h[:, 2]
            z[-1] = W
            candidates.append((s_h, z))
        
        # Random starts
        for _ in range(2):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                z = np.zeros(3 * N + 1)
                z[0::3][:N] = g[:, 0]
                z[1::3][:N] = g[:, 1]
                z[2::3][:N] = g[:, 2]
                z[-1] = W
                candidates.append((s, z))
    
    # Sort by initial score
    candidates.sort(key=lambda t: -t[0])
    
    # Phase 1: Force-directed relaxation + SLSQP optimization on top candidates
    top_k = min(10, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, z0 = candidates[idx]
        
        # Extract W, H for force relaxation
        W = z0[-1]
        H = 2.0 - W
        x0 = np.column_stack([z0[0::3][:N], z0[1::3][:N], z0[2::3][:N]])
        
        # Force relaxation
        x_relaxed = _force_relax(x0, W, H, max_iter=200)
        
        # Convert back to unified space
        z_relaxed = z0.copy()
        z_relaxed[0::3][:N] = x_relaxed[:, 0]
        z_relaxed[1::3][:N] = x_relaxed[:, 1]
        z_relaxed[2::3][:N] = x_relaxed[:, 2]
        
        # SLSQP optimization with explicit constraints
        z_opt, s_opt = _slsqp_unified(z_relaxed, max_iter=200)
        if z_opt is not None and s_opt > best_sum:
            best_sum = s_opt
            best_z = z_opt
    
    # Phase 2: Polish the best result
    if time_ok() and best_z is not None:
        remaining = DEADLINE - time.time()
        polished = _polish_result(best_z, time_budget=min(30, remaining - 5))
        if polished is not None:
            # Check validity
            x = polished[0::3][:N]
            y = polished[1::3][:N]
            r = polished[2::3][:N]
            w = polished[-1]
            h = 2.0 - w
            x_arr = np.column_stack([x, y, r])
            s_polished = _evaluate(x_arr, w, h)
            if s_polished is not None and s_polished > best_sum:
                best_sum = s_polished
                best_z = polished
    
    # Fallback
    if best_z is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.95)
        best_z = np.zeros(3 * N + 1)
        best_z[0::3][:N] = g[:, 0]
        best_z[1::3][:N] = g[:, 1]
        best_z[2::3][:N] = g[:, 2]
        best_z[-1] = W
    
    # Extract final result
    cx = best_z[0::3][:N]
    cy = best_z[1::3][:N]
    r = best_z[2::3][:N]
    W = best_z[-1]
    H = 2.0 - W
    
    # Final clamping
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    # Final validation
    s_final = _evaluate(out, W, H)
    if s_final is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
