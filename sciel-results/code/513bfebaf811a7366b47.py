"""Circle packing in perimeter-4 rectangle: multi-start SLSQP with free aspect ratio + force polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # safety margin before 150s kill

def _check_valid(x, y, r, w, h, tol=1e-9):
    """Check if packing is valid. Returns (valid, value)."""
    n = len(x)
    if w < 1e-6 or h < 1e-6:
        return False, 0.0
    for i in range(n):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False, 0.0
    return True, float(np.sum(r))

def _force_polish(x, y, r, w, h, max_iter=300, dt=0.002, time_budget=10.0):
    """Force-directed relaxation to resolve overlaps and push circles apart."""
    n = len(x)
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        if time.time() > DEADLINE - 2:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 50
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 50
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 50
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 50
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20
        
        # Update
        x += dt * fx
        y += dt * fy
        
        # Clamp
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    return x, y, r

def _radius_growth(x, y, r, w, h, max_iter=200, time_budget=5.0):
    """Try to grow radii while maintaining feasibility."""
    n = len(x)
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        if time.time() > DEADLINE - 2:
            break
        
        improved = False
        for i in range(n):
            # Max growth for circle i
            max_g = min(x[i] - r[i], w - x[i] - r[i],
                       y[i] - r[i], h - y[i] - r[i])
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                max_g = min(max_g, d - r[i] - r[j])
            
            if max_g > 1e-8:
                growth = min(0.002, max_g * 0.3)
                r[i] += growth
                improved = True
        
        if not improved:
            break
    
    return x, y, r

def _slsqp_optimize(x0, y0, r0, w0, time_budget=20.0):
    """SLSQP optimization on (x, y, r, w) with h = 2 - w."""
    n = N
    # Parameter vector: [x1..xn, y1..yn, r1..rn, w]
    z0 = np.concatenate([x0, y0, r0, [w0]])
    
    def objective(z):
        x = z[:n]
        y = z[n:2*n]
        r = z[2*n:3*n]
        w = z[3*n]
        h = 2.0 - w
        
        if w < 0.1 or h < 0.1:
            return 1e8
        
        val = -np.sum(r)
        
        # Overlap penalty (smooth)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                req = r[i] + r[j]
                if d < req:
                    val += 1000.0 * (req - d) ** 2
        
        # Boundary penalty
        for i in range(n):
            if x[i] < r[i]:
                val += 1000.0 * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                val += 1000.0 * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                val += 1000.0 * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                val += 1000.0 * (y[i] - (h - r[i])) ** 2
        
        # Aspect ratio penalty to keep reasonable
        if w < 0.3:
            val += 1000.0 * (0.3 - w) ** 2
        elif w > 1.7:
            val += 1000.0 * (w - 1.7) ** 2
        
        return val
    
    def check_validity(z):
        x = z[:n]
        y = z[n:2*n]
        r = z[2*n:3*n]
        w = z[3*n]
        h = 2.0 - w
        return _check_valid(x, y, r, w, h)
    
    maxiter = int(min(200, time_budget * 20))
    try:
        res = minimize(objective, z0, method='SLSQP',
                       bounds=[(0, 2)] * (3 * n) + [(0.2, 1.8)],
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 30})
        
        z_opt = res.x
        x = z_opt[:n]
        y = z_opt[n:2*n]
        r = z_opt[2*n:3*n]
        w = z_opt[3*n]
        h = 2.0 - w
        
        valid, val = check_validity(z_opt)
        if valid:
            return x, y, r, w, val
        
        # Try repair: shrink radii
        r_repair = r * 0.9
        valid, val = _check_valid(x, y, r_repair, w, h)
        if valid:
            return x, y, r_repair, w, val
        
        # Try more aggressive repair
        r_repair = r * 0.8
        valid, val = _check_valid(x, y, r_repair, w, h)
        if valid:
            return x, y, r_repair, w, val
        
        return None
    except Exception:
        return None

def construct():
    best_value = 0.0
    best_result = None
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Grid-based starts with various aspect ratios
    side = int(np.ceil(np.sqrt(N)))  # 5
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.65, 1.67, 0.6, 1.85, 0.55]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.25:
            continue
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.4
        configs.append((x, y, r, w0))
    
    # 2. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(8):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        x = rng.uniform(0.1, w0 - 0.1, N)
        y = rng.uniform(0.1, h0 - 0.1, N)
        r = rng.uniform(0.04, 0.10, N)
        configs.append((x, y, r, w0))
    
    # 3. Concentric ring starts
    for w0 in [1.0, 0.9, 1.1]:
        h0 = 2.0 - w0
        cx, cy = w0 / 2, h0 / 2
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        # Center circle
        x[0], y[0] = cx, cy
        r[0] = min(w0, h0) * 0.15
        # Ring of circles
        ring_count = N - 1
        ring_r = min(w0, h0) * 0.12
        for i in range(1, N):
            angle = 2 * np.pi * (i - 1) / ring_count
            dist = min(w0, h0) * 0.35
            x[i] = cx + dist * np.cos(angle)
            y[i] = cy + dist * np.sin(angle)
            r[i] = ring_r
        configs.append((x, y, r, w0))
    
    # Sort configs by initial potential (sum of radii) - try best first
    # But also keep diversity, so just process in order
    
    for idx, (x0, y0, r0, w0) in enumerate(configs):
        if time.time() > DEADLINE - 10:
            break
        
        elapsed = time.time() - START
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break
        
        # Force polish to resolve initial overlaps
        x_pol, y_pol, r_pol = _force_polish(x0, y0, r0, w0, 
                                              max_iter=100, dt=0.003,
                                              time_budget=min(3, remaining * 0.15))
        
        # Check if valid after polish
        h0 = 2.0 - w0
        valid, val = _check_valid(x_pol, y_pol, r_pol, w0, h0)
        if valid and val > best_value:
            best_value = val
            best_result = (x_pol.copy(), y_pol.copy(), r_pol.copy(), w0)
        
        # SLSQP optimization
        time_for_slsqp = min(15.0, remaining * 0.5)
        result = _slsqp_optimize(x_pol, y_pol, r_pol, w0, time_budget=time_for_slsqp)
        
        if result is not None:
            x_opt, y_opt, r_opt, w_opt, val_opt = result
            if val_opt > best_value:
                best_value = val_opt
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)
        
        # Try radius growth on best result
        if best_result is not None and remaining > 5:
            xb, yb, rb, wb = best_result
            hb = 2.0 - wb
            x_g, y_g, r_g = _radius_growth(xb, yb, rb, wb, hb,
                                            max_iter=100, time_budget=min(3, remaining * 0.15))
            valid_g, val_g = _check_valid(x_g, y_g, r_g, wb, hb)
            if valid_g and val_g > best_value:
                best_value = val_g
                best_result = (x_g.copy(), y_g.copy(), r_g.copy(), wb)
    
    # Final polish on best result
    if best_result is not None and time.time() < DEADLINE - 5:
        xb, yb, rb, wb = best_result
        hb = 2.0 - wb
        x_f, y_f, r_f = _force_polish(xb, yb, rb, wb, hb,
                                        max_iter=200, dt=0.001,
                                        time_budget=min(5, DEADLINE - time.time() - 3))
        valid_f, val_f = _check_valid(x_f, y_f, r_f, wb, hb)
        if valid_f and val_f > best_value:
            best_value = val_f
            best_result = (x_f.copy(), y_f.copy(), r_f.copy(), wb)
        
        # Final radius growth
        x_g, y_g, r_g = _radius_growth(x_f, y_f, r_f, wb, hb,
                                        max_iter=150, time_budget=min(3, DEADLINE - time.time() - 2))
        valid_g, val_g = _check_valid(x_g, y_g, r_g, wb, hb)
        if valid_g and val_g > best_value:
            best_value = val_g
            best_result = (x_g.copy(), y_g.copy(), r_g.copy(), wb)
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.01
            cells.append([cx, cy, r0])
        return np.array(cells)
    
    x, y, r, w = best_result
    # Safety shrink
    r = r * (1 - 1e-7)
    
    return np.column_stack((x, y, r))
