"""Circle packing in perimeter-4 rectangle: SLSQP + force relaxation + radius polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_DEADLINE = 145.0
_START = None


def _is_feasible(x, w, tol=1e-6):
    """Check if packing x (r, cx, cy) is feasible in rectangle w x (2-w)."""
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if np.any(r < -tol):
        return False
    if np.any(cx < r - tol) or np.any(cx > w - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
    # Pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(x):
    return float(np.sum(x[:N]))


def _grid_seed(w, h):
    """Generate a grid-based seed packing."""
    best_seed = None
    best_score = -1e9
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.99
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1
        if _is_feasible(x, w, tol=1e-7):
            score = np.sum(x[:N])
            if score > best_score:
                best_score = score
                best_seed = x.copy()
    if best_seed is None:
        w, h = 1.0, 1.0
        r = 1e-4
        x = np.zeros(3 * N)
        x[:N] = r
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        x[N:2 * N] = xs
        x[2 * N:3 * N] = ys
    return best_seed


def _make_constraints(w, h):
    """Build SLSQP constraints."""
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - 1e-5})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (w - x[N + i]) - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (h - x[2 * N + i]) - x[i]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[N + i] - x[N + j]) ** 2 +
                    (x[2 * N + i] - x[2 * N + j]) ** 2 -
                    (x[i] + x[j]) ** 2
                ),
            })
    return cons


def _force_relax(x, w, h, n_iters=500, dt=0.001):
    """Force-directed relaxation to resolve overlaps and push circles to walls."""
    x = x.copy()
    r = x[:N]
    cx = x[N:2 * N].copy()
    cy = x[2 * N:3 * N].copy()
    
    for it in range(n_iters):
        if time.time() - _START > _DEADLINE - 5:
            break
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall repulsion
        for i in range(N):
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 10
            if cx[i] > w - r[i]:
                fx[i] -= (cx[i] - (w - r[i])) * 10
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 10
            if cy[i] > h - r[i]:
                fy[i] -= (cy[i] - (h - r[i])) * 10
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - d
                if overlap > 0 and d > 1e-10:
                    f = overlap * 5.0 / d
                    fx[i] += f * dx
                    fy[i] += f * dy
                    fx[j] -= f * dx
                    fy[j] -= f * dy
                elif d < 1e-10:
                    # Random nudge
                    fx[i] += 0.01 * np.random.randn()
                    fy[i] += 0.01 * np.random.randn()
                    fx[j] -= 0.01 * np.random.randn()
                    fy[j] -= 0.01 * np.random.randn()
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to bounds
        for i in range(N):
            cx[i] = np.clip(cx[i], r[i], w - r[i])
            cy[i] = np.clip(cy[i], r[i], h - r[i])
        
        x[N:2 * N] = cx
        x[2 * N:3 * N] = cy
        
        # Check if converged
        max_force = np.max(np.abs(fx)) + np.max(np.abs(fy))
        if max_force < 1e-8:
            break
    
    return x


def _polish_radii(x, w, h, n_steps=100, grow_rate=0.001):
    """Gradually grow radii while maintaining feasibility."""
    x = x.copy()
    r = x[:N].copy()
    cx = x[N:2 * N].copy()
    cy = x[2 * N:3 * N].copy()
    
    for step in range(n_steps):
        if time.time() - _START > _DEADLINE - 5:
            break
        # Try to grow all radii by grow_rate
        new_r = r + grow_rate
        # Check wall constraints
        for i in range(N):
            if new_r[i] > cx[i] or new_r[i] > w - cx[i]:
                new_r[i] = min(cx[i], w - cx[i])
            if new_r[i] > cy[i] or new_r[i] > h - cy[i]:
                new_r[i] = min(cy[i], h - cy[i])
        
        # Check pairwise constraints
        feasible = True
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
                if d2 < (new_r[i] + new_r[j]) ** 2 - 1e-9:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            r = new_r
        else:
            # Binary search for max uniform growth
            lo, hi = 0.0, grow_rate
            for _ in range(10):
                mid = (lo + hi) / 2
                test_r = r + mid
                ok = True
                for i in range(N):
                    if test_r[i] > cx[i] or test_r[i] > w - cx[i]:
                        test_r[i] = min(cx[i], w - cx[i])
                    if test_r[i] > cy[i] or test_r[i] > h - cy[i]:
                        test_r[i] = min(cy[i], h - cy[i])
                for i in range(N):
                    for j in range(i + 1, N):
                        d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
                        if d2 < (test_r[i] + test_r[j]) ** 2 - 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    lo = mid
                else:
                    hi = mid
            r = r + lo
        
        x[:N] = r
    
    return x


def _slsqp_optimize(x0, w, h, maxiter=200, ftol=1e-10):
    """Run SLSQP optimization."""
    cons = _make_constraints(w, h)
    bounds = []
    for i in range(N):
        bounds.append((1e-5, min(w, h) / 2.0))
    for i in range(N):
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))
    
    def obj(x):
        return -np.sum(x[:N])
    
    try:
        res = minimize(
            obj, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': ftol},
        )
        return res.x
    except Exception:
        return x0


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    best_result = None
    best_val = -1e9
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.9, 1.4, 0.7, 1.5, 0.65]
    
    for w in aspect_ratios:
        if time.time() - _START > _DEADLINE - 15:
            break
        if w <= 0.05 or w >= 1.95:
            continue
        h = 2.0 - w
        
        # Generate seed
        seed = _grid_seed(w, h)
        
        # Force relaxation
        x = _force_relax(seed, w, h, n_iters=300)
        
        # SLSQP optimization
        x = _slsqp_optimize(x, w, h, maxiter=200, ftol=1e-10)
        
        # Polish radii
        x = _polish_radii(x, w, h, n_steps=80, grow_rate=0.002)
        
        # Second SLSQP pass with tighter tolerance
        if time.time() - _START < _DEADLINE - 10:
            x = _slsqp_optimize(x, w, h, maxiter=100, ftol=1e-12)
            x = _polish_radii(x, w, h, n_steps=50, grow_rate=0.001)
        
        # Check feasibility
        if _is_feasible(x, w, tol=1e-5):
            val = _sum_radii(x)
            if val > best_val:
                best_val = val
                best_result = (x.copy(), w)
    
    # Fallback
    if best_result is None:
        w = 1.0
        h = 1.0
        r = 1e-4
        x = np.zeros(3 * N)
        x[:N] = r
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        x[N:2 * N] = xs
        x[2 * N:3 * N] = ys
        return np.column_stack((xs, ys, r))
    
    x, w = best_result
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    
    # Final safety clamp
    r = np.maximum(r, 1e-5)
    cx = np.clip(cx, r, w - r)
    cy = np.clip(cy, r, h - r)
    
    # If clamping broke feasibility, shrink
    if not _is_feasible(np.concatenate([r, cx, cy]), w, tol=1e-5):
        scale = 0.99
        r = r * scale
        cx = np.clip(cx, r, w - r)
        cy = np.clip(cy, r, h - r)
        if not _is_feasible(np.concatenate([r, cx, cy]), w, tol=1e-5):
            # Last resort: very small circles
            r = np.full(N, 1e-4)
            xs = np.linspace(r[0], w - r[0], N)
            ys = np.full(N, h / 2.0)
            return np.column_stack((xs, ys, r))
    
    return np.column_stack((cx, cy, r))
