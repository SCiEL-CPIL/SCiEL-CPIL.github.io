"""
Circle packing in perimeter-4 rectangle:
Structural change: use SLSQP with explicit constraint functions for non-overlap
and boundary constraints, plus a "squeeze" phase that maximizes radii while
maintaining feasibility. Better seed generation with diverse aspect ratios.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 142.0  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
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
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _is_feasible(x, W, H, tol=1e-9):
    """Check if packing is feasible (no overlaps, within bounds)."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    for i in range(n):
        if r[i] < 1e-10:
            return False
        if cx[i] < r[i] - tol or cx[i] > W - r[i] + tol:
            return False
        if cy[i] < r[i] - tol or cy[i] > H - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _clamp_valid(arr, W, H):
    """Clamp a (N,3) array to be valid in rectangle W×H."""
    cx = np.clip(arr[:, 0], 0.0, W)
    cy = np.clip(arr[:, 1], 0.0, H)
    r = np.maximum(arr[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    return np.column_stack([cx, cy, r])


def _make_constraint_functions(W, H):
    """Create constraint functions for SLSQP."""
    n = N
    # Non-overlap constraints: d_ij >= r_i + r_j
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap(i, j):
                def constraint(x, i=i, j=j):
                    cx = x[0::3]
                    cy = x[1::3]
                    r = x[2::3]
                    d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                    return d - r[i] - r[j]
                constraint.name = f'overlap_{i}_{j}'
                return constraint
            overlap_constraints.append(make_overlap(i, j))
    
    # Boundary constraints: r_i <= min(cx_i, cy_i, W-cx_i, H-cy_i)
    boundary_constraints = []
    for i in range(n):
        def make_boundary(i):
            def constraint(x, i=i):
                cx = x[0::3]
                cy = x[1::3]
                r = x[2::3]
                # Distance to each wall
                d_left = cx[i]
                d_right = W - cx[i]
                d_bottom = cy[i]
                d_top = H - cy[i]
                return min(d_left, d_right, d_bottom, d_top) - r[i]
            constraint.name = f'boundary_{i}'
            return constraint
        boundary_constraints.append(make_boundary(i))
    
    return overlap_constraints + boundary_constraints


def _squeeze_radii(x, W, H, max_iter=200, dt=0.001):
    """Try to increase radii while maintaining feasibility."""
    x = x.copy()
    n = N
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        # Try to increase all radii slightly
        r_old = x[:, 2].copy()
        x[:, 2] = r_old * (1 + dt)
        
        # Check feasibility
        if _is_feasible(x, W, H, tol=1e-8):
            continue
        else:
            # Revert and try smaller increment
            x[:, 2] = r_old * (1 + dt * 0.5)
            if _is_feasible(x, W, H, tol=1e-8):
                dt *= 0.5
            else:
                x[:, 2] = r_old
                dt *= 0.3
                if dt < 1e-6:
                    break
    return x


def _grid_layout(kx, ky, W, H, shrink=0.999):
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


def _hexagonal_seed(W, H, n_circles=N):
    """Generate a hexagonal packing seed."""
    pts = []
    # Try different hexagonal grid spacings
    for spacing in [0.3, 0.25, 0.2, 0.15]:
        cx = 0
        cy = 0
        row = 0
        while len(pts) < n_circles and cy < H:
            for col in range(int(W / spacing) + 2):
                x = col * spacing + (row % 2) * spacing * 0.5
                y = cy + row * spacing * np.sqrt(3) / 2
                if x < W and y < H:
                    r = min(x, y, W - x, H - y) * 0.45
                    if r > 0.01:
                        pts.append([x, y, r])
            cy += spacing * np.sqrt(3) / 2
            row += 1
        if len(pts) >= n_circles:
            break
    if len(pts) < n_circles:
        # Fill remaining with small circles
        rng = random.Random(42)
        while len(pts) < n_circles:
            x = rng.uniform(0.1 * W, 0.9 * W)
            y = rng.uniform(0.1 * H, 0.9 * H)
            r = min(x, y, W - x, H - y) * 0.2
            pts.append([x, y, max(r, 0.01)])
    return np.array(pts[:n_circles])


def _random_layout(W, H, seed=None):
    rng = random.Random(seed)
    pts = []
    for _ in range(N):
        cx = rng.uniform(0.05 * W, 0.95 * W)
        cy = rng.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)


def _constrained_objective(x, W, H):
    """Objective: minimize -sum(r) (i.e., maximize sum of radii)."""
    r = x[2::3][:N]
    return -np.sum(r)


def _constrained_grad(x, W, H):
    """Gradient of objective."""
    grad = np.zeros_like(x)
    grad[2::3][:N] = -1.0
    return grad


def _constrained_optimize(x0, W, H, max_iter=300):
    """Run SLSQP with explicit constraints."""
    n = N
    bounds = [(0, W)] * n + [(0, H)] * n + [(1e-6, min(W, H) / 2)] * n
    
    # Create constraint functions
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap(i, j):
                def constraint(x, i=i, j=j):
                    cx = x[0::3]
                    cy = x[1::3]
                    r = x[2::3]
                    d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                    return d - r[i] - r[j]
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_overlap(i, j)})
    
    for i in range(n):
        def make_boundary(i):
            def constraint(x, i=i):
                cx = x[0::3]
                cy = x[1::3]
                r = x[2::3]
                d_left = cx[i]
                d_right = W - cx[i]
                d_bottom = cy[i]
                d_top = H - cy[i]
                return min(d_left, d_right, d_bottom, d_top) - r[i]
            return constraint
        constraints.append({'type': 'ineq', 'fun': make_boundary(i)})
    
    try:
        res = minimize(_constrained_objective, x0, 
                       args=(W, H),
                       jac=_constrained_grad,
                       method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': 0})
        return res.x
    except Exception:
        return x0


# ---------------------------------------------------------------------------
# Force-directed relaxation for preconditioning
# ---------------------------------------------------------------------------

def _force_relax(x, W, H, max_iter=200, dt=0.003):
    """Force-directed relaxation to fix overlaps and boundary violations."""
    x = x.copy().astype(float)
    n = N
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        cx = x[:, 0].copy()
        cy = x[:, 1].copy()
        r = x[:, 2].copy()
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Boundary forces
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 20
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 20
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                md = r[i] + r[j]
                if d < md and d > 1e-12:
                    ov = md - d
                    fx[i] += (dx / d) * ov * 10
                    fy[i] += (dy / d) * ov * 10
                    fx[j] -= (dx / d) * ov * 10
                    fy[j] -= (dy / d) * ov * 10
        
        x[:, 0] += dt * fx
        x[:, 1] += dt * fy
        
        # Clamp positions
        for i in range(n):
            x[i, 0] = np.clip(x[i, 0], r[i], W - r[i])
            x[i, 1] = np.clip(x[i, 1], r[i], H - r[i])
    
    return x


# ---------------------------------------------------------------------------
# Aspect ratio optimization
# ---------------------------------------------------------------------------

def _optimize_aspect_ratio(x0, max_iter=100):
    """Optimize aspect ratio W (with H = 2 - W) for a given packing."""
    def objective(W):
        H = 2.0 - W
        if W < 0.3 or H < 0.3:
            return 1e6
        # Check if packing fits
        cx = x0[:, 0]
        cy = x0[:, 1]
        r = x0[:, 2]
        n = len(cx)
        # Scale positions to fit in W x H
        scale_x = W / 1.0
        scale_y = H / 1.0
        # Check if circles fit
        for i in range(n):
            if cx[i] * scale_x < r[i] or cx[i] * scale_x > W - r[i]:
                return 1e6
            if cy[i] * scale_y < r[i] or cy[i] * scale_y > H - r[i]:
                return 1e6
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = (cx[i] - cx[j]) * scale_x
                dy = (cy[i] - cy[j]) * scale_y
                d = np.hypot(dx, dy)
                if d < r[i] + r[j]:
                    return 1e6
        return -np.sum(r)
    
    try:
        res = minimize(objective, 1.0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
        return res.x[0]
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Multi-start with diverse seeds
# ---------------------------------------------------------------------------

def _generate_seeds(W, H, n_seeds=10):
    """Generate diverse initial layouts."""
    seeds = []
    # Grid layouts
    for kx, ky in [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)]:
        if kx * ky >= N:
            seeds.append(_grid_layout(kx, ky, W, H, shrink=0.95))
    
    # Hexagonal
    seeds.append(_hexagonal_seed(W, H))
    
    # Random
    for i in range(3):
        seeds.append(_random_layout(W, H, seed=42 + i))
    
    return seeds[:n_seeds]


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    def time_ok():
        return time.time() < DEADLINE

    random.seed(42)
    np.random.seed(42)

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
    
    for w0 in aspect_ratios:
        if not time_ok():
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Generate seeds for this aspect ratio
        seeds = _generate_seeds(w0, h0, n_seeds=5)
        
        for seed_idx, x0 in enumerate(seeds):
            if not time_ok():
                break
            
            # Force relaxation to fix initial violations
            x_relaxed = _force_relax(x0, w0, h0, max_iter=100)
            
            # Constrained SLSQP optimization
            remaining = DEADLINE - time.time()
            maxiter = int(min(200, remaining * 20))
            if maxiter < 30:
                break
            
            x_opt = _constrained_optimize(x_relaxed, w0, h0, max_iter=maxiter)
            
            # Evaluate
            s = _evaluate(x_opt, w0, h0)
            if s is not None and s > best_sum:
                best_sum = s
                best_arr = x_opt.copy()
                best_W = w0
                best_H = h0
    
    # If we have a best solution, try to squeeze radii
    if best_arr is not None and time_ok():
        remaining = DEADLINE - time.time()
        if remaining > 10:
            # Squeeze radii
            best_arr_squeezed = _squeeze_radii(best_arr, best_W, best_H, 
                                              max_iter=100, dt=0.001)
            s_squeezed = _evaluate(best_arr_squeezed, best_W, best_H)
            if s_squeezed is not None and s_squeezed > best_sum:
                best_sum = s_squeezed
                best_arr = best_arr_squeezed
    
    # Final clamping & validation
    if best_arr is None:
        W, H = 1.0, 1.0
        best_arr = _grid_layout(5, 5, W, H, shrink=0.98)
        best_W, best_H = W, H

    best_arr = _clamp_valid(best_arr, best_W, best_H)

    s_final = _evaluate(best_arr, best_W, best_H)
    if s_final is None:
        # Fallback grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety shrink
    r = best_arr[:, 2] * (1 - 1e-7)
    best_arr = np.column_stack([best_arr[:, 0], best_arr[:, 1], r])
    best_arr = _clamp_valid(best_arr, best_W, best_H)

    return best_arr
