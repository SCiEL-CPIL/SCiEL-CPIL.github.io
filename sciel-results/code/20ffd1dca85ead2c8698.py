"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Combines SLSQP optimization with hexagonal lattice initialization and
affine deformation for better local optima.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 21
DEADLINE = 140.0  # safety margin under 150s wall-clock


def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _total_radius(circles):
    return float(np.sum(circles[:, 2]))


def _relax_circles(circles, w, h, iterations=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    This ensures the output is strictly valid.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]

                if dist < min_dist:
                    if dist < 1e-9:
                        # Random push if coincident
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp to ensure validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    # Final validation check
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    # If relaxation failed (shouldn't happen with small radii), return original
    return circles


def _make_hex_grid(n, w, h):
    """
    Create a hexagonal lattice packing of n circles in rectangle w x h.
    Hexagonal packing is denser than square grid, providing better initial positions.
    """
    # For hexagonal packing, rows alternate offset by half a diameter
    # Vertical spacing = r * sqrt(3), horizontal spacing = 2r
    # We want to fit n circles. Try different row/column configurations.
    
    best_cells = None
    best_score = -np.inf
    
    # Try different numbers of rows
    for n_rows in range(3, 8):
        if n_rows > n:
            continue
        n_cols = int(np.ceil(n / n_rows))
        
        # Calculate radius based on hexagonal packing
        # Horizontal: n_cols * 2r <= w => r <= w / (2 * n_cols)
        # Vertical: (n_rows - 1) * r * sqrt(3) + 2r <= h => r <= h / (2 + (n_rows-1)*sqrt(3))
        r_h = w / (2 * n_cols)
        r_v = h / (2 + (n_rows - 1) * np.sqrt(3))
        r = min(r_h, r_v) * 0.95  # 5% margin
        
        if r <= 0:
            continue
        
        cells = []
        count = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if count >= n:
                    break
                # Hexagonal offset: odd rows shifted by r
                offset = r if row % 2 == 1 else 0
                cx = r + offset + col * 2 * r
                cy = r + row * r * np.sqrt(3)
                
                # Check if within bounds
                if cx + r <= w + 1e-9 and cy + r <= h + 1e-9:
                    cells.append([cx, cy, r])
                    count += 1
        
        if len(cells) == n:
            cells = np.array(cells)
            score = _total_radius(cells)
            if score > best_score:
                best_score = score
                best_cells = cells
    
    if best_cells is not None:
        return best_cells
    
    # Fallback to square grid
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _make_grid(n, w, h):
    """Create a square grid packing of n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _slsqp_optimize(circles, w, h, deadline):
    """
    Optimize radii and positions using SLSQP with hard constraints.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.array([v for c in circles for v in c])

    # Objective: maximize sum of radii => minimize negative sum
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)

    # Constraints
    constraints = []

    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )

        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return None


def _apply_affine_deformation(circles, w, h, shear, scale_x, scale_y):
    """
    Apply an affine transformation to circle centers and adjust radii.
    This exploits the fact that hexagonal packings can be deformed
    to better fit rectangular boundaries.
    """
    n = len(circles)
    new_circles = circles.copy()
    
    # Center of the rectangle
    cx0, cy0 = w / 2, h / 2
    
    # Apply affine transformation to centers
    # x' = scale_x * (x - cx0) + shear * (y - cy0) + cx0
    # y' = scale_y * (y - cy0) + cy0
    for i in range(n):
        dx = circles[i, 0] - cx0
        dy = circles[i, 1] - cy0
        new_x = scale_x * dx + shear * dy + cx0
        new_y = scale_y * dy + cy0
        new_circles[i, 0] = new_x
        new_circles[i, 1] = new_y
    
    # Adjust radii to maintain non-overlap under deformation
    # The deformation stretches distances, so we can potentially increase radii
    # For simplicity, scale radii by the geometric mean of scale factors
    avg_scale = np.sqrt(scale_x * scale_y)
    new_circles[:, 2] = circles[:, 2] * avg_scale
    
    # Clamp to bounds
    for i in range(n):
        r = new_circles[i, 2]
        new_circles[i, 0] = np.clip(new_circles[i, 0], r, w - r)
        new_circles[i, 1] = np.clip(new_circles[i, 1], r, h - r)
    
    return new_circles


def _optimize_aspect_ratio(circles, w, h, deadline):
    """
    Optimize the aspect ratio by trying small perturbations.
    """
    if time.time() > deadline - 10.0:
        return None
    
    best = circles.copy()
    best_score = _total_radius(circles)
    
    # Try small perturbations of aspect ratio
    for delta_ar in [-0.1, -0.05, 0.05, 0.1]:
        if time.time() > deadline - 5.0:
            break
        
        new_ar = (w / h) * (1 + delta_ar)
        new_w = new_ar / (1 + new_ar) * 2
        new_h = 1 / (1 + new_ar) * 2
        
        # Scale circles to fit new rectangle
        scale_x = new_w / w
        scale_y = new_h / h
        scaled = circles.copy()
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        scaled[:, 2] *= min(scale_x, scale_y)
        
        # Relax
        scaled = _relax_circles(scaled, new_w, new_h, iterations=500)
        
        if _validate(scaled, new_w, new_h):
            score = _total_radius(scaled)
            if score > best_score:
                best_score = score
                best = scaled
    
    return best if best_score > _total_radius(circles) else None


def _solve_aspect(w, h, t0):
    """
    Solve the packing problem for a given aspect ratio.
    Uses multiple initialization strategies and SLSQP optimization.
    """
    # Try multiple initialization strategies
    inits = [
        _make_hex_grid(N, w, h),
        _make_grid(N, w, h),
    ]
    
    # Add a randomly perturbed version of the hex grid
    hex_init = inits[0].copy()
    np.random.seed(42)
    hex_init[:, :2] += np.random.uniform(-0.05, 0.05, size=(N, 2))
    inits.append(hex_init)
    
    best = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() > t0 + DEADLINE - 15.0:
            break
        
        # Relax to resolve any initial violations
        x = _relax_circles(init, w, h, iterations=500)
        
        if not _validate(x, w, h):
            continue
        
        score = _total_radius(x)
        
        # Try affine deformation
        if time.time() < t0 + DEADLINE - 20.0:
            for shear in [-0.1, 0.0, 0.1]:
                for scale_x in [0.95, 1.0, 1.05]:
                    for scale_y in [0.95, 1.0, 1.05]:
                        if time.time() > t0 + DEADLINE - 15.0:
                            break
                        
                        deformed = _apply_affine_deformation(x, w, h, shear, scale_x, scale_y)
                        deformed = _relax_circles(deformed, w, h, iterations=300)
                        
                        if _validate(deformed, w, h):
                            d_score = _total_radius(deformed)
                            if d_score > score:
                                score = d_score
                                x = deformed
        
        # Optimize with SLSQP
        if time.time() < t0 + DEADLINE - 15.0:
            optimized = _slsqp_optimize(x, w, h, t0 + DEADLINE)
            if optimized is not None:
                o_score = _total_radius(optimized)
                if o_score > score:
                    score = o_score
                    x = optimized
        
        if score > best_score:
            best_score = score
            best = x.copy()
    
    if best is not None:
        return best
    
    # Fallback: return the relaxed hex grid
    return inits[0]


def construct():
    t0 = time.time()
    best = None
    best_score = -np.inf
    
    # Try a range of aspect ratios, with more emphasis on near-square
    ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.4]
    
    for ar in ratios:
        if time.time() - t0 > DEADLINE:
            break
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Also try swapped orientation
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() - t0 > DEADLINE:
                break
            try:
                x = _solve_aspect(ww, hh, t0)
            except Exception:
                continue
            s = _total_radius(x)
            if np.isfinite(s) and s > best_score:
                best_score = s
                best = (x.copy(), ww, hh)
    
    if best is None:
        # fallback to seed-like grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, w, h = best
    
    # Final safety: if somehow invalid, shrink radii slightly
    if not _validate(x, w, h):
        for i in range(N):
            x[i, 2] *= 0.999
        if not _validate(x, w, h):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    
    return x
