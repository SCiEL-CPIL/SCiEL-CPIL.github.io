"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # safety margin under 150s wall-clock


def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
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


def _relax_circles(circles, w, h, iterations=500, tol=1e-8):
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

    # If relaxation failed, return original
    return circles


def _greedy_place(w, h, num_circles, grid_res=60):
    """
    Place circles sequentially to maximize the minimum radius at each step.
    Returns pos (N, 2) and radii (N,)
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
    # Grid of potential centers
    # We use a slightly inset grid to avoid boundary issues in initial placement
    margin = 0.01
    xs = np.linspace(margin*w, (1-margin)*w, grid_res)
    ys = np.linspace(margin*h, (1-margin)*h, grid_res)
    
    for k in range(num_circles):
        best_r = 0
        best_x = w / 2.0
        best_y = h / 2.0
        
        # Create meshgrid
        XX, YY = np.meshgrid(xs, ys)
        
        # Wall radius
        r_wall = np.minimum(np.minimum(XX, w - XX), np.minimum(YY, h - YY))
        
        # Circle radius constraint
        r_circle = np.full_like(r_wall, np.inf)
        for i in range(k):
            dx = XX - pos[i, 0]
            dy = YY - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_max_i = dist - radii[i]
            r_circle = np.minimum(r_circle, r_max_i)
            
        # Max allowed radius at each grid point
        r_allowed = np.minimum(r_wall, r_circle)
        
        # Find max
        max_idx = np.unravel_index(np.argmax(r_allowed), r_allowed.shape)
        best_r = r_allowed[max_idx]
        best_y = ys[max_idx[0]]
        best_x = xs[max_idx[1]]
        
        pos[k, 0] = best_x
        pos[k, 1] = best_y
        radii[k] = best_r
        
    return pos, radii


def _penalty_objective(x, w, h, penalty_weight=1000.0):
    """
    Objective function for optimization.
    x = [x1, y1, r1, x2, y2, r2, ...]
    We minimize: penalty_weight * (Sum of Squared Violations) - (Sum of Radii)
    """
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    # Compute violations
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    overlap_sq = np.zeros_like(d_sq_upper)
    overlap_mask = d_sq_upper < r_sum_upper**2
    if np.any(overlap_mask):
        d_overlap = np.sqrt(d_sq_upper[overlap_mask])
        overlap_val = r_sum_upper[overlap_mask] - d_overlap
        overlap_sq[overlap_mask] = overlap_val**2
        
    circle_viol = np.sum(overlap_sq)
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    return penalty_weight * total_viol - sum_rad


def _slsqp_optimize(circles, w, h, deadline):
    """
    Optimize radii and positions using SLSQP with hard constraints.
    Used for final polish.
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
        # x >= r  => x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w => w - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r  => y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h => h - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0 => r >= 1e-6
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
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-9}
        )

        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return None


def _solve_aspect(w, h, t0):
    # 1. Greedy Initialization
    pos, rad = _greedy_place(w, h, N, grid_res=50)
    x0 = np.column_stack((pos, rad))
    
    # Relax to ensure strict validity before optimization
    x0 = _relax_circles(x0, w, h, iterations=200)
    
    if not _validate(x0, w, h):
        return x0

    best_x = x0
    best_score = _total_radius(x0)
    
    # 2. L-BFGS-B Optimization with Penalty
    # This is fast and good for moving towards dense packing
    if time.time() < t0 + DEADLINE - 20.0:
        n = N
        bounds = []
        for i in range(n):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, min(w, h)/2))
            
        x_flat = x0.flatten()
        
        try:
            res = minimize(
                _penalty_objective,
                x_flat,
                args=(w, h, 1000.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 50}
            )
            
            x_opt = res.x.reshape(N, 3)
            # Relax to clean up any minor violations from L-BFGS-B
            x_opt = _relax_circles(x_opt, w, h, iterations=500)
            
            if _validate(x_opt, w, h):
                score = _total_radius(x_opt)
                if score > best_score:
                    best_score = score
                    best_x = x_opt
        except Exception:
            pass

    # 3. SLSQP Polish
    # Use SLSQP for rigorous constraint handling on the result from L-BFGS-B
    if time.time() < t0 + DEADLINE - 10.0:
        optimized = _slsqp_optimize(best_x, w, h, t0 + DEADLINE)
        if optimized is not None:
            score = _total_radius(optimized)
            if score > best_score:
                best_score = score
                best_x = optimized

    return best_x


def construct():
    t0 = time.time()
    best = None
    best_score = -np.inf

    # Try a range of aspect ratios
    # Perimeter 4 => w + h = 2.
    # ar = w/h. w = ar * h. h(1+ar) = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.5]
    
    # Sort by likely quality? Square is usually good, but elongated can pack better for some N.
    # We'll just iterate.
    
    for ar in ratios:
        if time.time() - t0 > DEADLINE - 5.0:
            break
            
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Try both orientations (w,h) and (h,w)
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() - t0 > DEADLINE - 5.0:
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
