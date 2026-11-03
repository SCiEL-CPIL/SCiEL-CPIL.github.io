"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

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

def _make_grid(n, w, h):
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _trust_constr_optimize(circles, w, h, deadline):
    """
    Optimize using trust-constr with smooth quadratic inequality constraints.
    Variables: x = [x1..xN, y1..yN, r1..rN]
    Objective: minimize -sum(r)
    Constraints:
      1. Boundary: x_i >= r_i, x_i + r_i <= w, y_i >= r_i, y_i + r_i <= h
      2. Overlap: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2)) # r

    # Objective and gradient
    def objective(x):
        return -np.sum(x[2*n:])
    
    def jac(x):
        g = np.zeros_like(x)
        g[2*n:] = -1.0
        return g

    # Constraints
    constraints = []
    
    # Boundary constraints (linear)
    # x_i - r_i >= 0
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i] - x[2*n+i],
            'jac': lambda x, i=i: np.concatenate([np.eye(1)[0], np.zeros(2*n-1), -np.eye(1)[0]])
        })
        # w - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: w - x[i] - x[2*n+i],
            'jac': lambda x, i=i: np.concatenate([-np.eye(1)[0], np.zeros(2*n-1), -np.eye(1)[0]])
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[n+i] - x[2*n+i],
            'jac': lambda x, i=i: np.concatenate([np.zeros(n), np.eye(1)[0], np.zeros(n-1), -np.eye(1)[0]])
        })
        # h - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: h - x[n+i] - x[2*n+i],
            'jac': lambda x, i=i: np.concatenate([np.zeros(n), -np.eye(1)[0], np.zeros(n-1), -np.eye(1)[0]])
        })

    # Overlap constraints (quadratic)
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = i, n+i, 2*n+i
            xj, yj, rj = j, n+j, 2*n+j
            
            def fun(x, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj):
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                dr = x[ri] + x[rj]
                return dx*dx + dy*dy - dr*dr
            
            def jac(x, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj):
                g = np.zeros_like(x)
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                dr = x[ri] + x[rj]
                
                g[xi] = 2 * dx
                g[xj] = -2 * dx
                g[yi] = 2 * dy
                g[yj] = -2 * dy
                g[ri] = -2 * dr
                g[rj] = -2 * dr
                return g
            
            constraints.append({'type': 'ineq', 'fun': fun, 'jac': jac})

    try:
        res = minimize(
            objective,
            x0,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            jac=jac,
            options={'maxiter': 100, 'gtol': 1e-8, 'xtol': 1e-8}
        )
        
        if res.success or res.fun < 0:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return None

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
            options={'maxiter': 200, 'ftol': 1e-9}
        )

        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
    
    for ar in ratios:
        if time.time() > deadline - 10.0:
            break
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Also try swapped orientation
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                # Initialize with grid
                circles = _make_grid(N, ww, hh)
                
                # Relax to ensure validity
                circles = _relax_circles(circles, ww, hh, iterations=500)
                
                if not _validate(circles, ww, hh):
                    continue
                
                current_value = _total_radius(circles)
                
                # Try trust-constr optimization first (new strategy)
                if time.time() < deadline - 15.0:
                    optimized = _trust_constr_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)
                
                # If trust-constr failed or time is running out, try SLSQP
                if time.time() < deadline - 10.0:
                    optimized_slsqp = _slsqp_optimize(circles, ww, hh, deadline)
                    if optimized_slsqp is not None:
                        circles = optimized_slsqp
                        current_value = _total_radius(circles)
                
                if current_value > best_value:
                    best_value = current_value
                    best_result = circles.copy()
                    
            except Exception:
                continue

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
