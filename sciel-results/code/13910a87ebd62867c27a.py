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
    # Vectorized overlap check
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=2000, tol=1e-10):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Vectorized for speed.
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

        # 2. Resolve overlaps (vectorized per circle)
        for i in range(n):
            dx = pos[i+1:, 0] - pos[i, 0]
            dy = pos[i+1:, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rad[i] + rad[i+1:]
            overlap = min_dist - dist
            
            for k in range(len(overlap)):
                if overlap[k] > 0:
                    if dist[k] < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap[k] / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[i+1+k, 0] += np.cos(angle) * push
                        pos[i+1+k, 1] += np.sin(angle) * push
                    else:
                        push = overlap[k] / 2.0
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[i+1+k, 0] += ux * push
                        pos[i+1+k, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    return circles

def _make_hexagonal_grid(n, w, h):
    """Create initial hexagonal-like arrangement for 21 circles."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows for hexagonal packing
            offset = 0.5 * (w / 5) if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5 + offset
            r = 0.35 * min(w/5, h/total_rows)
            circles.append([x, y, r])
    
    return np.array(circles[:n])

def _make_grid(n, w, h):
    """Fallback grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=1000.0, n_beta_steps=10):
    """
    Penalty-based optimization using L-BFGS-B.
    Faster than constraint-based methods due to vectorized penalty evaluation.
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

    def objective_and_grad(x, beta):
        """Compute objective and gradient with penalty."""
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        # Objective: -sum(r) + penalty
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        for i in range(n):
            if cx[i] - r[i] < 0:
                pen = beta * (cx[i] - r[i])**2
                obj += pen
                grad[i] += 2 * beta * (cx[i] - r[i])
                grad[2*n+i] -= 2 * beta * (cx[i] - r[i])
            if cx[i] + r[i] > w:
                pen = beta * (cx[i] + r[i] - w)**2
                obj += pen
                grad[i] += 2 * beta * (cx[i] + r[i] - w)
                grad[2*n+i] += 2 * beta * (cx[i] + r[i] - w)
            if cy[i] - r[i] < 0:
                pen = beta * (cy[i] - r[i])**2
                obj += pen
                grad[n+i] += 2 * beta * (cy[i] - r[i])
                grad[2*n+i] -= 2 * beta * (cy[i] - r[i])
            if cy[i] + r[i] > h:
                pen = beta * (cy[i] + r[i] - h)**2
                obj += pen
                grad[n+i] += 2 * beta * (cy[i] + r[i] - h)
                grad[2*n+i] += 2 * beta * (cy[i] + r[i] - h)
        
        # Overlap penalty (vectorized)
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[i+1:]
            overlap = np.maximum(0, min_dist - dist)
            
            # Only add penalty for actual overlaps
            mask = overlap > 0
            if np.any(mask):
                pen = beta * np.sum(overlap**2)
                obj += pen
                
                # Gradient with respect to positions
                for k in np.where(mask)[0]:
                    j = i + 1 + k
                    if dist[k] > 1e-9:
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        # d/dx_i of (min_dist - dist)^2 = -2*overlap * d(dist)/dx_i
                        # d(dist)/dx_i = -ux (since dist = |x_i - x_j|)
                        # So grad_x_i += 2 * beta * overlap * ux
                        grad[i] += 2 * beta * overlap[k] * ux
                        grad[n+i] += 2 * beta * overlap[k] * uy
                        # grad_x_j += -2 * beta * overlap * ux
                        grad[j] -= 2 * beta * overlap[k] * ux
                        grad[n+j] -= 2 * beta * overlap[k] * uy
                        
                    # Gradient with respect to radii
                    # d/dr_i of (r_i + r_j - dist)^2 = 2 * overlap
                    grad[2*n+i] += 2 * beta * overlap[k]
                    grad[2*n+j] += 2 * beta * overlap[k]
        
        return obj, grad

    # Annealing beta
    best_x = x0.copy()
    best_obj = None
    
    for step in range(n_beta_steps):
        if time.time() > deadline - 5.0:
            break
            
        t = step / max(1, n_beta_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                best_x,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Validate result
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            if _validate(new_circles, w, h, tol=1e-6):
                score = _total_radius(new_circles)
                if best_obj is None or score > best_obj:
                    best_obj = score
                    best_x = res.x.copy()
                    best_circles = new_circles
            else:
                # Relax and try again
                relaxed = _relax_circles(new_circles, w, h, iterations=500)
                if _validate(relaxed, w, h, tol=1e-6):
                    score = _total_radius(relaxed)
                    if best_obj is None or score > best_obj:
                        best_obj = score
                        best_x = np.concatenate([relaxed[:, 0], relaxed[:, 1], relaxed[:, 2]])
                        best_circles = relaxed
        except Exception:
            pass
    
    if best_obj is not None:
        return best_circles
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try various aspect ratios
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = 1.0
        h = 1.0 / aspect
        # Normalize to perimeter 4: 2(w+h) = 4 => w+h = 2
        w = 1.0
        h = 1.0
        # Actually, let's parameterize by aspect ratio
        # w + h = 2
        # h = w / aspect
        # w + w/aspect = 2 => w(1 + 1/aspect) = 2 => w = 2 / (1 + 1/aspect)
        w = 2.0 / (1.0 + 1.0/aspect)
        h = 2.0 - w
        
        # Create initial hexagonal grid
        circles = _make_hexagonal_grid(N, w, h)
        
        # Relax to resolve initial overlaps
        circles = _relax_circles(circles, w, h, iterations=1000)
        
        if _validate(circles, w, h, tol=1e-6):
            score = _total_radius(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
        
        # Penalty optimization
        if time.time() < deadline - 5.0:
            opt_circles = _penalty_optimize(circles, w, h, deadline)
            if opt_circles is not None:
                score = _total_radius(opt_circles)
                if score > best_score:
                    best_score = score
                    best_circles = opt_circles.copy()
        
        # Try grid initialization too
        if time.time() < deadline - 10.0:
            grid_circles = _make_grid(N, w, h)
            grid_circles = _relax_circles(grid_circles, w, h, iterations=1000)
            if _validate(grid_circles, w, h, tol=1e-6):
                score = _total_radius(grid_circles)
                if score > best_score:
                    best_score = score
                    best_circles = grid_circles.copy()
                
                if time.time() < deadline - 5.0:
                    opt_circles = _penalty_optimize(grid_circles, w, h, deadline)
                    if opt_circles is not None:
                        score = _total_radius(opt_circles)
                        if score > best_score:
                            best_score = score
                            best_circles = opt_circles.copy()
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: simple grid in 1x1 square
    w0, h0 = 1.0, 1.0
    side = 5
    initial_circles = []
    for i in range(side):
        for j in range(side):
            if len(initial_circles) >= N:
                break
            cx = (i + 0.5) / side * w0
            cy = (j + 0.5) / side * h0
            r = 0.5 / side * min(w0, h0) - 1e-3
            initial_circles.append([cx, cy, r])
    
    return np.array(initial_circles[:N])
