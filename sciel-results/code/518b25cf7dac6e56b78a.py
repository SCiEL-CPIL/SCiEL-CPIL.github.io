"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-9):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary check
    if np.any(cx - r < -tol) or np.any(cx + r > w + tol) or np.any(cy - r < -tol) or np.any(cy + r > h + tol):
        return False
    
    # Overlap check (vectorized)
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[i] + r[i+1:]
        if np.any(dist < min_dist - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _make_initial_grid(n, w, h):
    """Create a valid initial grid of circles."""
    # Use a 5x5 grid, taking the first 21
    # This ensures no overlaps initially if radii are small enough
    cols = 5
    rows = 5
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) * w / cols
            cy = (i + 0.5) * h / rows
            # Radius slightly less than half the cell size to ensure gap
            r = 0.45 * min(w / cols, h / rows)
            circles.append([cx, cy, r])
    return np.array(circles[:n])

def _optimize_single_aspect(w, h, deadline):
    """Optimize circle packing for a specific rectangle aspect ratio."""
    if time.time() > deadline:
        return None

    # Initialize
    circles = _make_initial_grid(N, w, h)
    
    # Ensure initial validity
    if not _validate(circles, w, h):
        # Shrink radii slightly if invalid
        circles[:, 2] *= 0.9
        if not _validate(circles, w, h):
            return None

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w - 1e-6)) # x
        bounds.append((1e-6, h - 1e-6)) # y
    for i in range(n):
        bounds.append((1e-6, min(w, h) / 2)) # r

    def objective_and_grad(x, beta):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty (quadratic)
        # Left: cx - r < 0 -> violation = r - cx
        viol_left = np.maximum(0, r - cx)
        obj += beta * np.sum(viol_left**2)
        grad[:n] -= 2 * beta * viol_left
        grad[2*n:] += 2 * beta * viol_left

        # Right: cx + r > w -> violation = cx + r - w
        viol_right = np.maximum(0, cx + r - w)
        obj += beta * np.sum(viol_right**2)
        grad[:n] += 2 * beta * viol_right
        grad[2*n:] += 2 * beta * viol_right

        # Bottom: cy - r < 0 -> violation = r - cy
        viol_bot = np.maximum(0, r - cy)
        obj += beta * np.sum(viol_bot**2)
        grad[n:2*n] -= 2 * beta * viol_bot
        grad[2*n:] += 2 * beta * viol_bot

        # Top: cy + r > h -> violation = cy + r - h
        viol_top = np.maximum(0, cy + r - h)
        obj += beta * np.sum(viol_top**2)
        grad[n:2*n] += 2 * beta * viol_top
        grad[2*n:] += 2 * beta * viol_top

        # Overlap penalty
        # Vectorized calculation for all pairs is O(N^2), which is fine for N=21
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[i+1:]
            
            # Only care if dist_sq < min_dist^2
            mask = dist_sq < min_dist**2
            if np.any(mask):
                dist = np.sqrt(dist_sq[mask] + 1e-12)
                overlap = min_dist[mask] - dist
                overlap = np.maximum(0, overlap)
                
                if np.any(overlap > 0):
                    pos = np.where(mask)[0]
                    for k in range(len(pos)):
                        idx = pos[k]
                        j = i + 1 + idx
                        if overlap[k] > 0:
                            pen = beta * overlap[k]**2
                            obj += pen
                            
                            ux = dx[idx] / dist[k]
                            uy = dy[idx] / dist[k]
                            
                            # Gradients
                            grad[i] -= 2 * beta * overlap[k] * ux
                            grad[j] += 2 * beta * overlap[k] * ux
                            grad[n+i] -= 2 * beta * overlap[k] * uy
                            grad[n+j] += 2 * beta * overlap[k] * uy
                            
                            # Radius gradients
                            grad[2*n+i] += 2 * beta * overlap[k]
                            grad[2*n+j] += 2 * beta * overlap[k]
        
        return obj, grad

    # Optimization loop with increasing beta
    x = x0.copy()
    betas = np.geomspace(10.0, 1000.0, 15)
    
    for beta in betas:
        if time.time() > deadline - 2.0:
            break
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < objective_and_grad(x, beta)[0]:
                x = res.x
        except Exception:
            pass

    new_circles = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    
    # Final repair step: simple relaxation to ensure validity
    # We will do a few iterations of moving circles apart if they overlap
    for _ in range(50):
        if time.time() > deadline - 1.0:
            break
        moved = False
        # Boundary clamp
        new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], w - new_circles[:, 2])
        new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], h - new_circles[:, 2])
        
        # Overlap resolution
        for i in range(n):
            for j in range(i+1, n):
                dx = new_circles[i, 0] - new_circles[j, 0]
                dy = new_circles[i, 1] - new_circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = new_circles[i, 2] + new_circles[j, 2]
                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) / 2.0
                        new_circles[i, 0] -= np.cos(angle) * push
                        new_circles[i, 1] -= np.sin(angle) * push
                        new_circles[j, 0] += np.cos(angle) * push
                        new_circles[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        new_circles[i, 0] -= ux * push
                        new_circles[i, 1] -= uy * push
                        new_circles[j, 0] += ux * push
                        new_circles[j, 1] += uy * push
                    moved = True
        
        if not moved:
            break

    # Final validation
    if _validate(new_circles, w, h, tol=1e-7):
        return new_circles
    else:
        # If still invalid, return the initial valid grid (shrunk if necessary)
        init = _make_initial_grid(N, w, h)
        if _validate(init, w, h):
            return init
        else:
            return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios
    # Perimeter 4 => w + h = 2
    # Aspect ratio r = w/h => w = r*h => h(r+1) = 2 => h = 2/(r+1), w = 2r/(r+1)
    ratios = [1.0, 1.2, 0.83, 1.5, 0.66, 2.0, 0.5, 0.7, 1.4, 1.1, 0.9]

    for ar in ratios:
        if time.time() > deadline - 10.0:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Try both orientations (w,h) and (h,w)
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                result = _optimize_single_aspect(ww, hh, deadline)
                if result is not None:
                    val = _total_radius(result)
                    if val > best_value:
                        best_value = val
                        best_result = result.copy()
            except Exception:
                continue

    # Fallback: Return a valid grid if optimization failed
    if best_result is None:
        w, h = 1.0, 1.0
        init = _make_initial_grid(N, w, h)
        if _validate(init, w, h):
            return init
        else:
            # Last resort: tiny circles
            circles = []
            for i in range(N):
                x = (i % 5 + 0.5) * 0.1
                y = (i // 5 + 0.5) * 0.1
                r = 0.04
                circles.append([x, y, r])
            return np.array(circles)
    
    return best_result
