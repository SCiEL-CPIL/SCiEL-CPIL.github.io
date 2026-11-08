"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # seconds, safety margin before 150s limit

# Precompute pairs for O(n^2) overlap checks
_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def _bounding_rect(circles):
    """Return (w, h) of bounding rectangle of circles."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    return x_max - x_min, y_max - y_min


def _perimeter(circles):
    w, h = _bounding_rect(circles)
    return 2.0 * (w + h)


def _sum_radii(circles):
    return np.sum(circles[:, 2])


def _is_valid(circles, tol=1e-6):
    """Check that circles are disjoint and perimeter <= 4 + tol."""
    if _perimeter(circles) > 4.0 + tol:
        return False
    for i, j in _PAIRS:
        dx = circles[i, 0] - circles[j, 0]
        dy = circles[i, 1] - circles[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < circles[i, 2] + circles[j, 2] - tol:
            return False
    return True


def _soft_min(values, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(values)
    return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta


def _objective_soft(params, n, W, H, beta=50.0):
    """
    Objective with soft constraints using smooth min for overlap detection.
    params = [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Enforce positive radii
    r_safe = np.maximum(r, 1e-6)
    
    obj = -np.sum(r_safe)
    
    # Boundary penalties: circles must be inside [0, W] x [0, H]
    left_viol = np.maximum(0.0, -(x - r_safe))
    obj += 1000.0 * np.sum(left_viol ** 2)
    
    right_viol = np.maximum(0.0, (x + r_safe) - W)
    obj += 1000.0 * np.sum(right_viol ** 2)
    
    bottom_viol = np.maximum(0.0, -(y - r_safe))
    obj += 1000.0 * np.sum(bottom_viol ** 2)
    
    top_viol = np.maximum(0.0, (y + r_safe) - H)
    obj += 1000.0 * np.sum(top_viol ** 2)
    
    # Overlap penalty using soft formulation
    for i, j in _PAIRS:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r_safe[i] + r_safe[j] - dist
        if overlap > 0:
            obj += 1000.0 * overlap ** 2
    
    return obj


def _gradient_soft(params, n, W, H, beta=50.0):
    """Compute gradient of the soft objective."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    r_safe = np.maximum(r, 1e-6)
    
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Gradient of -sum(r)
    gr[:] = -1.0
    
    # Boundary gradient contributions
    # Left: x - r >= 0
    left_viol = np.maximum(0.0, -(x - r_safe))
    mask = left_viol > 0
    gx[mask] += 2000.0 * left_viol[mask]
    gr[mask] -= 2000.0 * left_viol[mask]
    
    # Right: x + r <= W
    right_viol = np.maximum(0.0, (x + r_safe) - W)
    mask = right_viol > 0
    gx[mask] += 2000.0 * right_viol[mask]
    gr[mask] += 2000.0 * right_viol[mask]
    
    # Bottom: y - r >= 0
    bottom_viol = np.maximum(0.0, -(y - r_safe))
    mask = bottom_viol > 0
    gy[mask] += 2000.0 * bottom_viol[mask]
    gr[mask] -= 2000.0 * bottom_viol[mask]
    
    # Top: y + r <= H
    top_viol = np.maximum(0.0, (y + r_safe) - H)
    mask = top_viol > 0
    gy[mask] += 2000.0 * top_viol[mask]
    gr[mask] += 2000.0 * top_viol[mask]
    
    # Overlap gradient
    for i, j in _PAIRS:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1e-12:
            dist = 1e-12
        overlap = r_safe[i] + r_safe[j] - dist
        if overlap > 0:
            # d(overlap^2)/dx_i = 2 * overlap * (dx/dist)
            factor = 2000.0 * overlap / dist
            gx[i] += factor * dx
            gx[j] -= factor * dx
            gy[i] += factor * dy
            gy[j] -= factor * dy
            gr[i] += 2000.0 * overlap
            gr[j] += 2000.0 * overlap
    
    grad = np.zeros(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    
    return grad


def _grid_initialization_box(n, W, H, cols=None, rows=None, seed=42):
    """Generate grid-based initial packing in box W x H."""
    rng = np.random.RandomState(seed)
    
    if cols is None:
        aspect = W / H if H > 0 else 1.0
        if aspect > 1.0:
            cols = max(3, int(np.sqrt(n * aspect)))
        else:
            cols = max(3, int(np.sqrt(n / aspect)))
    if rows is None:
        rows = int(np.ceil(n / cols))
    
    circles = []
    for i in range(cols):
        for j in range(rows):
            if len(circles) >= n:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            r = 0.5 * min(W / cols, H / rows) * 0.9
            circles.append([cx, cy, r])
    
    return np.array(circles[:n])


def _hex_initialization_box(n, W, H, seed=42):
    """Generate hexagonal lattice initial packing."""
    rng = np.random.RandomState(seed)
    
    # Estimate number of columns based on aspect ratio
    aspect = W / H if H > 0 else 1.0
    cols = max(3, int(np.sqrt(n * aspect)))
    
    circles = []
    dx = W / cols
    dy = dx * np.sqrt(3) / 2
    
    row = 0
    while len(circles) < n:
        offset = 0.5 * dx if row % 2 == 1 else 0.0
        for col in range(cols):
            if len(circles) >= n:
                break
            cx = (col + 0.5 + offset) * dx
            cy = (row + 0.5) * dy
            if 0 <= cx <= W and 0 <= cy <= H:
                r = 0.5 * min(dx, dy) * 0.9
                circles.append([cx, cy, r])
        row += 1
    
    return np.array(circles[:n])


def _random_initialization_box(n, W, H, seed=42):
    """Generate random initial packing."""
    rng = np.random.RandomState(seed)
    
    circles = []
    max_r = min(W, H) / 4.0
    for _ in range(n):
        r = rng.uniform(0.02, max_r)
        cx = rng.uniform(r, W - r)
        cy = rng.uniform(r, H - r)
        circles.append([cx, cy, r])
    
    return np.array(circles)


def _run_optimization_fixed_box(initial_circles, W, H, start_time, max_time=10.0):
    """Run scipy optimization with fixed box dimensions."""
    n = len(initial_circles)
    x0 = initial_circles.ravel()
    
    max_r = min(W, H) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))
        bounds.append((0.0, H))
        bounds.append((0.001, max_r))
    
    best_circles = initial_circles.copy()
    best_obj = _objective_soft(x0, n, W, H)
    
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining <= 0.5:
        return best_circles, best_obj
    
    # Try L-BFGS-B with gradient
    try:
        result = minimize(
            _objective_soft, x0,
            jac=_gradient_soft,
            args=(n, W, H),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_circles = result.x.reshape(n, 3)
    except Exception:
        pass
    
    # Try Nelder-Mead on a perturbed version if time permits
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining > 3.0:
        try:
            perturbed = best_circles.copy()
            perturbed[:, 0] += np.random.randn(n) * 0.01 * W
            perturbed[:, 1] += np.random.randn(n) * 0.01 * H
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0.0, W)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0.0, H)
            
            result2 = minimize(
                _objective_soft, perturbed.ravel(),
                args=(n, W, H),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_circles = result2.x.reshape(n, 3)
        except Exception:
            pass
    
    return best_circles, best_obj


def _physics_relaxation(circles, W, H, start_time, max_time=5.0):
    """
    Physics-based relaxation using repulsive potential.
    Circles repel each other and are attracted to box center.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    dt = 0.001
    damping = 0.95
    repulsion_strength = 10.0
    wall_strength = 5.0
    
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    
    max_steps = int(remaining / dt)
    for step in range(max_steps):
        if time.time() - start_time > start_time + max_time:
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Repulsion between circles
        for i, j in _PAIRS:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                dist = 1e-6
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                force = repulsion_strength * overlap / dist
                fx[i] += force * dx
                fx[j] -= force * dx
                fy[i] += force * dy
                fy[j] -= force * dy
        
        # Wall forces
        for i in range(n):
            # Left wall
            if x[i] - r[i] < 0:
                fx[i] += wall_strength * (r[i] - x[i])
            # Right wall
            if x[i] + r[i] > W:
                fx[i] -= wall_strength * (x[i] + r[i] - W)
            # Bottom wall
            if y[i] - r[i] < 0:
                fy[i] += wall_strength * (r[i] - y[i])
            # Top wall
            if y[i] + r[i] > H:
                fy[i] -= wall_strength * (y[i] + r[i] - H)
        
        # Update positions (simple Euler with damping)
        x += dt * fx
        y += dt * fy
        
        # Apply damping by pulling toward previous position
        # (implicit in the force-based approach)
        
        # Clamp to box
        x = np.clip(x, r, W - r)
        y = np.clip(y, r, H - r)
    
    circles[:, 0] = x
    circles[:, 1] = y
    
    return circles


def construct():
    start_time = time.time()
    
    best_circles = None
    best_sum_r = -np.inf
    
    # Try multiple aspect ratios
    # W + H = 2 (since perimeter = 2*(W+H) = 4)
    # Aspect ratio a = W/H, so W = 2a/(1+a), H = 2/(1+a)
    aspect_ratios = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    for a in aspect_ratios:
        if time.time() - start_time > DEADLINE - 15.0:
            break
        
        W = 2.0 * a / (1.0 + a)
        H = 2.0 / (1.0 + a)
        
        # Try multiple initialization strategies
        inits = []
        
        # Grid initialization with various column counts
        for cols in [3, 4, 5, 6, 7]:
            rows = int(np.ceil(N / cols))
            inits.append(_grid_initialization_box(N, W, H, cols=cols, rows=rows, seed=int(a * 100) + cols))
        
        # Hexagonal initialization
        inits.append(_hex_initialization_box(N, W, H, seed=int(a * 100)))
        
        # Random initialization
        inits.append(_random_initialization_box(N, W, H, seed=int(a * 100)))
        
        for initial in inits:
            if time.time() - start_time > DEADLINE - 10.0:
                break
            
            # Run optimization
            circles, obj = _run_optimization_fixed_box(initial, W, H, start_time, max_time=8.0)
            
            # Check validity
            if _is_valid(circles, tol=1e-6):
                sum_r = _sum_radii(circles)
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_circles = circles.copy()
            
            # Try physics relaxation on the result
            if time.time() - start_time < DEADLINE - 5.0:
                try:
                    relaxed = _physics_relaxation(circles.copy(), W, H, start_time, max_time=3.0)
                    if _is_valid(relaxed, tol=1e-6):
                        sum_r_relaxed = _sum_radii(relaxed)
                        if sum_r_relaxed > best_sum_r:
                            best_sum_r = sum_r_relaxed
                            best_circles = relaxed.copy()
                except Exception:
                    pass
    
    # If we still don't have a valid solution, use a simple fallback
    if best_circles is None:
        W, H = 1.0, 1.0
        best_circles = _grid_initialization_box(N, W, H, seed=1)
        w, h = _bounding_rect(best_circles)
        perim = 2.0 * (w + h)
        if perim > 4.0:
            scale = 4.0 / perim * 0.999
            best_circles[:, 0] *= scale
            best_circles[:, 1] *= scale
            best_circles[:, 2] *= scale
    
    # Final refinement: try to increase radii slightly while maintaining validity
    if time.time() - start_time < DEADLINE - 5.0:
        try:
            improved = True
            for _ in range(100):
                if time.time() - start_time > DEADLINE - 3.0:
                    break
                if not improved:
                    break
                improved = False
                for i in range(N):
                    if time.time() - start_time > DEADLINE - 3.0:
                        break
                    test = best_circles.copy()
                    test[i, 2] *= 1.003
                    if _is_valid(test, tol=1e-7):
                        best_circles = test
                        best_sum_r = _sum_radii(best_circles)
                        improved = True
        except Exception:
            pass
    
    # Final safety: ensure perimeter <= 4 and no overlaps
    w, h = _bounding_rect(best_circles)
    perim = 2.0 * (w + h)
    if perim > 4.0:
        scale = 4.0 / perim * 0.9999
        best_circles[:, 0] *= scale
        best_circles[:, 1] *= scale
        best_circles[:, 2] *= scale
    
    # Ensure no overlaps by slight shrink if needed
    for i, j in _PAIRS:
        dx = best_circles[i, 0] - best_circles[j, 0]
        dy = best_circles[i, 1] - best_circles[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = best_circles[i, 2] + best_circles[j, 2]
        if dist < min_dist:
            factor = dist / min_dist * 0.999
            best_circles[i, 2] *= factor
            best_circles[j, 2] *= factor
    
    # Final perimeter check
    w, h = _bounding_rect(best_circles)
    perim = 2.0 * (w + h)
    if perim > 4.0:
        scale = 4.0 / perim * 0.9999
        best_circles[:, 0] *= scale
        best_circles[:, 1] *= scale
        best_circles[:, 2] *= scale
    
    return best_circles
