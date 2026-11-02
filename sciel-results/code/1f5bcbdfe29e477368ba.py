"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_valid(circles):
    """Check if circles are valid: inside rectangle with perimeter <= 4, disjoint interiors."""
    if len(circles) != N:
        return False
    if np.any(circles[:, 2] < 1e-9):
        return False
    # Find bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    W = xmax - xmin
    H = ymax - ymin
    if W <= 0 or H <= 0:
        return False
    if W + H > 2.0 + 1e-7:  # perimeter = 2(W+H) <= 4
        return False
    # Check pairwise non-overlap (disjoint interiors: distance >= r_i + r_j)
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-7:
                return False
    return True


def _penalty(circles, W, H):
    """Penalty for violations: circles outside [0,W]x[0,H] and overlapping pairs."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    pen = 0.0
    # Boundary violations
    pen += np.sum(np.maximum(0, rs - xs) ** 2)
    pen += np.sum(np.maximum(0, rs - (W - xs)) ** 2)
    pen += np.sum(np.maximum(0, rs - ys) ** 2)
    pen += np.sum(np.maximum(0, rs - (H - ys)) ** 2)
    # Pairwise overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                pen += overlap ** 2
    return pen


def _objective(params, W, H):
    """Minimize negative sum of radii plus penalties."""
    n = N
    xs = params[:n]
    ys = params[n:2*n]
    rs = params[2*n:3*n]
    circles = np.column_stack([xs, ys, rs])
    obj = -np.sum(rs) + 10.0 * _penalty(circles, W, H)
    return obj


def _make_initial(W, H):
    """Create initial grid-like configuration."""
    # Place circles in a grid pattern
    cols = int(np.ceil(np.sqrt(n * W / H))) if H > 0 else 5
    cols = max(3, min(7, cols))
    rows = int(np.ceil(N / cols))
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    for k in range(N):
        i = k % cols
        j = k // cols
        xs[k] = (i + 0.5) * W / cols
        ys[k] = (j + 0.5) * H / rows
        # Initial radius estimate
        rs[k] = 0.5 * min(W / cols, H / rows) * 0.95
    
    return np.concatenate([xs, ys, rs])


def _local_optimize(W, H, x0, max_iter=500):
    """Run local optimization for a given rectangle size."""
    n = N
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))  # x
    for i in range(n):
        bounds.append((1e-6, H - 1e-6))  # y
    for i in range(n):
        bounds.append((1e-6, min(W, H) * 0.5))  # r
    
    result = minimize(_objective, x0, args=(W, H), method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    return result.x, result.fun


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 115s for safety
    
    best_circles = None
    best_score = -np.inf
    
    # Try various aspect ratios
    # Perimeter = 2(W+H) = 4, so W+H = 2
    # W/H ranges from about 0.3 to 3.0
    aspect_ratios = []
    for ar in np.linspace(0.4, 2.5, 12):
        W = 2.0 * ar / (1 + ar)
        H = 2.0 / (1 + ar)
        aspect_ratios.append((W, H))
    
    # Also try square
    aspect_ratios.append((1.0, 1.0))
    
    for W, H in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        x0 = _make_initial(W, H)
        
        # Run local optimization
        x_opt, fun = _local_optimize(W, H, x0, max_iter=300)
        
        if time.time() > deadline - 10.0:
            break
        
        circles = np.column_stack([x_opt[:N], x_opt[N:2*N], x_opt[2*N:3*N]])
        
        # Check validity
        if _is_valid(circles):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    # If no valid solution found, try a fallback grid
    if best_circles is None:
        # Fallback: simple grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    # Final polish: try to improve the best solution with more iterations
    if time.time() < deadline - 20.0 and best_circles is not None:
        xs = best_circles[:, 0]
        ys = best_circles[:, 1]
        rs = best_circles[:, 2]
        
        # Find bounding box
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        W = xmax - xmin
        H = ymax - ymin
        
        # Shift to [0,W]x[0,H]
        xs_shifted = xs - xmin
        ys_shifted = ys - ymin
        
        x0 = np.concatenate([xs_shifted, ys_shifted, rs])
        
        # Polish with tighter tolerance
        x_opt, fun = _local_optimize(W, H, x0, max_iter=200)
        circles = np.column_stack([x_opt[:N], x_opt[N:2*N], x_opt[2*N:3*N]])
        
        if _is_valid(circles):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    return best_circles
# EVOLVE-BLOCK-END
