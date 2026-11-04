<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _generate_hexagonal_grid():
    """
    Generate a hexagonal arrangement of 21 circles.
    Rows: 3, 4, 5, 4, 3, 2 = 21 circles.
    """
    points = []
    # Hexagonal spacing
    # Let's just use a standard grid and let the optimizer fix it,
    # but a good initial guess helps.
    # 5x5 grid minus 4 corners is a good start.
    # Or 3-4-5-4-3-2
    
    # Let's use a 5x5 grid, remove 4 corners to get 21.
    # This is a square-ish start.
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            # Skip corners (0,0), (0,4), (4,0), (4,4)
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            x = (i + 0.5)
            y = (j + 0.5)
            cells.append([x, y])
            
    # Scale to fit in a unit square roughly
    xs = np.array([c[0] for c in cells])
    ys = np.array([c[1] for c in cells])
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    # Normalize to [0, 1]
    xs = (xs - x_min) / (x_max - x_min)
    ys = (ys - y_min) / (y_max - y_min)
    
    # Initial radius: 1/side
    r = 0.5 / side
    
    init = []
    for i in range(len(cells)):
        init.append([xs[i], ys[i], r])
        
    return np.array(init)

def _validate(circles, W, H):
    """Check if circles are within [0, W]x[0, H] and non-overlapping."""
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return False
        
    if np.any(xs - rs < -1e-9) or np.any(xs + rs > W + 1e-9):
        return False
    if np.any(ys - rs < -1e-9) or np.any(ys + rs > H + 1e-9):
        return False
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True

def _compute_score(circles):
    """Compute sum of radii if valid in bounding box with perim <= 4, else 0."""
    n = len(circles)
    if n != N:
        return 0.0
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return 0.0
        
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + 1e-9:
        return 0.0
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return 0.0
    return np.sum(rs)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Initial guess
    init_grid = _generate_hexagonal_grid()
    
    # We will optimize for a fixed aspect ratio alpha = W/H
    # 2(W+H) = 4 => W+H = 2 => H = 2/(alpha+1), W = 2*alpha/(alpha+1)
    
    # Try several aspect ratios
    alphas = [1.0, 0.8, 1.2, 0.5, 2.0, 0.7, 1.4]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        W = 2 * alpha / (alpha + 1)
        H = 2 / (alpha + 1)
        
        # Scale initial grid to fit W x H
        init = init_grid.copy()
        xs = init[:, 0]
        ys = init[:, 1]
        
        # Normalize xs to [0, 1] and ys to [0, 1]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        
        if x_max > x_min:
            xs = (xs - x_min) / (x_max - x_min)
        else:
            xs = np.zeros_like(xs)
            
        if y_max > y_min:
            ys = (ys - y_min) / (y_max - y_min)
        else:
            ys = np.zeros_like(ys)
            
        # Scale to W and H
        # Add a margin so they don't start on the boundary
        margin = 0.05
        xs = xs * (W * (1 - 2*margin)) + W * margin
        ys = ys * (H * (1 - 2*margin)) + H * margin
        
        # Initial radii: small
        rs = np.full(N, 0.05)
        
        init_vec = np.concatenate([xs, ys, rs])
        
        def objective(v):
            xs = v[0:N]
            ys = v[N:2*N]
            rs = v[2*N:3*N]
            return -np.sum(rs)
            
        def constraint_non_overlap(i, j):
            def con(v):
                xs = v[0:N]
                ys = v[N:2*N]
                rs = v[2*N:3*N]
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                return dist - (rs[i] + rs[j])
            return con
            
        constraints = []
        # Boundary constraints
        # x - r >= 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (W - v[i]) - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (H - v[N+i]) - v[2*N+i]})
            
        # Non-overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({'type': 'ineq', 'fun': constraint_non_overlap(i, j)})
                
        # Radii > 0 handled by bounds
        bounds = [(0, W) for _ in range(N)] + [(0, H) for _ in range(N)] + [(1e-6, 1) for _ in range(N)]
        
        try:
            result = minimize(
                objective,
                init_vec,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if time.time() > deadline:
                break
                
            if result.success:
                circles = result.x.reshape(N, 3)
                # Reorder to x, y, r
                xs = result.x[0:N]
                ys = result.x[N:2*N]
                rs = result.x[2*N:3*N]
                circles = np.column_stack((xs, ys, rs))
                
                score = _compute_score(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
                    
        except Exception as e:
            pass
            
    # If no valid solution found, fallback to grid
    if best_circles is None:
        best_circles = _generate_hexagonal_grid()
        # Scale to perimeter 4
        xs = best_circles[:, 0]
        ys = best_circles[:, 1]
        rs = best_circles[:, 2]
        left = np.min(xs - rs)
        right = np.max(xs + rs)
        bottom = np.min(ys - rs)
        top = np.max(ys + rs)
        width = right - left
        height = top - bottom
        perim = 2 * (width + height)
        if perim > 4:
            scale = 4 / perim
            best_circles[:, 0] *= scale
            best_circles[:, 1] *= scale
            best_circles[:, 2] *= scale
            
    return best_circles
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

def _check_validity(circles):
    """Check if circles are non-overlapping and have positive radii."""
    n = len(circles)
    if n != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 1e-6):
        return False
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            # Use squared distance to avoid sqrt in tight loops if possible, 
            # but here we need actual distance for overlap check
            # dist < rs[i] + rs[j]  <=> dist_sq < (rs[i]+rs[j])^2
            sum_r = rs[i] + rs[j]
            if dist_sq < sum_r * sum_r - 1e-9:
                return False
    return True

def _compute_score(circles):
    """Compute sum of radii if valid in bounding box with perim <= 4, else 0."""
    if not _check_validity(circles):
        return 0.0
        
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + 1e-9:
        return 0.0
        
    return np.sum(rs)

def _get_bounding_box(circles):
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    return left, right, bottom, top

def _normalize_and_scale(circles, W, H):
    """
    Normalize circles to fit within [0, W] x [0, H] while preserving relative positions.
    Returns new circles array.
    """
    left, right, bottom, top = _get_bounding_box(circles)
    width = right - left
    height = top - bottom
    
    if width < 1e-9 or height < 1e-9:
        # Degenerate case, just center them
        new_circles = circles.copy()
        new_circles[:, 0] = W / 2
        new_circles[:, 1] = H / 2
        return new_circles

    scale_x = W / width
    scale_y = H / height
    scale = min(scale_x, scale_y)
    
    new_circles = circles.copy()
    # Translate so min is at 0
    new_circles[:, 0] -= left
    new_circles[:, 1] -= bottom
    # Scale
    new_circles[:, 0] *= scale
    new_circles[:, 1] *= scale
    new_circles[:, 2] *= scale
    
    # Center in W x H
    new_left, new_right, new_bottom, new_top = _get_bounding_box(new_circles)
    new_width = new_right - new_left
    new_height = new_top - new_bottom
    new_circles[:, 0] += (W - new_width) / 2
    new_circles[:, 1] += (H - new_height) / 2
    
    return new_circles

def _local_optimize(circles, W, H, time_limit=5.0):
    """
    Perform local optimization using SLSQP to maximize sum of radii 
    given fixed W, H.
    """
    start_time = time.time()
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    init_vec = np.concatenate([xs, ys, rs])
    
    def objective(v):
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        return -np.sum(rs)
        
    constraints = []
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (W - v[i]) - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (H - v[N+i]) - v[2*N+i]})
        
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            def con(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[N+i] - v[N+j]
                dist = np.sqrt(dx * dx + dy * dy)
                return dist - (v[2*N+i] + v[2*N+j])
            constraints.append({'type': 'ineq', 'fun': con})
            
    bounds = [(0, W) for _ in range(N)] + [(0, H) for _ in range(N)] + [(1e-6, max(W, H)) for _ in range(N)]
    
    try:
        result = minimize(
            objective,
            init_vec,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if result.success:
            new_xs = result.x[0:N]
            new_ys = result.x[N:2*N]
            new_rs = result.x[2*N:3*N]
            new_circles = np.column_stack((new_xs, new_ys, new_rs))
            return new_circles
    except Exception:
        pass
        
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (limit 150s)
    
    best_score = 0.0
    best_circles = None
    
    # Try a range of aspect ratios
    # Perimeter 4 => W+H=2. Let alpha = W/H. W = 2*alpha/(1+alpha), H = 2/(1+alpha)
    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    # Generate a few random initial configurations
    num_initials = 5
    
    for alpha_idx, alpha in enumerate(alphas):
        if time.time() > deadline:
            break
            
        W = 2 * alpha / (alpha + 1)
        H = 2 / (alpha + 1)
        
        for k in range(num_initials):
            if time.time() > deadline:
                break
                
            # Random initial positions
            # Ensure they fit in the box initially
            # Use small random radii
            rs = np.random.uniform(0.05, 0.1, N)
            xs = np.random.uniform(0, W, N)
            ys = np.random.uniform(0, H, N)
            
            # A more structured initial guess: grid-like
            if k == 0:
                # 5x5 grid minus corners
                side = 5
                cells = []
                for i in range(side):
                    for j in range(side):
                        if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                            continue
                        x = (i + 0.5)
                        y = (j + 0.5)
                        cells.append([x, y])
                
                xs_init = np.array([c[0] for c in cells])
                ys_init = np.array([c[1] for c in cells])
                
                # Normalize to [0,1]
                x_min, x_max = np.min(xs_init), np.max(xs_init)
                y_min, y_max = np.min(ys_init), np.max(ys_init)
                if x_max > x_min: xs_init = (xs_init - x_min) / (x_max - x_min)
                if y_max > y_min: ys_init = (ys_init - y_min) / (y_max - y_min)
                
                # Scale to box with margin
                margin = 0.1
                xs = xs_init * (W * (1 - 2*margin)) + W * margin
                ys = ys_init * (H * (1 - 2*margin)) + H * margin
                rs = np.full(N, 0.08) # Uniform small radius
            else:
                # Random
                pass
            
            init_circles = np.column_stack((xs, ys, rs))
            
            # Check if initial is valid
            if not _check_validity(init_circles):
                # Try to separate overlapping circles slightly
                # Simple relaxation
                for _ in range(10):
                    moved = False
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            dist = np.sqrt(dx * dx + dy * dy)
                            sum_r = rs[i] + rs[j]
                            if dist < sum_r:
                                # Push apart
                                if dist < 1e-9:
                                    dx = 0.1
                                    dy = 0.1
                                    dist = np.sqrt(0.02)
                                push = (sum_r - dist) / 2 + 0.01
                                xs[i] += (dx / dist) * push
                                xs[j] -= (dx / dist) * push
                                ys[i] += (dy / dist) * push
                                ys[j] -= (dy / dist) * push
                                moved = True
                    if not moved:
                        break
                init_circles = np.column_stack((xs, ys, rs))
                if not _check_validity(init_circles):
                    continue
                    
            # Optimize
            opt_circles = _local_optimize(init_circles, W, H)
            
            # Validate result
            score = _compute_score(opt_circles)
            if score > best_score:
                best_score = score
                best_circles = opt_circles.copy()
                
    # If no valid solution found, return a valid fallback
    if best_circles is None:
        # Create a simple valid configuration
        # 21 circles in a 1x1 square (perimeter 4)
        # 5x5 grid minus corners
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                    continue
                x = (i + 0.5)
                y = (j + 0.5)
                cells.append([x, y])
        
        xs = np.array([c[0] for c in cells])
        ys = np.array([c[1] for c in cells])
        rs = np.full(N, 0.1) # 0.1 * 2 = 0.2 diameter, spacing 1.0, fits
        
        # Normalize to 1x1
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        xs = (xs - x_min) / (x_max - x_min)
        ys = (ys - y_min) / (y_max - y_min)
        
        # Scale down slightly to ensure no overlap and fit
        # Max radius in 1x1 with spacing 1/5 = 0.2 is 0.1
        # But we removed corners, so edge circles have more room?
        # Let's just use a very small radius to be safe
        rs = np.full(N, 0.05)
        
        best_circles = np.column_stack((xs, ys, rs))
        
    return best_circles
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
