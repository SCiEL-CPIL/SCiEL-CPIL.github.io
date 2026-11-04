"""Circle packing in perimeter-4 rectangle: projected Gauss-Seidel with contact graph."""
import time
import numpy as np

N = 21
DEADLINE = 115.0  # safety margin before 120s limit

def _check_validity(circles, W, H, tol=1e-5):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True

def _compute_bounding_rect(circles):
    """Compute bounding rectangle dimensions from circles."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    return max_x - min_x, max_y - min_y

def _project_constraints(circles, W, H, max_iter=2000):
    """
    Iterative projected Gauss-Seidel to enforce non-overlap and boundary constraints.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    for it in range(max_iter):
        moved = False
        
        # Boundary constraints: clamp positions and shrink radii if needed
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-6
            # Clamp x
            x[i] = max(r[i], min(W - r[i], x[i]))
            # Clamp y
            y[i] = max(r[i], min(H - r[i], y[i]))
            # Shrink radius if it doesn't fit
            max_r = min(x[i], W - x[i], y[i], H - y[i])
            if r[i] > max_r:
                r[i] = max(max_r, 1e-6)
        
        # Non-overlap constraints: pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    if dist < 1e-12:
                        # Random nudge for coincident points
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-8
                        dy = np.sin(angle) * 1e-8
                        dist = 1e-8
                    
                    overlap = min_dist - dist
                    # Weighted push based on relative radii
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r
                    wj = r[j] / total_r
                    
                    push_x = overlap * dx / dist
                    push_y = overlap * dy / dist
                    
                    x[i] += wi * push_x
                    y[i] += wi * push_y
                    x[j] -= wj * push_x
                    y[j] -= wj * push_y
                    moved = True
                    
                    # Re-clamp after push
                    x[i] = max(r[i], min(W - r[i], x[i]))
                    y[i] = max(r[i], min(H - r[i], y[i]))
                    x[j] = max(r[j], min(W - r[j], x[j]))
                    y[j] = max(r[j], min(H - r[j], y[j]))
                    
                    # Re-shrink radii if needed after re-clamp
                    for k in [i, j]:
                        max_r = min(x[k], W - x[k], y[k], H - y[k])
                        if r[k] > max_r:
                            r[k] = max(max_r, 1e-6)
        
        if not moved:
            break
    
    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid

def _init_grid(W, H, rows, cols, seed_offset=0):
    """Create a grid of circles in the rectangle."""
    cells = []
    for i in range(rows):
        for j in range(cols):
            if len(cells) >= N:
                break
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            r = min(W / cols, H / rows) / 2 - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])

def _init_random(W, H, rng):
    """Random initialization with small radii."""
    circles = []
    max_r = min(W, H) / 6
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)

def _grow_radii(circles, W, H, max_iter=500):
    """
    Iteratively grow radii while maintaining constraints.
    Uses projected Gauss-Seidel to resolve conflicts.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(circles)
    
    for it in range(max_iter):
        # Try to grow each radius
        grew = False
        for i in range(n):
            # Max possible radius at current position
            max_r_pos = min(x[i], W - x[i], y[i], H - y[i])
            
            # Check against all other circles
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_pair = dist - r[j]
                max_r_pos = min(max_r_pos, max_r_pair)
            
            # Try to grow
            new_r = min(max_r_pos, r[i] * 1.02 + 1e-5)
            if new_r > r[i] + 1e-8:
                r[i] = new_r
                grew = True
        
        # Resolve constraints
        circles_temp = np.column_stack([x, y, r])
        circles_proj, valid = _project_constraints(circles_temp, W, H, max_iter=200)
        
        if valid:
            x = circles_proj[:, 0]
            y = circles_proj[:, 1]
            r = circles_proj[:, 2]
        else:
            # If projection failed, revert growth and break
            break
        
        if not grew:
            break
    
    return np.column_stack([x, y, r])

def construct():
    start_time = time.time()
    best_circles = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Aspect ratios to try: w/h where w+h=2
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Compute W, H from aspect ratio and w+h=2
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Try multiple initializations
        inits = []
        
        # Grid inits
        inits.append(_init_grid(W, H, 3, 7))
        inits.append(_init_grid(W, H, 7, 3))
        inits.append(_init_grid(W, H, 4, 5))
        inits.append(_init_grid(W, H, 5, 4))
        
        # Random inits
        for _ in range(3):
            inits.append(_init_random(W, H, rng))
        
        for init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Project constraints
            circles, valid = _project_constraints(init, W, H, max_iter=3000)
            
            if valid:
                # Grow radii
                circles_grown = _grow_radii(circles, W, H, max_iter=300)
                
                # Final validation
                final_valid = _check_validity(circles_grown, W, H)
                if final_valid:
                    score = np.sum(circles_grown[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles_grown.copy()
            
            # Also try just the projected version without growth
            if valid:
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
    
    # Local refinement: perturb best solution and re-optimize
    if best_circles is not None and time.time() - start_time < DEADLINE - 10:
        W, H = _compute_bounding_rect(best_circles)
        # Ensure perimeter constraint
        if 2.0 * (W + H) > 4.0:
            scale = 2.0 / (W + H)
            best_circles[:, 0] *= scale
            best_circles[:, 1] *= scale
            best_circles[:, 2] *= scale
            W, H = _compute_bounding_rect(best_circles)
        
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            perturbed = best_circles.copy()
            perturbed[:, 0] += np.random.normal(0, 0.01, N)
            perturbed[:, 1] += np.random.normal(0, 0.01, N)
            
            circles, valid = _project_constraints(perturbed, W, H, max_iter=3000)
            if valid:
                circles_grown = _grow_radii(circles, W, H, max_iter=200)
                final_valid = _check_validity(circles_grown, W, H)
                if final_valid:
                    score = np.sum(circles_grown[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles_grown.copy()
    
    # Final safety: ensure valid output
    if best_circles is None:
        # Fallback: simple grid
        W, H = 1.0, 1.0
        best_circles = _init_grid(W, H, 3, 7)
        best_circles, _ = _project_constraints(best_circles, W, H, max_iter=3000)
    
    # Final validation
    if not _check_validity(best_circles, W, H):
        # Last resort: shrink all radii
        for i in range(N):
            best_circles[i, 2] *= 0.9
        best_circles, valid = _project_constraints(best_circles, W, H, max_iter=1000)
    
    return best_circles
