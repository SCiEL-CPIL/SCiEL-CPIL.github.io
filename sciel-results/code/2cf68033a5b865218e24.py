"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s limit (with 150s kill, but evaluator says 120s)


def _objective_and_grad(params, W, H):
    """
    params: (2N+2) array: [cx_0..cx_N-1, cy_0..cy_N-1, r_0..r_N-1, log_scale]
    Actually let's use a cleaner parameterization.
    """
    pass


def _compute_objective(x, W, H):
    """
    x: (2N, 3) array of (cx, cy, r) for N circles.
    W, H: rectangle dimensions with 2*(W+H) = 4, so W+H = 2.
    Returns: (sum_radii, is_valid, penalty)
    """
    n = len(x)
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    # Sum of radii
    sum_r = np.sum(r)
    
    # Check boundaries
    # Circle i is valid if cx_i - r_i >= 0, cx_i + r_i <= W, cy_i - r_i >= 0, cy_i + r_i <= H
    left_ok = cx - r >= -1e-9
    right_ok = cx + r <= W + 1e-9
    bottom_ok = cy - r >= -1e-9
    top_ok = cy + r <= H + 1e-9
    
    # Check pairwise distances
    # dist(i,j) >= r_i + r_j
    is_valid = True
    penalty = 0.0
    
    for i in range(n):
        if not (left_ok[i] and right_ok[i] and bottom_ok[i] and top_ok[i]):
            is_valid = False
            # Penalty for boundary violation
            penalty += max(0, r[i] - cx[i]) ** 2
            penalty += max(0, cx[i] + r[i] - W) ** 2
            penalty += max(0, r[i] - cy[i]) ** 2
            penalty += max(0, cy[i] + r[i] - H) ** 2
        
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist - 1e-9:
                is_valid = False
                penalty += (min_dist - dist) ** 2
    
    return sum_r, is_valid, penalty


def _make_initial_grid(W, H):
    """Create a grid of circles in the rectangle."""
    # Try different grid configurations
    best_result = None
    best_score = -1
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            n_cells = rows * cols
            if n_cells < N:
                continue
            
            # Place circles in a grid
            cells = []
            for i in range(rows):
                for j in range(cols):
                    cx = (j + 0.5) * W / cols
                    cy = (i + 0.5) * H / rows
                    # Max radius is half the cell size
                    r = min(W / cols, H / rows) / 2 - 1e-4
                    cells.append([cx, cy, r])
            
            if len(cells) > N:
                cells = cells[:N]
            
            sum_r, is_valid, penalty = _compute_objective(np.array(cells), W, H)
            score = sum_r if is_valid else sum_r - penalty * 100
            
            if is_valid and score > best_score:
                best_score = score
                best_result = np.array(cells)
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) * W / side, (j + 0.5) * H / side, min(W, H) / side / 2 - 1e-4)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result


def _optimize_phase(x0, W, H, max_iter=2000, tol=1e-10):
    """Optimize circle positions and radii using scipy."""
    n = len(x0)
    
    def objective(params):
        x = params.reshape(n, 3)
        sum_r, is_valid, penalty = _compute_objective(x, W, H)
        # We want to maximize sum_r, so minimize -sum_r + penalty
        if is_valid:
            return -sum_r
        else:
            return -sum_r + penalty * 1000
    
    x0_flat = x0.flatten()
    result = minimize(objective, x0_flat, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    
    x_opt = result.x.reshape(n, 3)
    # Ensure radii are positive
    x_opt[:, 2] = np.maximum(x_opt[:, 2], 1e-6)
    
    return x_opt, result


def _try_rectangles(circles, max_time=5.0):
    """Try to find the best rectangle aspect ratio for given circles."""
    best_score = -1
    best_W = 1.0
    
    # Try different aspect ratios
    for W in np.linspace(0.5, 1.5, 20):
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Check if circles fit
        cx = circles[:, 0]
        cy = circles[:, 1]
        r = circles[:, 2]
        
        # Scale circles to fit in [0, W] x [0, H]
        # First, normalize to unit square
        max_x = np.max(cx)
        min_x = np.min(cx)
        max_y = np.max(cy)
        min_y = np.min(cy)
        
        span_x = max_x - min_x + 2 * np.max(r)
        span_y = max_y - min_y + 2 * np.max(r)
        
        # Scale factor
        scale = min(W / span_x, H / span_y)
        
        # Check if all circles fit
        new_cx = (cx - min_x + r) * scale
        new_cy = (cy - min_y + r) * scale
        new_r = r * scale
        
        # Check boundaries
        if np.min(new_cx - new_r) < -1e-6 or np.max(new_cx + new_r) > W + 1e-6:
            continue
        if np.min(new_cy - new_r) < -1e-6 or np.max(new_cy + new_r) > H + 1e-6:
            continue
        
        # Check pairwise
        valid = True
        for i in range(len(new_r)):
            for j in range(i + 1, len(new_r)):
                dx = new_cx[i] - new_cx[j]
                dy = new_cy[i] - new_cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < new_r[i] + new_r[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_W = W
    
    return best_score, best_W


def construct():
    start_time = time.time()
    
    # Start with a reasonable initial configuration
    # Try several initial aspect ratios
    best_result = None
    best_score = 0
    
    # Initial grid in a square (W=H=1)
    x0 = _make_initial_grid(1.0, 1.0)
    
    # Phase 1: Optimize in unit square
    x_opt, res = _optimize_phase(x0, 1.0, 1.0, max_iter=500, tol=1e-8)
    
    # Check validity
    sum_r, is_valid, penalty = _compute_objective(x_opt, 1.0, 1.0)
    if is_valid:
        best_score = sum_r
        best_result = x_opt.copy()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        # Not enough time for more phases
        # Scale to perimeter 4
        return _scale_to_perimeter_4(best_result if best_result is not None else x0)
    
    # Phase 2: Try different aspect ratios
    for W in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        if time.time() - start_time > DEADLINE - 15:
            break
        H = 2.0 - W
        
        # Re-initialize for this aspect ratio
        x0_ar = _make_initial_grid(W, H)
        x_opt_ar, res_ar = _optimize_phase(x0_ar, W, H, max_iter=300, tol=1e-8)
        
        sum_r_ar, is_valid_ar, penalty_ar = _compute_objective(x_opt_ar, W, H)
        if is_valid_ar and sum_r_ar > best_score:
            best_score = sum_r_ar
            best_result = x_opt_ar.copy()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return _scale_to_perimeter_4(best_result if best_result is not None else x0)
    
    # Phase 3: Fine-tune with smaller steps
    if best_result is not None:
        # Determine which W,H was used
        # We need to track this... let's just re-optimize in unit square for simplicity
        # Actually, let's keep the best result as is if it's valid
        
        # Try local perturbations
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Add small random perturbation
            x_perturbed = best_result + np.random.randn(*best_result.shape) * 0.01
            x_perturbed[:, 2] = np.maximum(x_perturbed[:, 2], 1e-6)
            
            x_opt_p, res_p = _optimize_phase(x_perturbed, 1.0, 1.0, max_iter=200, tol=1e-10)
            sum_r_p, is_valid_p, penalty_p = _compute_objective(x_opt_p, 1.0, 1.0)
            
            if is_valid_p and sum_r_p > best_score:
                best_score = sum_r_p
                best_result = x_opt_p.copy()
    
    # Final: ensure validity and scale
    result = best_result if best_result is not None else x0
    return _scale_to_perimeter_4(result)


def _scale_to_perimeter_4(circles):
    """Scale circles so their bounding rectangle has perimeter at most 4."""
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Find bounding box
    min_x = np.min(cx - r)
    max_x = np.max(cx + r)
    min_y = np.min(cy - r)
    max_y = np.max(cy + r)
    
    W = max_x - min_x
    H = max_y - min_y
    
    if W <= 0 or H <= 0:
        # Degenerate case
        scale = 1.0
    else:
        # Perimeter = 2*(W+H), we want 2*(W*H_scale) <= 4, so W*H_scale <= 2
        # Scale factor s: 2*(W*s + H*s) <= 4 => s*(W+H) <= 2 => s <= 2/(W+H)
        s = 2.0 / (W + H)
        scale = min(s, 1.0)  # Don't scale up if already fits
    
    # Apply scale
    new_cx = (cx - min_x) * scale
    new_cy = (cy - min_y) * scale
    new_r = r * scale
    
    return np.column_stack([new_cx, new_cy, new_r])


# Quick test
if __name__ == "__main__":
    result = construct()
    print("Result shape:", result.shape)
    print("Sum of radii:", np.sum(result[:, 2]))
    
    # Verify perimeter
    cx = result[:, 0]
    cy = result[:, 1]
    r = result[:, 2]
    min_x = np.min(cx - r)
    max_x = np.max(cx + r)
    min_y = np.min(cy - r)
    max_y = np.max(cy + r)
    W = max_x - min_x
    H = max_y - min_y
    print(f"Bounding box: {W:.4f} x {H:.4f}, perimeter: {2*(W+H):.4f}")
    
    # Verify no overlaps
    valid = True
    for i in range(len(r)):
        if cx[i] - r[i] < -1e-6 or cx[i] + r[i] > W + 1e-6:
            valid = False
            print(f"Circle {i} out of bounds (x)")
        if cy[i] - r[i] < -1e-6 or cy[i] + r[i] > H + 1e-6:
            valid = False
            print(f"Circle {i} out of bounds (y)")
        for j in range(i + 1, len(r)):
            dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            if dist < r[i] + r[j] - 1e-6:
                valid = False
                print(f"Circles {i} and {j} overlap: dist={dist:.6f}, min={r[i]+r[j]:.6f}")
    
    print("Valid:", valid)
