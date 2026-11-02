"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s limit (with 150s kill, but evaluator says 120s)


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


def _build_contact_graph(x, W, H, threshold=0.1):
    """
    Build a contact graph where edges connect circles that are close to touching.
    Returns: list of (i, j) pairs for circles that are in contact or nearly so.
    """
    n = len(x)
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            # Consider in contact if distance is within threshold of sum of radii
            if dist < min_dist + threshold:
                contacts.append((i, j))
    
    return contacts


def _force_directed_optimize(x0, W, H, max_iter=5000, tol=1e-12):
    """
    Force-directed optimization with dynamic contact graph.
    Uses spring forces for contacts and penalty forces for violations.
    """
    x = x0.copy()
    n = len(x)
    
    # Adaptive learning rate
    lr = 0.01
    best_x = x.copy()
    best_score = -np.inf
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Compute forces
        forces = np.zeros_like(x)
        
        cx = x[:, 0]
        cy = x[:, 1]
        r = x[:, 2]
        
        # Rebuild contact graph periodically
        if iteration % 50 == 0:
            contacts = _build_contact_graph(x, W, H, threshold=0.05)
        else:
            # Use previous contacts for speed
            if 'contacts' not in locals():
                contacts = _build_contact_graph(x, W, H, threshold=0.05)
        
        # Spring forces for contacts (push apart if too close, pull together if too far)
        for i, j in contacts:
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-10
            min_dist = r[i] + r[j]
            
            # Spring constant
            k = 10.0
            
            # Force magnitude: proportional to displacement from target distance
            # If dist < min_dist, push apart (positive force along dx, dy)
            # If dist > min_dist, pull together (negative force along dx, dy)
            force_mag = k * (dist - min_dist)
            
            fx = force_mag * dx / dist
            fy = force_mag * dy / dist
            
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[j, 0] -= fx
            forces[j, 1] -= fy
        
        # Boundary forces (penalty for going out of bounds)
        for i in range(n):
            # Left wall
            if cx[i] - r[i] < 0:
                forces[i, 0] += 50.0 * (0 - (cx[i] - r[i]))
            # Right wall
            if cx[i] + r[i] > W:
                forces[i, 0] -= 50.0 * (cx[i] + r[i] - W)
            # Bottom wall
            if cy[i] - r[i] < 0:
                forces[i, 1] += 50.0 * (0 - (cy[i] - r[i]))
            # Top wall
            if cy[i] + r[i] > H:
                forces[i, 1] -= 50.0 * (cy[i] + r[i] - H)
        
        # Force to increase radii (we want to maximize sum of radii)
        # Apply a small force to increase each radius
        forces[:, 2] += 0.1
        
        # Gradient descent step
        x += lr * forces
        
        # Clamp radii to be positive
        x[:, 2] = np.maximum(x[:, 2], 1e-6)
        
        # Clamp positions to stay in bounds (with small margin)
        x[:, 0] = np.clip(x[:, 0], 0, W)
        x[:, 1] = np.clip(x[:, 1], 0, H)
        
        # Evaluate and track best
        sum_r, is_valid, penalty = _compute_objective(x, W, H)
        score = sum_r if is_valid else sum_r - penalty * 1000
        
        if score > best_score:
            best_score = score
            best_x = x.copy()
        
        # Adaptive learning rate: reduce if oscillating
        if iteration % 100 == 0 and iteration > 0:
            # Simple heuristic: if score hasn't improved much, reduce lr
            if best_score - score > 0.01:
                lr *= 0.9
        
        # Convergence check
        if iteration > 100 and abs(forces).max() < tol:
            break
    
    return best_x


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


def _optimize_with_force_directed(x0, W, H, max_fd_iter=3000, max_lbfgs_iter=500):
    """
    Combined optimization: force-directed phase followed by L-BFGS-B polish.
    """
    # Phase 1: Force-directed optimization
    x_fd = _force_directed_optimize(x0, W, H, max_iter=max_fd_iter)
    
    # Phase 2: L-BFGS-B polish
    x_lbfgs, _ = _optimize_phase(x_fd, W, H, max_iter=max_lbfgs_iter, tol=1e-12)
    
    return x_lbfgs


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
    
    # Try multiple aspect ratios with force-directed optimization
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W in aspect_ratios:
        if time.time() - start_time > DEADLINE - 30:
            break
        H = 2.0 - W
        
        # Create initial grid for this aspect ratio
        x0 = _make_initial_grid(W, H)
        
        # Use force-directed + L-BFGS-B optimization
        x_opt = _optimize_with_force_directed(x0, W, H, max_fd_iter=2000, max_lbfgs_iter=300)
        
        # Check validity
        sum_r, is_valid, penalty = _compute_objective(x_opt, W, H)
        if is_valid and sum_r > best_score:
            best_score = sum_r
            best_result = x_opt.copy()
    
    # If we have time, try some random restarts with perturbation
    if best_result is not None and time.time() - start_time < DEADLINE - 15:
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Add random perturbation to best result
            x_perturbed = best_result + np.random.randn(*best_result.shape) * 0.02
            x_perturbed[:, 2] = np.maximum(x_perturbed[:, 2], 1e-6)
            
            # Determine W, H for this configuration
            cx = x_perturbed[:, 0]
            cy = x_perturbed[:, 1]
            r = x_perturbed[:, 2]
            min_x = np.min(cx - r)
            max_x = np.max(cx + r)
            min_y = np.min(cy - r)
            max_y = np.max(cy + r)
            W_p = max_x - min_x
            H_p = max_y - min_y
            
            # Scale to fit perimeter 4
            s = 2.0 / (W_p + H_p)
            if s < 1.0:
                x_perturbed = (x_perturbed - np.array([min_x, min_y, 0])) * s
            
            # Re-optimize
            x_opt_p = _optimize_with_force_directed(x_perturbed, 1.0, 1.0, max_fd_iter=1000, max_lbfgs_iter=200)
            sum_r_p, is_valid_p, penalty_p = _compute_objective(x_opt_p, 1.0, 1.0)
            
            if is_valid_p and sum_r_p > best_score:
                best_score = sum_r_p
                best_result = x_opt_p.copy()
    
    # Final: ensure validity and scale
    result = best_result if best_result is not None else _make_initial_grid(1.0, 1.0)
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
