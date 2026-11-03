"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(pts):
    """Compute the radius for a given set of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Wall constraint
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    r_wall = np.min(wall)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_wall, r_pair)


def _soft_min_val(x, k=100.0):
    """Soft minimum approximation value."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    min_val = np.min(x)
    shifted = x - min_val
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return min_val - log_sum / k


def _soft_min_grad(x, k=100.0):
    """Gradient of soft min with respect to x."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([])
    min_val = np.min(x)
    shifted = x - min_val
    exp_terms = np.exp(-k * shifted)
    sum_exp = np.sum(exp_terms)
    # d(soft_min)/dx_i = -1/sum_exp * (-k) * exp(-k*x_i) = k * exp(-k*x_i) / sum_exp
    # But soft_min = min_val - log(sum_exp)/k, so d(soft_min)/dx_i = -1/k * (1/sum_exp) * (-k) * exp(-k*x_i)
    # = exp(-k*x_i) / sum_exp
    # Wait, let me redo: soft_min = min - (1/k)log(sum(exp(-k*(x_i - min))))
    # d(soft_min)/dx_j = - (1/k) * (1/sum) * (-k) * exp(-k*(x_j-min)) = exp(-k*(x_j-min))/sum
    grad = exp_terms / sum_exp
    return grad


def _compute_constraints(pts):
    """Compute all constraint values and return them along with indices."""
    n = len(pts)
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    dist_half = dist / 2.0
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_vals = dist_half[iu]
    
    # Wall constraints
    wall_x = 0.5 - np.abs(pts[:, 0])
    wall_y = 0.5 - np.abs(pts[:, 1])
    
    all_vals = np.concatenate([pair_vals, wall_x, wall_y])
    return all_vals, iu, n


def _objective(pts, k=50.0):
    """Compute soft-min objective and gradient."""
    all_vals, iu, n = _compute_constraints(pts)
    
    soft_min_val = _soft_min_val(all_vals, k=k)
    grad_all = _soft_min_grad(all_vals, k=k)
    
    # Objective is -soft_min (we want to maximize soft_min, so minimize negative)
    obj = -soft_min_val
    
    # Build gradient with respect to pts
    grad = np.zeros_like(pts)
    
    # Pairwise part: grad_all[:len(pair_vals)]
    n_pairs = len(iu[0])
    pair_grad = grad_all[:n_pairs]
    
    # For each pair (i,j), the contribution to grad[i] and grad[j]
    # dist(i,j)/2 derivative w.r.t. pts[i] = (pts[i] - pts[j]) / dist(i,j)
    # dist(i,j)/2 derivative w.r.t. pts[j] = (pts[j] - pts[i]) / dist(i,j)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_full = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_full, np.inf)
    
    # For pairs
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        d = dist_full[i, j]
        if d > 1e-15:
            # d(dist_half)/d(pts[i]) = (pts[i] - pts[j]) / (2 * d)
            # But we're using soft_min of dist_half, and d(soft_min)/d(dist_half) = pair_grad[idx]
            # So contribution to grad[i] = pair_grad[idx] * (pts[i] - pts[j]) / (2*d)
            # And we're maximizing, so grad[i] += pair_grad[idx] * (pts[i]-pts[j])/(2*d)
            # But obj = -soft_min, so d(obj)/d(pts[i]) = -pair_grad[idx] * (pts[i]-pts[j])/(2*d)
            # We want to do gradient ascent on soft_min, so we add pair_grad[idx] * (pts[i]-pts[j])/(2*d)
            factor = pair_grad[idx] / (2 * d)
            grad[i] += factor * (pts[i] - pts[j])
            grad[j] += factor * (pts[j] - pts[i])
    
    # Wall x part: grad_all[n_pairs:n_pairs+n]
    wall_x_grad = grad_all[n_pairs:n_pairs + n]
    # d(0.5 - |x_i|)/dx_i = -sign(x_i)
    # So contribution: wall_x_grad[i] * (-sign(x_i))
    # For obj = -soft_min: d(obj)/dx_i = -wall_x_grad[i] * (-sign(x_i)) = wall_x_grad[i] * sign(x_i)
    # For gradient ascent on soft_min: grad[i,0] += wall_x_grad[i] * (-sign(x_i))
    signs_x = np.sign(pts[:, 0])
    signs_x[signs_x == 0] = 1.0
    grad[:, 0] += wall_x_grad * (-signs_x)
    
    # Wall y part
    wall_y_grad = grad_all[n_pairs + n:]
    signs_y = np.sign(pts[:, 1])
    signs_y[signs_y == 0] = 1.0
    grad[:, 1] += wall_y_grad * (-signs_y)
    
    return obj, grad


def _initial_grid():
    """Create initial 7x7 grid plus one extra point."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    pts = np.vstack([grid, extra])
    return pts[:N].copy()


def construct():
    start_time = time.time()
    deadline = 135.0  # Leave 15s safety margin
    
    pts = _initial_grid()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Gradient ascent with adaptive step
    step_size = 0.02
    k = 30.0  # Soft-min parameter
    patience = 0
    max_patience = 200
    consecutive_no_improve = 0
    
    while time.time() < deadline:
        obj, grad = _objective(pts, k=k)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Normalize and step
        pts_new = pts + step_size * grad / grad_norm
        pts_new = np.clip(pts_new, -0.49999, 0.49999)
        
        new_score = _score(pts_new)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_pts = pts_new.copy()
            patience = 0
            consecutive_no_improve = 0
            pts = pts_new.copy()
            step_size = min(step_size * 1.02, 0.05)
        else:
            patience += 1
            consecutive_no_improve += 1
            step_size *= 0.92
            if patience > max_patience or consecutive_no_improve > 500:
                break
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Phase 2: Finer refinement
    if time.time() < deadline - 5:
        step_size = 0.005
        k = 50.0
        patience = 0
        max_patience = 500
        pts = best_pts.copy()
        
        while time.time() < deadline - 5:
            obj, grad = _objective(pts, k=k)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            pts_new = pts + step_size * grad / grad_norm
            pts_new = np.clip(pts_new, -0.49999, 0.49999)
            
            new_score = _score(pts_new)
            
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_pts = pts_new.copy()
                patience = 0
                pts = pts_new.copy()
            else:
                patience += 1
                step_size *= 0.95
                if patience > max_patience:
                    break
            
            if time.time() > deadline - 5:
                break
    
    # Phase 3: Very fine polish
    if time.time() < deadline - 3:
        step_size = 0.001
        k = 100.0
        patience = 0
        max_patience = 200
        pts = best_pts.copy()
        
        while time.time() < deadline - 3:
            obj, grad = _objective(pts, k=k)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-13:
                break
            
            pts_new = pts + step_size * grad / grad_norm
            pts_new = np.clip(pts_new, -0.49999, 0.49999)
            
            new_score = _score(pts_new)
            
            if new_score > best_score + 1e-14:
                best_score = new_score
                best_pts = pts_new.copy()
                patience = 0
                pts = pts_new.copy()
            else:
                patience += 1
                step_size *= 0.9
                if patience > max_patience:
                    break
            
            if time.time() > deadline - 3:
                break
    
    result = best_pts.copy()
    result = np.clip(result, -0.4999999, 0.4999999)
    
    return result
# EVOLVE-BLOCK-END
