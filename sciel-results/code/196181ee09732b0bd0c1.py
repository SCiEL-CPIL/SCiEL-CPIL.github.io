"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _build_contact_graph(x, y, r, tol=0.01):
    """Build contact graph: pairs of circles that are in contact (distance <= r_i + r_j + tol)."""
    n = len(x)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist <= r[i] + r[j] + tol:
                contacts.append((i, j))
    return contacts


def _force_directed_packing(N, w, h, n_iters=8000, lr_start=0.02, lr_end=0.0005, 
                            contact_tol=0.02, spring_k=5.0, overlap_k=20.0, 
                            wall_k=10.0, center_k=0.01):
    """
    Force-directed layout for circle packing in a w x h rectangle.
    Uses contact graph with spring forces for touching pairs.
    Returns (x, y, r) arrays.
    """
    # Initialize: grid layout with equal radii
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.full(N, min(w / cols, h / rows) * 0.35)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
    
    # Force-directed relaxation with contact graph
    for it in range(n_iters):
        # Learning rate schedule: exponential decay
        t = it / n_iters
        lr = lr_start * (lr_end / lr_start) ** t
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-circle interactions: use contact graph approach
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < 1e-12:
                    dx = np.random.randn() * 1e-6
                    dy = np.random.randn() * 1e-6
                    dist = np.sqrt(dx * dx + dy * dy)
                
                if dist < min_dist:
                    # Overlapping: strong repulsive force to push apart
                    overlap = min_dist - dist
                    force = overlap_k * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist + contact_tol:
                    # Near contact: spring force to maintain exact contact distance
                    # This creates rigid contact graph structure
                    diff = dist - min_dist
                    force = spring_k * diff
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                # else: no force (non-overlapping, non-contacting)
        
        # Boundary forces: repulsion from walls
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                penetration = r[i] - x[i]
                fx[i] += wall_k * penetration
            # Right wall
            if x[i] > w - r[i]:
                penetration = x[i] - (w - r[i])
                fx[i] -= wall_k * penetration
            # Bottom wall
            if y[i] < r[i]:
                penetration = r[i] - y[i]
                fy[i] += wall_k * penetration
            # Top wall
            if y[i] > h - r[i]:
                penetration = y[i] - (h - r[i])
                fy[i] -= wall_k * penetration
        
        # Weak centering force to prevent drift
        cx = w / 2.0
        cy = h / 2.0
        for i in range(N):
            fx[i] -= center_k * (x[i] - cx)
            fy[i] -= center_k * (y[i] - cy)
        
        # Update positions with velocity damping
        x += lr * fx
        y += lr * fy
        
        # Clamp to valid range
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    # After relaxation, try to grow radii iteratively
    # Use binary search approach: try growing, if invalid, revert and stop
    for grow_iter in range(100):
        # Try to grow all radii by a small factor
        growth = 1.0 + 0.005
        r_new = r * growth
        
        # Check validity
        valid = True
        for i in range(N):
            if x[i] < r_new[i] or x[i] > w - r_new[i]:
                valid = False
                break
            if y[i] < r_new[i] or y[i] > h - r_new[i]:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r_new[i] + r_new[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            r = r_new
        else:
            break
    
    return x, y, r


def _validate_packing(x, y, r, w, h, tol=1e-6):
    """Check if packing is valid."""
    n = len(x)
    for i in range(n):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    
    return True


def _refine_with_scipy(x, y, r, w, h, max_iter=50):
    """
    Use scipy.optimize to refine positions for better packing.
    Objective: minimize sum of overlaps (should be zero for valid packing).
    """
    n = len(x)
    
    def objective(params):
        x_new = params[:n]
        y_new = params[n:2*n]
        
        # Penalty for boundary violations
        penalty = 0.0
        for i in range(n):
            if x_new[i] < r[i]:
                penalty += 100.0 * (r[i] - x_new[i]) ** 2
            if x_new[i] > w - r[i]:
                penalty += 100.0 * (x_new[i] - (w - r[i])) ** 2
            if y_new[i] < r[i]:
                penalty += 100.0 * (r[i] - y_new[i]) ** 2
            if y_new[i] > h - r[i]:
                penalty += 100.0 * (y_new[i] - (h - r[i])) ** 2
        
        # Penalty for circle-circle overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x_new[i] - x_new[j]
                dy = y_new[i] - y_new[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += 10.0 * overlap ** 2
        
        return penalty
    
    # Initial parameters
    x0 = np.concatenate([x, y])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.success or result.fun < 1e-8:
            x_refined = result.x[:n]
            y_refined = result.x[n:2*n]
            
            # Clamp to valid range
            for i in range(n):
                x_refined[i] = np.clip(x_refined[i], r[i], w - r[i])
                y_refined[i] = np.clip(y_refined[i], r[i], h - r[i])
            
            # Validate
            if _validate_packing(x_refined, y_refined, r, w, h):
                return x_refined, y_refined
    except Exception:
        pass
    
    return x, y


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Use force-directed contact graph layout with scipy refinement.
    """
    best_val = -1
    best_result = None
    
    # Try multiple aspect ratios
    # Perimeter = 2(w+h) = 4, so w+h = 2
    # Focus on aspect ratios near 1.0 which tend to be optimal
    aspect_ratios = np.linspace(0.7, 1.3, 10)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Run force-directed packing with multiple restarts for robustness
        best_for_this_w = None
        best_val_for_this_w = -1
        
        for restart in range(3):
            # Add small random perturbation for different initializations
            np.random.seed(restart * 123 + int(w * 100))
            
            # Run force-directed packing
            x, y, r = _force_directed_packing(N, w, h, n_iters=6000, 
                                              lr_start=0.015, lr_end=0.0003,
                                              contact_tol=0.015, spring_k=4.0, 
                                              overlap_k=15.0, wall_k=8.0, center_k=0.005)
            
            # Refine with scipy
            x_ref, y_ref = _refine_with_scipy(x, y, r, w, h, max_iter=30)
            
            # Validate
            if _validate_packing(x_ref, y_ref, r, w, h):
                val = np.sum(r)
                if val > best_val_for_this_w:
                    best_val_for_this_w = val
                    best_for_this_w = np.column_stack([x_ref, y_ref, r])
        
        if best_for_this_w is not None and best_val_for_this_w > best_val:
            best_val = best_val_for_this_w
            best_result = best_for_this_w
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
