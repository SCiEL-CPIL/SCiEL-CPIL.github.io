"""Circle packing in perimeter-4 rectangle using constrained NLP with aspect scanning."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_value = 0.0
    best_result = None
    
    # Try different aspect ratios and random initializations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5]
    
    for attempt in range(25):
        if time.time() > deadline - 8:
            break
        
        rng = np.random.default_rng(attempt * 7919 + 31337)
        
        # Cycle through aspect ratios
        ar_idx = attempt % len(aspect_ratios)
        aspect = aspect_ratios[ar_idx]
        
        # Perimeter = 2(w+h) = 4, so w+h = 2
        # w/h = aspect => w = aspect*h => h(aspect+1) = 2
        h_init = 2.0 / (aspect + 1.0)
        w_init = aspect * h_init
        
        # Generate initial packing
        centers, radii = _initialize_packing(N, w_init, h_init, rng)
        
        # Force-directed pre-relaxation
        centers, radii = _force_directed_relax(centers, radii, w_init, h_init, 
                                                iterations=200, rng=rng)
        
        # Constrained NLP optimization
        result = _constrained_optimize(centers, radii, w_init, h_init,
                                       time_budget=deadline - (time.time() - start_time))
        
        if result is not None:
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_result = result
        
        if time.time() > deadline - 8:
            break
    
    # Final refinement pass on best result
    if best_result is not None:
        if time.time() < deadline - 5:
            refined = _local_refine(best_result, time_budget=deadline - (time.time() - start_time))
            if refined is not None:
                value = np.sum(refined[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = refined
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _initialize_packing(n, w, h, rng):
    """Initialize circle positions and radii in a grid pattern within w x h rectangle."""
    side = int(np.ceil(np.sqrt(n)))
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.08)
    
    for i in range(n):
        row, col = divmod(i, side)
        centers[i, 0] = (col + 0.5) * w / side + rng.uniform(-0.01, 0.01)
        centers[i, 1] = (row + 0.5) * h / side + rng.uniform(-0.01, 0.01)
    
    # Clip to stay in bounds
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
    
    return centers, radii


def _force_directed_relax(centers, radii, w, h, iterations=200, rng=None):
    """Force-directed relaxation to resolve overlaps and push circles apart."""
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    for it in range(iterations):
        forces = np.zeros((n, 2))
        
        # Repulsive forces between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    # Strong repulsion
                    fx = (dx / dist) * overlap * 5.0
                    fy = (dy / dist) * overlap * 5.0
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
        
        # Attractive forces toward center (to keep circles in bounds)
        for i in range(n):
            cx, cy = centers[i]
            r = radii[i]
            # Push away from walls
            if cx - r < 0:
                forces[i, 0] += (r - cx) * 3.0
            if cx + r > w:
                forces[i, 0] -= (cx + r - w) * 3.0
            if cy - r < 0:
                forces[i, 1] += (r - cy) * 3.0
            if cy + r > h:
                forces[i, 1] -= (cy + r - h) * 3.0
        
        # Update positions
        learning_rate = 0.1 * (1.0 - it / iterations) + 0.01
        for i in range(n):
            centers[i, 0] += forces[i, 0] * learning_rate
            centers[i, 1] += forces[i, 1] * learning_rate
            # Keep in bounds
            centers[i, 0] = np.clip(centers[i, 0], radii[i] * 0.5, w - radii[i] * 0.5)
            centers[i, 1] = np.clip(centers[i, 1], radii[i] * 0.5, h - radii[i] * 0.5)
        
        # Gradually increase radii
        if it < iterations // 2:
            for i in range(n):
                radii[i] = min(radii[i] + 0.0002, min(w, h) * 0.25)
    
    # Final clip
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
    
    return centers, radii


def _constrained_optimize(centers, radii, w_init, h_init, time_budget):
    """Constrained NLP using SLSQP with explicit inequality constraints."""
    n = len(centers)
    start_time = time.time()
    
    # Parameter vector: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    params_init = np.concatenate([centers.flatten(), radii])
    
    def objective(params):
        """Minimize negative sum of radii."""
        radii_opt = params[2::3]
        return -np.sum(radii_opt)
    
    # Inequality constraints (must be >= 0)
    constraints = []
    
    # Perimeter constraint: w + h <= 2 (fixed, so this is always satisfied)
    # But we need circles to fit in the rectangle
    
    # Wall constraints: each circle must be inside the rectangle
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, idx=i: p[idx*3] - p[idx*3 + 2]
        })
        # w - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, idx=i: w_init - p[idx*3] - p[idx*3 + 2]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, idx=i: p[idx*3 + 1] - p[idx*3 + 2]
        })
        # h - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, idx=i: h_init - p[idx*3 + 1] - p[idx*3 + 2]
        })
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, idx1=i, idx2=j: (
                    (p[idx1*3] - p[idx2*3])**2 + 
                    (p[idx1*3+1] - p[idx2*3+1])**2 - 
                    (p[idx1*3+2] + p[idx2*3+2])**2
                )
            })
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, w_init))  # x
        bounds.append((0.0, h_init))  # y
        bounds.append((0.01, min(w_init, h_init) * 0.3))  # r
    
    # Optimize with SLSQP
    maxiter = int(min(300, time_budget * 30))
    
    try:
        res = minimize(objective, params_init, method='SLSQP', bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': maxiter, 'ftol': 1e-10})
        
        if res.fun < 1e3:  # Check if solution is reasonable
            params_opt = res.x
            centers_opt = params_opt[0::3].copy()
            radii_opt = params_opt[2::3].copy()
            
            # Verify validity
            valid = _verify_packing(centers_opt, radii_opt, w_init, h_init)
            
            if valid:
                result = np.column_stack([centers_opt, radii_opt])
                return result
    except Exception:
        pass
    
    return None


def _verify_packing(centers, radii, w, h, tol=1e-6):
    """Verify that the packing is valid."""
    n = len(centers)
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 0:
            return False
        if x - r < -tol or x + r > w + tol or y - r < -tol or y + r > h + tol:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    
    return True


def _local_refine(result, time_budget):
    """Local refinement of a valid packing using small perturbations."""
    centers = result[:, :2].copy()
    radii = result[:, 2].copy()
    n = len(centers)
    
    w = np.max(centers[:, 0] + radii)
    h = np.max(centers[:, 1] + radii)
    
    # Try small random perturbations and accept if valid and improves
    start_time = time.time()
    current_value = np.sum(radii)
    
    for step in range(200):
        if time.time() > start_time + time_budget:
            break
        
        # Random perturbation
        idx = np.random.randint(n)
        delta = np.random.normal(0, 0.005, 2)
        new_center = centers[idx].copy() + delta
        new_radius = radii[idx] * np.random.uniform(0.98, 1.02)
        
        # Try the perturbation
        old_center = centers[idx].copy()
        old_radius = radii[idx]
        centers[idx] = new_center
        radii[idx] = new_radius
        
        if _verify_packing(centers, radii, w, h):
            new_value = np.sum(radii)
            if new_value > current_value:
                current_value = new_value
            else:
                # Revert
                centers[idx] = old_center
                radii[idx] = old_radius
        else:
            # Revert
            centers[idx] = old_center
            radii[idx] = old_radius
    
    result_final = np.column_stack([centers, radii])
    return result_final
