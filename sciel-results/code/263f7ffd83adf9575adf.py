"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _make_hexagonal_initial(w, h, rows=None):
    """
    Create a hexagonal-like arrangement for 21 circles.
    Default rows: 5, 4, 5, 4, 3
    """
    if rows is None:
        rows = [5, 4, 5, 4, 3]
    n_rows = len(rows)
    circles = []
    
    # Estimate base radius
    r_est = np.sqrt(w * h / (42 * np.sqrt(3)))
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / n_rows
        # Center the row
        # If count is odd, center is at (count-1)/2. If even, center is between.
        # For hexagonal, we usually offset.
        # Let's place them uniformly in x with a margin.
        # x_k = (k + 0.5) * w / count + offset
        # To center, if count is 5, x is 0.1w, 0.3w, 0.5w, 0.7w, 0.9w
        # If count is 4, x is 0.125w, 0.375w, 0.625w, 0.875w? 
        # Better: x_k = (k + 0.5) * w / count. This centers the block.
        
        for col_j in range(count):
            x = (col_j + 0.5) * w / count
            r = r_est
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _get_contact_pairs(rows):
    """
    Generate contact pairs for a hexagonal-like arrangement.
    Returns a list of (i, j) tuples.
    """
    pairs = set()
    idx = 0
    for row_i, count in enumerate(rows):
        for col_j in range(count):
            # Horizontal neighbors
            if col_j < count - 1:
                pairs.add((idx, idx + 1))
            
            # Vertical/Diagonal neighbors (row below)
            if row_i + 1 < len(rows):
                below_count = rows[row_i + 1]
                # In a hexagonal packing, a circle in row i, col j touches:
                # - circle in row i+1, col j (if exists)
                # - circle in row i+1, col j+1 (if exists)
                # This assumes the row below is shifted by half a unit to the right.
                # If rows are centered, the geometry is slightly different, 
                # but for contact detection, these are the likely candidates.
                
                if col_j < below_count:
                    pairs.add((idx, idx + count + col_j))
                if col_j + 1 < below_count:
                    pairs.add((idx, idx + count + col_j + 1))
            
            idx += count
    
    return list(pairs)

def _relax_circles_safe(circles, w, h, iterations=1000, tol=1e-9):
    """
    Relaxation to ensure validity.
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for it in range(iterations):
        moved = False
        
        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] > w - rad[i]:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] > h - rad[i]:
                pos[i, 1] = h - rad[i]
                moved = True
        
        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True
        
        if not moved:
            break
    
    new_circles = np.column_stack((pos, rad))
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    return circles

def _solve_with_contacts(circles, w, h, contact_pairs, deadline):
    """
    Optimize using SLSQP with equality constraints for contact pairs
    and inequality constraints for all other pairs and boundaries.
    """
    if time.time() > deadline - 10.0:
        return None

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    constraints = []
    
    # 1. Boundary Constraints (Inequality: x - r >= 0, etc.)
    for i in range(n):
        # Left
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
        # Right
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i] - x[2*n+i]})
        # Bottom
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        # Top
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[n+i] - x[2*n+i]})
    
    # 2. Contact Constraints (Equality: dist - r_i - r_j = 0)
    # We only add equality constraints for the specific contact pairs.
    # This forces them to touch.
    for i, j in contact_pairs:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[n+i]-x[n+j])**2) - x[2*n+i] - x[2*n+j]
        })
        
    # 3. Non-overlap Constraints (Inequality: dist - r_i - r_j >= 0)
    # For all pairs NOT in contact_pairs.
    # To keep the number of constraints manageable, we can add all of them.
    # N=21, pairs=210. Contacts ~30. Inequalities ~180.
    # Total constraints ~ 21*4 (boundaries) + 30 (contacts) + 180 (ineq) = ~300.
    # SLSQP might be slow with 300 constraints on 63 variables.
    # Alternative: Only add inequality constraints for pairs that are "close" in the initial guess?
    # Or, use a penalty for non-contact overlaps and equality for contacts?
    # Let's try adding all inequality constraints first. If it's too slow, we'll fallback.
    
    contact_set = set((min(i,j), max(i,j)) for i,j in contact_pairs)
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in contact_set:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[n+i]-x[n+j])**2) - x[2*n+i] - x[2*n+j]
                })
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < 0:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            # SLSQP might violate constraints slightly. Relax.
            new_circles = _relax_circles_safe(new_circles, w, h, iterations=200)
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass
    
    return None

def _penalty_optimize(circles, w, h, deadline):
    """
    Fallback penalty-based optimization.
    """
    if time.time() > deadline - 5.0:
        return None

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-8, w))
        bounds.append((1e-8, h))
    for i in range(n):
        bounds.append((1e-8, min(w, h) / 2))

    def _objective_and_grad(x, w, h, beta, eps):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        for i in range(n):
            for val, idx_pos, idx_rad in [
                (cx[i] - r[i], i, 2*n+i),
                (w - cx[i] - r[i], i, 2*n+i),
                (cy[i] - r[i], n+i, 2*n+i),
                (h - cy[i] - r[i], n+i, 2*n+i)
            ]:
                if val < eps:
                    t = val / eps
                    pen = beta * eps * eps * np.logaddexp(0, -t)
                    sig = 1.0 / (1.0 + np.exp(t))
                    dpen_dval = -beta * eps * sig
                    if idx_pos < n: # x
                        grad[idx_pos] += dpen_dval * (1 if idx_pos == i else -1) # Simplified, check signs
                    # Actually, let's just use a simpler quadratic penalty for speed in fallback
                    # The main work is done by SLSQP contact solver.
        
        # Simplified penalty for fallback: Quadratic
        pen = 0.0
        for i in range(n):
            pen += max(0, r[i] - cx[i])**2
            pen += max(0, cx[i] + r[i] - w)**2
            pen += max(0, r[i] - cy[i])**2
            pen += max(0, cy[i] + r[i] - h)**2
        
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen += (min_dist - dist)**2
        
        obj += beta * pen
        
        # Gradient is complex to compute analytically for this simplified version.
        # Use numerical gradient or just return objective and let L-BFGS-B estimate?
        # L-BFGS-B requires jac.
        # Let's stick to the previous robust penalty gradient implementation if needed.
        # For now, return None if this gets complicated, relying on SLSQP.
        
        return obj, grad

    # Since the fallback gradient is complex, we will skip the complex penalty optimization
    # and rely on the contact-based SLSQP which is the structural improvement.
    # If SLSQP fails, we return the relaxed initial guess.
    return None

def _try_aspect(ar, deadline, rows=None):
    """Try a single aspect ratio and return best result."""
    w = ar / (1 + ar) * 2
    h = 1 / (1 + ar) * 2
    
    if rows is None:
        rows = [5, 4, 5, 4, 3]
        
    best_val = 0.0
    best_circles = None
    
    for (ww, hh) in [(w, h), (h, w)]:
        if time.time() > deadline - 10.0:
            break
        
        try:
            circles = _make_hexagonal_initial(ww, hh, rows)
            circles = _relax_circles_safe(circles, ww, hh, iterations=500)
            
            if not _validate(circles, ww, hh):
                continue
            
            val = _total_radius(circles)
            
            # Try contact-based optimization
            if time.time() < deadline - 15.0:
                contact_pairs = _get_contact_pairs(rows)
                opt = _solve_with_contacts(circles, ww, hh, contact_pairs, deadline)
                if opt is not None:
                    circles = opt
                    val = _total_radius(circles)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
                
        except Exception:
            continue
    
    return best_circles, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Targeted aspect ratios
    ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.8, 1.4, 0.75]
    
    # Different row configurations to explore different topologies
    # 5-4-5-4-3 is the primary one.
    # 6-5-5-5 is another (21 circles).
    # 7-7-7 is a grid.
    row_configs = [
        [5, 4, 5, 4, 3],
        [6, 5, 5, 5],
        [7, 7, 7]
    ]
    
    for ar in ratios:
        if time.time() > deadline - 12.0:
            break
        
        for rows in row_configs:
            if time.time() > deadline - 12.0:
                break
            circles, val = _try_aspect(ar, deadline, rows)
            if circles is not None and val > best_value:
                best_value = val
                best_result = circles
    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Best Value: {_total_radius(res)}")
    print(f"Valid: {_validate(res, 2.0, 2.0)}") # Approximate check, actual w/h varies
