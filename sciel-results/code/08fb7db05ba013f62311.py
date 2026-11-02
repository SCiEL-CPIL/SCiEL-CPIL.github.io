import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140.0

def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if configuration is strictly valid."""
    if np.any(r <= 0):
        return False
    if w <= 0 or h <= 0:
        return False
    if 2*(w+h) > 4.0 + tol:
        return False
        
    # Boundary check
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
        
    # Overlap check
    n = len(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    
    # Check all pairs i < j
    diff = dist_sq - r_sum**2
    np.fill_diagonal(diff, 1.0)
    
    if np.min(diff) < -tol:
        return False
        
    return True

def _force_relax(x, y, r, w, h, max_iter=2000, dt=0.01):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    Moves circles apart and into the rectangle.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Boundary forces
        # Left
        viol = r - x
        mask = viol > 0
        fx[mask] += viol[mask]
        # Right
        viol = x + r - w
        mask = viol > 0
        fx[mask] -= viol[mask]
        # Bottom
        viol = r - y
        mask = viol > 0
        fy[mask] += viol[mask]
        # Top
        viol = y + r - h
        mask = viol > 0
        fy[mask] -= viol[mask]
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-8:
                        dx = 0.01
                        dy = 0.01
                        dist = np.sqrt(dx*dx + dy*dy)
                    # Force magnitude proportional to overlap
                    f_mag = (min_dist - dist)
                    fx[i] += (dx/dist) * f_mag
                    fy[i] += (dy/dist) * f_mag
                    fx[j] -= (dx/dist) * f_mag
                    fy[j] -= (dy/dist) * f_mag
        
        # Update positions
        x += fx * dt
        y += fy * dt
        
        # Clamp to boundaries (hard constraint for stability)
        x = np.clip(x, r, w-r)
        y = np.clip(y, r, h-r)
        
        # Check convergence
        if np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break
            
    return x, y, r

def _initialize_hexagonal(w, h, n_circles=21, seed=0):
    """
    Initialize circles in a hexagonal packing pattern.
    This is a much better starting point than a grid for dense packings.
    """
    rng = np.random.default_rng(seed)
    
    # Estimate optimal radius for hexagonal packing
    # Area per circle in hex packing is approx 2.309 * r^2
    # Total area = w*h
    # n * 2.309 * r^2 <= w*h => r <= sqrt(w*h / (n * 2.309))
    r_est = np.sqrt(w * h / (n_circles * 2.309))
    
    # Adjust for boundary effects and safety
    r_est *= 0.95
    
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    r = np.zeros(n_circles)
    
    # Hexagonal lattice parameters
    # dx = 2*r, dy = sqrt(3)*r
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    idx = 0
    # Generate hexagonal grid points
    row = 0
    while idx < n_circles:
        # Offset every other row
        offset = (row % 2) * (dx / 2)
        
        col = 0
        while idx < n_circles:
            cx = offset + col * dx + r_est
            cy = row * dy + r_est
            
            # Check if inside rectangle (with margin)
            if cx - r_est >= 0 and cx + r_est <= w and cy - r_est >= 0 and cy + r_est <= h:
                # Add small random perturbation to break symmetry
                cx += rng.uniform(-0.05*r_est, 0.05*r_est)
                cy += rng.uniform(-0.05*r_est, 0.05*r_est)
                
                x[idx] = np.clip(cx, r_est, w - r_est)
                y[idx] = np.clip(cy, r_est, h - r_est)
                r[idx] = r_est
                idx += 1
            else:
                col += 1
            
            # Stop if we've gone too far in x
            if cx > w + r_est:
                break
        
        row += 1
        # Stop if we've gone too far in y
        if row * dy + r_est > h + r_est:
            break
    
    # If we didn't fill all N, add remaining at random positions
    while idx < n_circles:
        x[idx] = rng.uniform(0.05*w, 0.95*w)
        y[idx] = rng.uniform(0.05*h, 0.95*h)
        r[idx] = 0.5 * r_est
        idx += 1
    
    # Ensure initial feasibility by shrinking radii if needed
    scale = 1.0
    for _ in range(50):
        if _is_valid(x, y, r * scale, w, h, tol=1e-6):
            break
        scale *= 0.95
    r *= scale
    
    return np.concatenate([x, y, r])

def _granular_polish(v, w, h, max_iter=1000):
    """
    Vectorized force-directed polish to refine positions and radii.
    """
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:3*N].copy()
    
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-7
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Boundary forces
        fx += np.maximum(r - x, 0.0)
        fx -= np.maximum(x + r - w, 0.0)
        fy += np.maximum(r - y, 0.0)
        fy -= np.maximum(y + r - h, 0.0)
        
        # Overlap forces (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        
        mask = d > 1e-8
        d_safe = np.where(mask, d, 1.0)
        
        overlap = np.maximum(r_sum - d, 0.0)
        
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        overlap_iu = overlap[iu]
        d_iu = d_safe[iu]
        dx_iu = dx[iu]
        dy_iu = dy[iu]
        
        # Accumulate forces
        for i, j in zip(iu[0], iu[1]):
            if overlap[i, j] > 0:
                f_mag = overlap[i, j] / d_safe[i, j]
                fx[i] += dx[i, j] * f_mag
                fy[i] += dy[i, j] * f_mag
                fx[j] -= dx[i, j] * f_mag
                fy[j] -= dy[i, j] * f_mag
        
        # Update positions
        x += fx * dt
        y += fy * dt
        
        # Clamp to boundaries
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        # Check convergence
        if np.max(np.abs(fx)) < convergence_tol and np.max(np.abs(fy)) < convergence_tol:
            break
        
        # Adaptive step
        if it % 50 == 0:
            dt = max(min_dt, dt * 0.9)
            
    return np.concatenate([x, y, r])

def _epigraph_refine(v, w, h, max_iter=50):
    """
    Try to increase the sum of radii by solving a convex sub-problem.
    This is a heuristic approximation of epigraph refinement.
    """
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:3*N].copy()
    
    best_val = r.sum()
    best_v = v.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        # Try to expand each circle slightly
        improved = False
        for i in range(N):
            old_r = r[i]
            r[i] = old_r * 1.005  # 0.5% expansion
            
            # Relax positions to accommodate
            v_new = np.concatenate([x, y, r])
            v_new = _granular_polish(v_new, w, h, max_iter=100)
            
            x_new, y_new, r_new = v_new[:N], v_new[N:2*N], v_new[2*N:3*N]
            
            if _is_valid(x_new, y_new, r_new, w, h, tol=1e-9):
                val = r_new.sum()
                if val > best_val:
                    best_val = val
                    best_v = v_new
                    x, y, r = x_new, y_new, r_new
                    improved = True
            else:
                r[i] = old_r  # Revert if invalid
        
        if not improved:
            break
            
    return best_v

def construct():
    start_time = time.time()
    best_value = 0.0
    best_circles = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1, 0.6, 1.4, 0.85, 1.15, 0.75, 1.25]
    
    rng = np.random.default_rng(42)
    rng.shuffle(aspect_ratios)
    
    # Also try some random aspect ratios
    for _ in range(5):
        aspect_ratios.append(rng.uniform(0.5, 1.5))
        
    for w in aspect_ratios:
        if time.time() > DEADLINE - 20:
            break
            
        h = 2.0 - w
        if h < 0.3 or w < 0.3:
            continue
            
        # Multiple random restarts for this aspect ratio
        for restart in range(2):
            if time.time() > DEADLINE - 15:
                break
                
            # Initialize with hexagonal pattern
            v0 = _initialize_hexagonal(w, h, seed=restart * 100 + int(w * 100))
            
            # Optimize with SLSQP
            x, y, r = v0[:N], v0[N:2*N], v0[2*N:3*N]
            x, y, r = _force_relax(x, y, r, w, h, max_iter=500, dt=0.01)
            v0 = np.concatenate([x, y, r])
            
            # SLSQP objective and constraints
            def objective(v):
                return -np.sum(v[2*N:3*N])
            
            def constraints(v):
                x, y, r = v[:N], v[N:2*N], v[2*N:3*N]
                cons = []
                
                # Boundary constraints
                for i in range(N):
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: (w - v[i]) - v[2*N+i]})
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: (h - v[N+i]) - v[2*N+i]})
                
                # Overlap constraints
                for i in range(N):
                    for j in range(i+1, N):
                        cons.append({
                            'type': 'ineq',
                            'fun': lambda v, i=i, j=j: 
                                (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                        })
                
                return cons
            
            bounds = []
            for i in range(N):
                bounds.append((0.001, w - 0.001))  # x_i
            for i in range(N):
                bounds.append((0.001, h - 0.001))  # y_i
            for i in range(N):
                bounds.append((0.001, min(w, h)/2))  # r_i
            
            try:
                result = minimize(
                    objective,
                    v0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                
                if result.success and time.time() < (DEADLINE - 10):
                    v_opt = result.x
                else:
                    # Even if not fully converged, check if result is valid
                    x, y, r = result.x[:N], result.x[N:2*N], result.x[2*N:3*N]
                    if _is_valid(x, y, r, w, h, tol=1e-8):
                        v_opt = result.x
                    else:
                        continue
                        
            except Exception:
                continue
            
            # Polish with granular search
            if time.time() < DEADLINE - 10:
                v_opt = _granular_polish(v_opt, w, h, max_iter=500)
                
            # Epigraph refine
            if time.time() < DEADLINE - 5:
                v_opt = _epigraph_refine(v_opt, w, h, max_iter=30)
                
            x, y, r = v_opt[:N], v_opt[N:2*N], v_opt[2*N:3*N]
            val = r.sum()
            
            if val > best_value:
                best_value = val
                best_circles = np.column_stack((x, y, r))
                
    if best_circles is None:
        # Fallback: Simple grid in square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) / side
                y = (i + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    return best_circles
