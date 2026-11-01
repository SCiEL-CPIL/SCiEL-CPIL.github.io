"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a Physics-based Force-Directed Relaxation approach enhanced with scipy.optimize:
1. Initialize circles in a hexagonal lattice.
2. Use molecular dynamics (repulsive forces) to resolve overlaps and fit into the rectangle.
3. Iteratively grow radii while maintaining validity via local force relaxation.
4. Optimize aspect ratio by simulating pressure on the walls.
5. Final polish with SLSQP using smooth penalty-based objective.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def relax_forces(x, y, r, w, iterations=200, dt=0.01):
        """
        Force-directed relaxation to resolve overlaps and boundary violations.
        Returns updated x, y.
        """
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        
        # Adaptive step size
        current_dt = dt
        
        for _ in range(iterations):
            if time.time() > deadline - 1.0:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            max_force = 0.0

            # 1. Wall Forces - use quadratic penalty for smoother behavior
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    viol = r[i] - x[i]
                    f = viol * 100.0 + viol * viol * 50.0
                    fx[i] += f
                # Right wall
                if x[i] > w - r[i]:
                    viol = x[i] - (w - r[i])
                    f = viol * 100.0 + viol * viol * 50.0
                    fx[i] -= f
                # Bottom wall
                if y[i] < r[i]:
                    viol = r[i] - y[i]
                    f = viol * 100.0 + viol * viol * 50.0
                    fy[i] += f
                # Top wall
                if y[i] > h - r[i]:
                    viol = y[i] - (h - r[i])
                    f = viol * 100.0 + viol * viol * 50.0
                    fy[i] -= f

            # 2. Pairwise Repulsion - use stronger nonlinear force
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                    r_sum = r[i] + r[j]
                    if d < r_sum * 1.05:  # Start repelling slightly before overlap
                        # Use inverse-square law for stronger long-range repulsion
                        overlap = r_sum - d
                        if d < 1e-9:
                            # Random direction if perfectly overlapping
                            ux = np.random.uniform(-1, 1)
                            uy = np.random.uniform(-1, 1)
                            norm = np.sqrt(ux*ux + uy*uy)
                            if norm > 0:
                                ux /= norm
                                uy /= norm
                            force_mag = 50.0
                        else:
                            ux = dx / d
                            uy = dy / d
                            # Stronger force when closer
                            force_mag = 20.0 * overlap + 5.0 * (overlap / max(d, 1e-6)) ** 2
                        fx[i] += ux * force_mag
                        fy[i] += uy * force_mag
                        fx[j] -= ux * force_mag
                        fy[j] -= uy * force_mag
                        max_force = max(max_force, force_mag)

            # Update positions with velocity damping
            x += current_dt * fx
            y += current_dt * fy
            
            # Clamp to bounds to prevent explosion
            for i in range(N):
                x[i] = np.clip(x[i], 0, w)
                y[i] = np.clip(y[i], 0, h)
                
            # Adaptive timestep: reduce if forces are high
            if max_force > 1.0:
                current_dt = max(0.001, current_dt * 0.7)
            elif max_force < 0.1:
                current_dt = min(0.05, current_dt * 1.1)

        return x, y

    def optimize_radius(x, y, r, w, time_limit=5.0):
        """
        Try to increase radii.
        Strategy: Iteratively try to scale all radii by a small factor,
        then relax positions. If valid, keep.
        """
        h = 2.0 - w
        best_r = r.copy()
        best_val = np.sum(r)
        
        # Initial relaxation
        x, y = relax_forces(x, y, r, w, iterations=150)
        valid, val = check_validity(x, y, r, w)
        if not valid:
             return None

        growth_factors = [1.002, 1.001, 1.0005, 1.0002, 1.0001]
        
        while time.time() < time_limit:
            success = False
            for gf in growth_factors:
                new_r = r * gf
                # Relax with new radii
                x_new, y_new = relax_forces(x, y, new_r, w, iterations=100)
                
                valid, val_new = check_validity(x_new, y_new, new_r, w)
                if valid:
                    r = new_r
                    x = x_new
                    y = y_new
                    best_r = r.copy()
                    best_val = val_new
                    success = True
                    break
            
            if not success:
                break
                    
        return x, y, best_r, w

    def optimize_aspect_ratio(x, y, r, w, time_limit=10.0):
        """
        Optimize width w.
        We can try different w values and see which allows larger radii.
        """
        best_result = (x.copy(), y.copy(), r.copy(), w, np.sum(r))
        
        # Explore a range of aspect ratios
        # The optimal aspect ratio depends on the packing.
        # Let's try perturbing w and re-optimizing radii.
        current_best_val = np.sum(r)
        
        for _ in range(25):
            if time.time() > time_limit:
                break
                
            # Perturb w - try both small and large steps
            if np.random.random() < 0.5:
                dw = np.random.uniform(-0.05, 0.05)
            else:
                dw = np.random.uniform(-0.15, 0.15)
            new_w = np.clip(w + dw, 0.4, 1.6)
            
            # Scale positions to fit new width roughly
            scale_x = new_w / w
            scale_y = (2.0 - new_w) / (2.0 - w)
            
            x_new = x * scale_x
            y_new = y * scale_y
            
            # Clamp
            for i in range(N):
                x_new[i] = np.clip(x_new[i], r[i], new_w - r[i])
                y_new[i] = np.clip(y_new[i], r[i], (2.0 - new_w) - r[i])
            
            # Relax and grow radii
            res = optimize_radius(x_new, y_new, r.copy(), new_w, time_limit=min(2.0, time_limit - (time.time() - (time_limit - 10.0))))
            if res is not None:
                rx, ry, rr, rw = res
                val = np.sum(rr)
                if val > current_best_val + 1e-8:
                    current_best_val = val
                    best_result = (rx, ry, rr, rw, val)
                    w = rw
                    x = rx
                    y = ry
                    r = rr
                    # Restart search around new best
                    continue

        return best_result[0], best_result[1], best_result[2], best_result[3]

    def hex_initialization(w, h, n):
        """Initialize circles in a hexagonal lattice pattern."""
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        
        # Estimate spacing for hexagonal packing
        # Area per circle = w*h/n
        # For hex packing, effective radius ~ sqrt(area/(pi*2/sqrt(3)))
        area_per = w * h / n
        r_est = np.sqrt(area_per / (np.pi * 2.0 / np.sqrt(3))) * 0.85
        r_est = min(r_est, 0.18)
        
        # Determine grid dimensions
        spacing = 2 * r_est * 0.95  # slight overlap allowed initially
        cols = max(1, int(np.ceil(w / spacing)))
        rows = max(1, int(np.ceil(h / (spacing * np.sqrt(3) / 2))))
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                # Hex offset: odd rows shifted by half spacing
                offset = (spacing / 2) if (row % 2 == 1) else 0
                cx = (col + 0.5) * spacing + offset
                cy = (row + 0.5) * spacing * np.sqrt(3) / 2
                
                # Clamp to bounds
                cx = np.clip(cx, r_est, w - r_est)
                cy = np.clip(cy, r_est, h - r_est)
                
                x[idx] = cx
                y[idx] = cy
                r[idx] = r_est
                idx += 1
        
        # If we have more circles than fit in the grid, add remaining
        if idx < n:
            for i in range(idx, n):
                x[i] = np.random.uniform(r_est, w - r_est)
                y[i] = np.random.uniform(r_est, h - r_est)
                r[i] = r_est
        
        return x, y, r

    def run_single_trial(w_init, time_budget):
        """
        Run a full optimization cycle for a given initial aspect ratio.
        """
        h_init = 2.0 - w_init
        
        # 1. Initialization: Hexagonal lattice
        x, y, r = hex_initialization(w_init, h_init, N)
        
        # 2. Initial Relaxation
        x, y = relax_forces(x, y, r, w_init, iterations=300)
        
        # 3. Grow Radii
        res = optimize_radius(x, y, r, w_init, time_limit=time_budget * 0.5)
        if res is None:
             return None
        x, y, r, w = res
        val = np.sum(r)
        
        # 4. Optimize Aspect Ratio
        if time.time() < deadline - 5.0:
             x, y, r, w = optimize_aspect_ratio(x, y, r, w, time_limit=time_budget * 0.4)
             val = np.sum(r)
             
        # 5. Final Polish with SLSQP
        if time.time() < deadline - 3.0:
            x, y, r, w = _slsqp_polish(x, y, r, w, time_limit=min(5.0, deadline - time.time() - 2.0))
            val = np.sum(r)
        
        # 6. Final validity check
        x, y = relax_forces(x, y, r, w, iterations=100)
        valid, val_final = check_validity(x, y, r, w)
        if valid:
            return x, y, r, w, val_final
        else:
            return None

    def _slsqp_polish(x, y, r, w, time_limit=5.0):
        """
        Final polish using SLSQP with smooth penalty-based objective.
        Maximizes sum of radii subject to non-overlap and boundary constraints.
        """
        h = 2.0 - w
        n = len(r)
        
        # Flatten variables: [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}]
        def flatten(xy):
            return np.concatenate([xy[0], xy[1]])
        
        def unflatten(v):
            return v[:n], v[n:]
        
        def objective(v):
            """Minimize negative sum of radii (we maximize sum of radii)."""
            xi, yi = unflatten(v)
            # Penalty for boundary violations
            pen = 0.0
            for i in range(n):
                if xi[i] < r[i]:
                    pen += (r[i] - xi[i]) ** 2 * 1000
                if xi[i] > w - r[i]:
                    pen += (xi[i] - (w - r[i])) ** 2 * 1000
                if yi[i] < r[i]:
                    pen += (r[i] - yi[i]) ** 2 * 1000
                if yi[i] > h - r[i]:
                    pen += (yi[i] - (h - r[i])) ** 2 * 1000
            # Penalty for overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    d2 = dx * dx + dy * dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        pen += (r_sum - d) ** 2 * 500
            return -np.sum(r) + pen
        
        def constraints(v):
            """List of constraint functions (>= 0)."""
            xi, yi = unflatten(v)
            cons = []
            # Boundary constraints
            for i in range(n):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - r[i]})  # x >= r
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: w - r[i] - v[i]})  # x <= w-r
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - r[i]})  # y >= r
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: h - r[i] - v[n+i]})  # y <= h-r
            # Non-overlap constraints
            for i in range(n):
                for j in range(i + 1, n):
                    def overlap_constraint(v, i=i, j=j):
                        xi, yi = unflatten(v)
                        dx = xi[i] - xi[j]
                        dy = yi[i] - yi[j]
                        d2 = dx * dx + dy * dy
                        d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                        return d - (r[i] + r[j])
                    cons.append({'type': 'ineq', 'fun': overlap_constraint})
            return cons
        
        # Initial guess
        v0 = flatten((x.copy(), y.copy()))
        
        # Try SLSQP
        try:
            result = minimize(
                objective, v0,
                method='SLSQP',
                constraints=constraints(v0),
                bounds=[(r[i], w - r[i]) for i in range(n)] + [(r[i], h - r[i]) for i in range(n)],
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if result.success and result.fun < objective(v0):
                xi, yi = unflatten(result.x)
                # Verify validity
                valid, val = check_validity(xi, yi, r, w)
                if valid:
                    return xi, yi, r, w
        except Exception:
            pass
        
        # Fallback: just return original
        return x, y, r, w

    best_value = 0.0
    best_result = None

    # Try multiple initial aspect ratios
    # The optimal aspect ratio is unknown, so we sample a range.
    # For N=21, square (1.0) or slightly rectangular are common.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 1.6, 0.6, 1.3, 0.75]
    
    # Sort by likelihood (square first)
    aspect_ratios.sort(key=lambda x: abs(x - 1.0))
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        time_budget = min(12.0, deadline - time.time() - 5.0)
        if time_budget < 2.0:
            break
            
        result = run_single_trial(w0, time_budget)
        
        if result is not None:
            x, y, r, w, val = result
            if val > best_value:
                best_value = val
                best_result = (x, y, r, w)
                
    # If we have a result, do a final high-precision polish
    if best_result is not None:
        x, y, r, w = best_result
        
        # Final aggressive radius growth with very small steps
        # and very strict relaxation
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            new_r = r * 1.0002
            x_new, y_new = relax_forces(x, y, new_r, w, iterations=200)
            valid, val_new = check_validity(x_new, y_new, new_r, w)
            if valid:
                x, y, r, w = x_new, y_new, new_r, w
                best_value = val_new
            else:
                new_r = r * 1.0001
                x_new, y_new = relax_forces(x, y, new_r, w, iterations=150)
                valid, val_new = check_validity(x_new, y_new, new_r, w)
                if valid:
                    x, y, r, w = x_new, y_new, new_r, w
                    best_value = val_new
                else:
                    break
                
        # Final SLSQP polish if time permits
        if time.time() < deadline - 1.5:
            x, y, r, w = _slsqp_polish(x, y, r, w, time_limit=min(3.0, deadline - time.time() - 1.0))
            best_value = np.sum(r)
                
        # Final validity check and safety shrink
        valid, val = check_validity(x, y, r, w)
        if valid:
             # Shrink slightly to ensure strict inequality for evaluator
             r_final = r * (1 - 1e-9)
             return np.column_stack((x, y, r_final))
        else:
             # Fallback to previous best if polish failed
             return np.column_stack((best_result[0], best_result[1], best_result[2] * (1 - 1e-9)))

    # Fallback: Simple grid in unit square (Perimeter 4 -> 1x1)
    # This should not happen if any trial succeeded
    cols = 5
    rows = 5
    cells = []
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) / cols
        cy = (row + 0.5) / rows
        r0 = min(1.0 / cols, 1.0 / rows) * 0.45
        cells.append((cx, cy, r0))
    return np.array(cells)

if __name__ == "__main__":
    res = construct()
    print(res.shape)
