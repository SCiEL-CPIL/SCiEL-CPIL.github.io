import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit

    def check_time():
        return time.time() > deadline

    def pack_hexagonal(rows):
        """
        Generate a hexagonal packing configuration.
        Returns: w, h, r, xs, ys
        """
        if sum(rows) != N:
            return None

        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        pts = np.array(pts)
        if len(pts) != N:
            return None

        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def pack_grid(rows, cols):
        """
        Generate a grid packing configuration.
        """
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break
        
        pts = np.array(pts[:N])
        
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def physics_simulate(w, h, r, xs, ys, max_iter=500):
        """
        Physics-based refinement using Lennard-Jones-like repulsion and soft boundaries.
        Returns improved (w, h, r, xs, ys) or original if no improvement.
        """
        if check_time():
            return w, h, r, xs, ys

        # Work in scaled coordinates where the bounding box is [0, w] x [0, h]
        # We will simulate forces and then rescale.
        # Actually, it's easier to simulate in the target space directly.
        
        x = xs.copy()
        y = ys.copy()
        
        # Initial velocity is zero
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        # Parameters
        # We want to maximize r. 
        # The constraint is that circles don't overlap and stay in box.
        # We can simulate a system where circles repel each other and are attracted to the center?
        # Or better: simulate a system where we try to increase r by pushing circles apart.
        # A common trick: simulate particles with radius r. If they overlap, push apart.
        # If they hit walls, push in.
        # Then, if the system stabilizes with no overlaps, we can try to increase r slightly.
        
        # Let's use a simple iterative relaxation.
        # For each pair, if dist < 2r, push apart.
        # For each wall, if x < r or x > w-r, push in.
        
        # To allow r to grow, we can add a small outward force from the center?
        # No, the perimeter is fixed (w+h=2). We can vary w and h?
        # The current program fixes w and h in the simulation step?
        # The current program optimizes w, r, xs, ys.
        # Let's stick to optimizing positions for a fixed w, h, r first, then scale?
        # No, let's simulate with variable r? That's hard.
        
        # Alternative: Simulate with fixed r (current r), fix w, h.
        # Relax positions to remove overlaps.
        # Then, try to increase r slightly and relax again.
        
        # Let's try a simple gradient descent on a soft penalty function.
        # Penalty = sum of max(0, 2r - dist_ij)^2 + sum of boundary violations^2
        # Minimize this penalty by moving circles.
        
        # But we want to MAXIMIZE r.
        # So:
        # 1. Start with current r.
        # 2. Relax positions to minimize overlap penalty.
        # 3. Increase r by a small delta.
        # 4. Repeat until no more improvement or time out.
        
        # This is essentially what SLSQP does, but maybe physics is faster for initial relaxation?
        # Let's implement a simple relaxation step.
        
        current_r = r
        current_w = w
        current_h = h
        
        # Relaxation function
        def relax(x, y, r_val, w_val, h_val, steps=50):
            for _ in range(steps):
                if check_time():
                    break
                fx = np.zeros(N)
                fy = np.zeros(N)
                
                # Pairwise repulsion
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = 2.0 * r_val
                        if dist < min_dist:
                            if dist > 1e-6:
                                # Force proportional to penetration
                                overlap = min_dist - dist
                                force = 10.0 * overlap
                                fx[i] += (dx/dist) * force
                                fy[i] += (dy/dist) * force
                                fx[j] -= (dx/dist) * force
                                fy[j] -= (dy/dist) * force
                            else:
                                # Random kick
                                fx[i] += random.uniform(-1, 1)
                                fy[i] += random.uniform(-1, 1)
                                fx[j] -= random.uniform(-1, 1)
                                fy[j] -= random.uniform(-1, 1)
                
                # Boundary repulsion
                for i in range(N):
                    # Left
                    if x[i] < r_val:
                        fx[i] += 10.0 * (r_val - x[i])
                    # Right
                    if x[i] > w_val - r_val:
                        fx[i] -= 10.0 * (x[i] - (w_val - r_val))
                    # Bottom
                    if y[i] < r_val:
                        fy[i] += 10.0 * (r_val - y[i])
                    # Top
                    if y[i] > h_val - r_val:
                        fy[i] -= 10.0 * (y[i] - (h_val - r_val))
                
                # Update positions (simple Euler)
                x += 0.01 * fx
                y += 0.01 * fy
                
                # Clip to valid range to prevent divergence
                x = np.clip(x, r_val, w_val - r_val)
                y = np.clip(y, r_val, h_val - r_val)
                
            return x, y

        # Try to increase r
        # We need to check if the current configuration is valid.
        def is_valid(x, y, r_val, w_val, h_val):
            for i in range(N):
                if x[i] < r_val - 1e-7 or x[i] > w_val - r_val + 1e-7:
                    return False
                if y[i] < r_val - 1e-7 or y[i] > h_val - r_val + 1e-7:
                    return False
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    if dist_sq < 4*(r_val**2) - 1e-7:
                        return False
            return True

        if not is_valid(x, y, current_r, current_w, current_h):
            # Relax first
            x, y = relax(x, y, current_r, current_w, current_h, steps=100)
            if not is_valid(x, y, current_r, current_w, current_h):
                return w, h, r, xs, ys # Failed to relax

        # Now try to increase r
        # We can vary w and h? No, let's keep w, h fixed for now and see if r can increase.
        # Actually, increasing r might require changing aspect ratio w/h.
        # But let's just try to increase r with fixed w, h first.
        
        # Max possible r is limited by w/2 and h/2.
        max_r_possible = min(current_w, current_h) / 2.0
        # Also limited by packing density.
        
        r_try = current_r
        delta_r = 0.001
        improved = False
        
        while r_try < max_r_possible - 1e-4 and not check_time():
            # Try to relax with new r
            x_new, y_new = relax(x, y, r_try + delta_r, current_w, current_h, steps=20)
            
            if is_valid(x_new, y_new, r_try + delta_r, current_w, current_h):
                r_try += delta_r
                x, y = x_new, y_new
                improved = True
            else:
                break
        
        if improved:
            return current_w, current_h, r_try, x, y
        else:
            # Return relaxed position even if r didn't increase, might help SLSQP
            return current_w, current_h, current_r, x, y

    def refine(w, h, r, xs, ys, is_polish=False):
        """
        Refine the packing using SLSQP.
        Variables: w, r, xs, ys. h = 2 - w.
        """
        if check_time():
            return w, h, r, xs, ys

        # Pre-relax with physics to get a better starting point
        w, h, r, xs, ys = physics_simulate(w, h, r, xs, ys)
        
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            cons = []
            
            # Aspect ratio sanity
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            
            # Boundary constraints
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
                
            # Overlap constraints (squared distance)
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: ((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 4*(v[1]**2)
                    })
            return cons

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        try:
            if is_polish:
                options = {'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            else:
                options = {'maxiter': 150, 'ftol': 1e-9, 'disp': False}

            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options=options)
            
            if res.success or res.fun < -r:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                
                # Validate
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                                if dist_sq < 4*(r_res**2) - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
            
        return w, h, r, xs, ys

    best_r = 0
    best_config = None

    configs = []
    
    # Hexagonal variations
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    
    # Grid variations
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))
    
    # Random restarts
    for _ in range(8):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    # First pass: Coarse optimization
    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg
        w_new, h_new, r_new, xs_new, ys_new = refine(w, h, r, xs, ys, is_polish=False)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Second pass: Polish the best configuration
    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is None:
        # Fallback
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    
    # Final safety adjustment
    r_final = r * (1 - 1e-5)
    
    return np.column_stack([xs, ys, np.full(N, r_final)])
