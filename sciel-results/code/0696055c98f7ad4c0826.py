import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def check_time():
        return time.time() > deadline

    def pack_hexagonal(rows):
        """
        Generate a hexagonal packing configuration.
        Returns: w, h, r, xs, ys
        """
        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        pts = np.array(pts)
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale * 1.0
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
        for i in range(cols):
            for j in range(rows):
                if i * rows + j >= N: break
                pts.append((i * 2.0, j * 2.0))
        
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
        r = scale * 1.0
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def physics_simulate(x0, y0, w0, h0, r0, max_steps=5000, dt=0.002, cooling_rate=0.9995):
        """
        Physics-based simulation using Lennard-Jones potential and soft boundaries.
        Circles repel each other and are confined within the rectangle.
        Returns: w, h, r, xs, ys
        """
        if check_time():
            return w0, h0, r0, x0.copy(), y0.copy()
        
        # Work in normalized space: rectangle is [0, w] x [0, h], circles have radius r
        # Use Lennard-Jones for repulsion: V(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6)
        # sigma = 2*r (contact distance), eps controls strength
        
        sigma = 2.0 * r0
        eps = 1.0
        
        # Soft boundary potential: quadratic penalty when circle exceeds boundary
        boundary_strength = 50.0
        
        x = x0.copy().astype(float)
        y = y0.copy().astype(float)
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        temperature = 0.01
        damping = 0.99
        
        for step in range(max_steps):
            if check_time():
                break
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise Lennard-Jones forces
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    
                    if dist < 1e-10:
                        # Add small random perturbation
                        dx = random.uniform(-1e-6, 1e-6)
                        dy = random.uniform(-1e-6, 1e-6)
                        dist_sq = dx*dx + dy*dy
                        dist = np.sqrt(dist_sq)
                    
                    # LJ force: F = 24*eps*(2*(sigma/dist)^12 - (sigma/dist)^6) / dist
                    sr6 = (sigma/dist)**6
                    sr12 = sr6*sr6
                    force_mag = 24*eps*(2*sr12 - sr6) / dist
                    
                    fx[i] += force_mag * dx
                    fy[i] += force_mag * dy
                    fx[j] -= force_mag * dx
                    fy[j] -= force_mag * dy
            
            # Boundary forces (soft walls)
            for i in range(N):
                # Left wall
                if x[i] < r0:
                    penetration = r0 - x[i]
                    fx[i] += boundary_strength * penetration
                # Right wall
                if x[i] > w0 - r0:
                    penetration = x[i] - (w0 - r0)
                    fx[i] -= boundary_strength * penetration
                # Bottom wall
                if y[i] < r0:
                    penetration = r0 - y[i]
                    fy[i] += boundary_strength * penetration
                # Top wall
                if y[i] > h0 - r0:
                    penetration = y[i] - (h0 - r0)
                    fy[i] -= boundary_strength * penetration
            
            # Velocity Verlet integration
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            
            x += vx * dt
            y += vy * dt
            
            # Cool down
            temperature *= cooling_rate
            
            # Add small random noise to escape local minima
            if step % 100 == 0 and temperature > 1e-4:
                x += np.random.normal(0, temperature*0.01, N)
                y += np.random.normal(0, temperature*0.01, N)
            
            # Clamp to reasonable bounds
            x = np.clip(x, -0.5, w0 + 0.5)
            y = np.clip(y, -0.5, h0 + 0.5)
        
        # After simulation, compute the minimal bounding box and scale
        # Find the tightest rectangle that contains all circles
        xs_min = x.min() - r0
        xs_max = x.max() + r0
        ys_min = y.min() - r0
        ys_max = y.max() + r0
        
        w_actual = xs_max - xs_min
        h_actual = ys_max - ys_min
        
        # Scale to fit perimeter 4: w + h = 2
        if w_actual + h_actual > 0:
            scale = 2.0 / (w_actual + h_actual)
            w_final = scale * w_actual
            h_final = scale * h_actual
            r_final = scale * r0
            
            # Transform coordinates
            x_final = (x - xs_min) * scale
            y_final = (y - ys_min) * scale
        else:
            w_final = w0
            h_final = h0
            r_final = r0
            x_final = x.copy()
            y_final = y.copy()
        
        return w_final, h_final, r_final, x_final, y_final

    def refine_slsqp(w, h, r, xs, ys, max_iter=200):
        """
        Refine the packing using SLSQP with smooth constraints.
        """
        if check_time():
            return w, h, r, xs, ys

        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.05 or h_val <= 0.05 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_arr = v[2:2+N]
            ys_arr = v[2+N:2+2*N]
            
            cons = []
            
            # Aspect ratio sanity
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.15})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.15})
            
            # Boundary constraints
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
                
            # Overlap constraints (smooth approximation)
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: np.sqrt((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 2*v[1]
                    })
            return cons

        bounds = [(0.15, 1.85), (0.005, 1.0)] + [(-0.5, 2.5)] * (2*N)

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False})
            
            if res.success or res.fun < 0:
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
                        if xs_res[i] < r_res - 1e-5 or xs_res[i] > w_res - r_res + 1e-5:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-5 or ys_res[i] > h_res - r_res + 1e-5:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2)
                                if dist < 2*r_res - 1e-5:
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

    # List of initial configurations to try
    configs = []
    
    # 1. Hexagonal 7-7-7
    configs.append(pack_hexagonal([7, 7, 7]))
    
    # 2. Hexagonal 4-5-4-4-4 (21 circles)
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    
    # 3. Hexagonal 4-4-4-4-4-1
    configs.append(pack_hexagonal([4, 4, 4, 4, 4, 1]))
    
    # 4. Grid 5x5 (using 21)
    configs.append(pack_grid(5, 5))
    
    # 5. Grid 3x7 (using 21)
    configs.append(pack_grid(3, 7))
    
    # 6. Grid 4x6 (using 24, take 21)
    configs.append(pack_grid(4, 6))
    
    # 7. Random restarts
    for _ in range(8):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.05
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    # Process each configuration: physics simulation first, then SLSQP refinement
    for cfg in configs:
        if check_time():
            break
            
        w, h, r, xs, ys = cfg
        
        # Step 1: Physics simulation to find a dense local optimum
        w_phys, h_phys, r_phys, xs_phys, ys_phys = physics_simulate(xs, ys, w, h, r)
        
        if r_phys > best_r:
            best_r = r_phys
            best_config = (w_phys, h_phys, r_phys, xs_phys, ys_phys)
        
        # Step 2: SLSQP refinement from physics result
        if not check_time():
            w_slsqp, h_slsqp, r_slsqp, xs_slsqp, ys_slsqp = refine_slsqp(w_phys, h_phys, r_phys, xs_phys, ys_phys)
            
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_config = (w_slsqp, h_slsqp, r_slsqp, xs_slsqp, ys_slsqp)
        
        # Step 3: Also try SLSQP directly from original config
        if not check_time():
            w_slsqp2, h_slsqp2, r_slsqp2, xs_slsqp2, ys_slsqp2 = refine_slsqp(w, h, r, xs, ys)
            
            if r_slsqp2 > best_r:
                best_r = r_slsqp2
                best_config = (w_slsqp2, h_slsqp2, r_slsqp2, xs_slsqp2, ys_slsqp2)

    # Final high-precision polish on the best configuration
    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_final, h_final, r_final, xs_final, ys_final = refine_slsqp(w, h, r, xs, ys, max_iter=500)
        
        if r_final > best_r:
            best_r = r_final
            best_config = (w_final, h_final, r_final, xs_final, ys_final)

    if best_config is None:
        # Fallback
        w, h, r, xs, ys = configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    
    # Final safety adjustment
    r_final = r * (1 - 1e-6)
    
    return np.column_stack([xs, ys, np.full(N, r_final)])
