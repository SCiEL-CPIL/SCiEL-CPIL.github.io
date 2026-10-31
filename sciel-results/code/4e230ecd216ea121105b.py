import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_score = -np.inf
    best_result = None

    def get_score(x):
        """
        x: array of shape (3*N + 1)
        x[0:N]: xs
        x[N:2N]: ys
        x[2N:3N]: rs
        x[3N]: W
        H = 2 - W
        Returns sum of radii if valid, else -inf
        """
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = 2.0 - W
        
        if W <= 0 or H <= 0:
            return -np.inf
            
        # Check bounds
        for i in range(N):
            if xs[i] < rs[i] - 1e-10 or xs[i] > W - rs[i] + 1e-10 or ys[i] < rs[i] - 1e-10 or ys[i] > H - rs[i] + 1e-10:
                return -np.inf
                
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum_sq = (rs[i]+rs[j])**2
                if dist_sq < rad_sum_sq - 1e-9:
                    return -np.inf
                    
        return np.sum(rs)

    def force_directed_simulate(W, H, x0, n_steps=2000, dt=0.001, damping=0.95):
        """
        Force-directed simulation treating circles as incompressible particles.
        Circles repel each other and are confined by walls.
        Radii grow under pressure until equilibrium.
        """
        xs = x0[:N].copy()
        ys = x0[N:2*N].copy()
        rs = x0[2*N:3*N].copy()
        
        # Track growth pressure
        growth_rate = 0.0005
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Circle-circle repulsion (soft sphere model)
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    r_sum = rs[i] + rs[j]
                    
                    if d < 1e-12:
                        # Random nudge for coincident points
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        d = np.sqrt(dx*dx + dy*dy)
                    
                    if d < r_sum:
                        # Overlap: strong repulsive force
                        overlap = r_sum - d
                        # Force magnitude proportional to overlap
                        force_mag = 50.0 * overlap
                        fx[i] += force_mag * dx / d
                        fy[i] += force_mag * dy / d
                        fx[j] -= force_mag * dx / d
                        fy[j] -= force_mag * dy / d
                    
                    elif d < r_sum * 1.5:
                        # Near overlap: moderate repulsion to prevent sticking
                        force_mag = 5.0 * (r_sum * 1.5 - d)
                        fx[i] += force_mag * dx / d
                        fy[i] += force_mag * dy / d
                        fx[j] -= force_mag * dx / d
                        fy[j] -= force_mag * dy / d
            
            # Wall forces (keep circles inside)
            wall_stiffness = 100.0
            for i in range(N):
                # Left wall
                if xs[i] < rs[i]:
                    violation = rs[i] - xs[i]
                    fx[i] += wall_stiffness * violation
                # Right wall
                if xs[i] > W - rs[i]:
                    violation = xs[i] - (W - rs[i])
                    fx[i] -= wall_stiffness * violation
                # Bottom wall
                if ys[i] < rs[i]:
                    violation = rs[i] - ys[i]
                    fy[i] += wall_stiffness * violation
                # Top wall
                if ys[i] > H - rs[i]:
                    violation = ys[i] - (H - rs[i])
                    fy[i] -= wall_stiffness * violation
            
            # Update positions with velocity damping
            xs += fx * dt
            ys += fy * dt
            # Apply damping (implicit in the small dt and force scaling)
            
            # Grow radii gradually (pressure-driven expansion)
            # Only grow if there's "room" - check if any circle is touching a wall or another
            can_grow = True
            for i in range(N):
                # Check wall proximity
                if xs[i] < rs[i] * 1.01 or xs[i] > W - rs[i] * 0.99:
                    can_grow = False
                    break
                if ys[i] < rs[i] * 1.01 or ys[i] > H - rs[i] * 0.99:
                    can_grow = False
                    break
                # Check neighbor proximity
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(xs[i]-xs[j], ys[i]-ys[j])
                    if d < (rs[i] + rs[j]) * 1.01:
                        can_grow = False
                        break
                if not can_grow:
                    break
            
            if can_grow:
                rs += growth_rate
            else:
                # Try to redistribute: shrink slightly and let forces push apart
                # This helps escape local minima
                if step % 50 == 0:
                    rs *= 0.98
                    # Add small random perturbation to break symmetry
                    xs += np.random.normal(0, 0.005, N)
                    ys += np.random.normal(0, 0.005, N)
            
            # Clamp positions to valid range (soft constraint)
            for i in range(N):
                xs[i] = np.clip(xs[i], rs[i]*0.5, W - rs[i]*0.5)
                ys[i] = np.clip(ys[i], rs[i]*0.5, H - rs[i]*0.5)
        
        return np.concatenate([xs, ys, rs, [W]])

    def relax_radii(x, n_iter=100):
        """
        Fix positions, maximize radii iteratively.
        """
        x = x.copy()
        xs = x[:N].copy()
        ys = x[N:2*N].copy()
        rs = x[2*N:3*N].copy()
        W = x[3*N]
        H = 2.0 - W
        
        for _ in range(n_iter):
            improved = False
            for i in range(N):
                # Max radius allowed by walls
                r_wall = min(xs[i], W - xs[i], ys[i], H - ys[i])
                r_max = r_wall
                # Max radius allowed by neighbors
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    r_neighbor = d - rs[j]
                    if r_neighbor < r_max:
                        r_max = r_neighbor
                r_max = max(r_max, 1e-8)
                if r_max > rs[i] + 1e-9:
                    rs[i] = r_max
                    improved = True
            if not improved:
                break
        x[2*N:3*N] = rs
        return x

    def grid_init(W, H, cols=None, rows=None):
        if cols is None:
            cols = int(np.ceil(np.sqrt(N)))
        if rows is None:
            rows = int(np.ceil(N / cols))
            
        dx = W / cols
        dy = H / rows
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        k = 0
        for i in range(cols):
            for j in range(rows):
                if k >= N:
                    break
                xs[k] = (i + 0.5) * dx
                ys[k] = (j + 0.5) * dy
                rs[k] = 0.3 * min(dx, dy)
                k += 1
        if k < N:
            for m in range(k, N):
                xs[m] = W/2
                ys[m] = H/2
                rs[m] = 0.01
        
        return np.concatenate([xs, ys, rs, [W]])

    def hex_init(W, H):
        """Hexagonal packing initialization."""
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        # Estimate radius for hex packing
        # For hex packing, area per circle ~ (4*sqrt(3)/3) * r^2
        # Total area = W*H, so r ~ sqrt(W*H / (N * 4*sqrt(3)/3))
        r_est = np.sqrt(W*H / (N * 4*np.sqrt(3)/3)) * 0.9
        
        spacing = 2.1 * r_est
        row_h = spacing * np.sqrt(3)/2
        
        k = 0
        row = 0
        while k < N:
            offset = (row % 2) * spacing/2
            col = 0
            while k < N:
                x = spacing/2 + offset + col * spacing
                y = spacing/2 + row * row_h
                if x + r_est > W or y + r_est > H:
                    break
                xs[k] = x
                ys[k] = y
                rs[k] = r_est * 0.8
                k += 1
                col += 1
            row += 1
            if row * row_h > H + spacing:
                break
        
        # Fill remaining
        for m in range(k, N):
            xs[m] = W/2
            ys[m] = H/2
            rs[m] = 0.05
        
        return np.concatenate([xs, ys, rs, [W]])

    def random_init(W, H):
        xs = np.random.uniform(0.2, W-0.2, N)
        ys = np.random.uniform(0.2, H-0.2, N)
        rs = np.random.uniform(0.05, 0.12, N)
        return np.concatenate([xs, ys, rs, [W]])

    # Main search loop
    attempts = 0
    max_attempts = 30
    
    # Try various aspect ratios
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.83, 1.3, 0.77, 1.4, 0.7, 1.5]
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Select W
        if attempts <= len(aspect_ratios):
            W = aspect_ratios[attempts - 1]
        else:
            W = np.random.uniform(0.65, 1.35)
        
        H = 2.0 - W
        if H < 0.5:
            W = 1.5
            H = 0.5
        
        # Choose initialization strategy
        init_type = attempts % 4
        if init_type == 0:
            x0 = hex_init(W, H)
        elif init_type == 1:
            x0 = grid_init(W, H, cols=5, rows=5)
        elif init_type == 2:
            x0 = grid_init(W, H, cols=4, rows=6)
        else:
            x0 = random_init(W, H)
        
        # Run force-directed simulation
        x_sim = force_directed_simulate(W, H, x0, n_steps=1500, dt=0.001)
        
        # Post-process: Relax radii to maximum
        x_relax = relax_radii(x_sim, n_iter=50)
        
        # Check validity and score
        score = get_score(x_relax)
        
        if score > best_score:
            best_score = score
            best_result = x_relax.copy()
        
        # If we have time, try a second simulation with different parameters
        if best_result is not None and time.time() < deadline - 15 and attempts > 3:
            # Perturb best result
            x_perturb = best_result.copy()
            x_perturb[:N] += np.random.normal(0, 0.02, N)
            x_perturb[N:2*N] += np.random.normal(0, 0.02, N)
            x_perturb[3*N] = np.clip(x_perturb[3*N], 0.5, 1.5)
            
            x_sim2 = force_directed_simulate(W, H, x_perturb, n_steps=800, dt=0.001)
            x_relax2 = relax_radii(x_sim2, n_iter=30)
            score2 = get_score(x_relax2)
            if score2 > best_score:
                best_score = score2
                best_result = x_relax2.copy()

    # Final polish
    if best_result is not None and time.time() < deadline - 3:
        best_result = relax_radii(best_result, n_iter=100)
        s = get_score(best_result)
        if s > best_score:
            best_score = s

    if best_result is None:
        # Fallback: Simple grid
        side = 5
        W = 1.0
        H = 1.0
        cells = []
        k = 0
        for i in range(side):
            for j in range(side):
                if k >= N:
                    break
                cx = (i + 0.5) * (W/side)
                cy = (j + 0.5) * (H/side)
                r = 0.3 * (W/side)
                cells.append([cx, cy, r])
                k += 1
        return np.array(cells)

    xs = best_result[:N]
    ys = best_result[N:2*N]
    rs = best_result[2*N:3*N]
    result = np.column_stack((xs, ys, rs))
    return result
