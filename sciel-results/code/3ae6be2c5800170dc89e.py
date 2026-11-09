"""Combined program: physics-based soft-body simulation + SLSQP polish."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s wall-clock limit

    best_result = None
    best_score = -1.0

    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_result, best_score
        if positions is None or radii is None:
            return
        if len(positions) != N or len(radii) != N:
            return
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            return
        if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
            return
        if np.any(radii < -1e-9):
            return
        # Boundary check
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-9:
                return
            if positions[i, 0] + radii[i] > w + 1e-9:
                return
            if positions[i, 1] - radii[i] < -1e-9:
                return
            if positions[i, 1] + radii[i] > h + 1e-9:
                return
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()

    def compute_grid_positions(rows, cols, w, h):
        """Generate grid positions for N circles."""
        if rows * cols < N:
            return None
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            if row == rows - 1:
                count_in_last_row = N - (rows - 1) * cols
                if count_in_last_row < cols:
                    k = count_in_last_row
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            positions[i] = [x, y]
        return positions

    def hex_layout(w, h):
        """Hexagonal packing layout for N circles."""
        # Try to arrange in hexagonal pattern
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        # Estimate radius from hex packing
        # For hex packing in w x h, density ~ 0.9069
        # Approximate: r ~ min(w, h) / (2 * cols)
        cols = int(math.ceil(math.sqrt(N * h / w)))
        if cols < 1:
            cols = 1
        rows_needed = int(math.ceil(N / cols))
        # Horizontal spacing
        dx = w / cols
        # Vertical spacing (hex: sqrt(3)/2 * dx)
        dy = dx * math.sqrt(3) / 2
        # Scale to fit
        total_h = (rows_needed - 1) * dy + dx
        scale = min(w / (cols * dx), h / total_h) if total_h > 0 else 1.0
        dx *= scale
        dy *= scale
        r = dx / 2 * 0.95
        idx = 0
        for row in range(rows_needed):
            if idx >= N:
                break
            offset = (dx / 2) if (row % 2 == 1) else 0
            for col in range(cols):
                if idx >= N:
                    break
                x = dx / 2 + col * dx + offset
                y = dy / 2 + row * dy
                if x + r > w or y + r > h:
                    # Clamp
                    x = min(x, w - r)
                    y = min(y, h - r)
                    x = max(x, r)
                    y = max(y, r)
                positions[idx] = [x, y]
                radii[idx] = r
                idx += 1
        # Fill remaining
        while idx < N:
            positions[idx] = [w / 2, h / 2]
            radii[idx] = r * 0.5
            idx += 1
        return positions, radii

    def physics_simulate(positions, radii, w, h, steps=300, dt=0.01):
        """
        Physics-based soft-body simulation using Lennard-Jones-like repulsive
        potentials and attractive boundary forces to allow circles to dynamically
        settle into optimal non-uniform positions and radii.
        
        Energy components:
        1. Inter-circle repulsion: LJ-like potential (strong at short range)
        2. Boundary attraction: push circles toward center to avoid walls
        3. Radius growth: gentle force to grow radii (maximize sum)
        4. Damping: velocity damping for stability
        """
        pos = positions.copy().astype(float)
        rad = radii.copy().astype(float)
        vel = np.zeros_like(pos)
        
        # Ensure initial validity (shrink if needed)
        for _ in range(20):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.9
                    valid = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.9
                        rad[j] *= 0.9
                        valid = False
            if valid:
                break

        # LJ parameters
        sigma_base = 0.1  # Reference distance scale
        eps_rep = 50.0     # Repulsion strength
        eps_att = 5.0      # Boundary attraction strength
        growth_rate = 0.002  # Radius growth per step
        damping = 0.9       # Velocity damping
        
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            # Compute forces
            forces = np.zeros_like(pos)
            
            # 1. Inter-circle LJ-like repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1e-6:
                        # Random nudge
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        dist = math.sqrt(dx * dx + dy * dy)
                    
                    # LJ-like: strong repulsion when close
                    sigma = sigma_base * (rad[i] + rad[j])
                    if dist < 2 * sigma:
                        # Repulsive force ~ 1/dist^2 (strong at short range)
                        r_ratio = dist / sigma
                        force_mag = eps_rep / (r_ratio ** 2 + 0.1)
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    
                    # Moderate attraction at mid-range (to pack tightly)
                    elif dist < 3 * sigma:
                        force_mag = eps_rep * 0.1 / (r_ratio ** 2 + 0.1)
                        fx = -force_mag * dx / dist  # Attractive
                        fy = -force_mag * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # 2. Boundary forces (push toward center)
            for i in range(N):
                # Left wall
                if pos[i, 0] - rad[i] < rad[i]:
                    force = eps_att * (rad[i] - (pos[i, 0] - rad[i])) / rad[i]
                    forces[i, 0] += force
                # Right wall
                if pos[i, 0] + rad[i] > w - rad[i]:
                    force = eps_att * ((pos[i, 0] + rad[i]) - (w - rad[i])) / rad[i]
                    forces[i, 0] -= force
                # Bottom wall
                if pos[i, 1] - rad[i] < rad[i]:
                    force = eps_att * (rad[i] - (pos[i, 1] - rad[i])) / rad[i]
                    forces[i, 1] += force
                # Top wall
                if pos[i, 1] + rad[i] > h - rad[i]:
                    force = eps_att * ((pos[i, 1] + rad[i]) - (h - rad[i])) / rad[i]
                    forces[i, 1] -= force
            
            # 3. Update velocities and positions
            for i in range(N):
                vel[i] = vel[i] * damping + forces[i] * dt
                pos[i, 0] += vel[i, 0]
                pos[i, 1] += vel[i, 1]
                
                # Hard boundary constraints
                if pos[i, 0] < rad[i]:
                    pos[i, 0] = rad[i]
                    vel[i, 0] = max(0, vel[i, 0])
                if pos[i, 0] > w - rad[i]:
                    pos[i, 0] = w - rad[i]
                    vel[i, 0] = min(0, vel[i, 0])
                if pos[i, 1] < rad[i]:
                    pos[i, 1] = rad[i]
                    vel[i, 1] = max(0, vel[i, 1])
                if pos[i, 1] > h - rad[i]:
                    pos[i, 1] = h - rad[i]
                    vel[i, 1] = min(0, vel[i, 1])
            
            # 4. Grow radii gently
            rad += growth_rate
            
            # 5. Resolve any remaining overlaps by shrinking
            for i in range(N):
                # Boundary validity
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.98
                        rad[j] *= 0.98

        # Final cleanup: ensure strict validity
        for _ in range(10):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95
                    valid = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.95
                        rad[j] *= 0.95
                        valid = False
            if valid:
                break

        return pos, rad

    def run_slsqp(w_init, positions, radii):
        """Polish with SLSQP optimization."""
        h = 2.0 - w_init
        if h <= 0.1 or w_init <= 0.1:
            return

        n_var = 3 * N + 1

        def objective(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[3*N]
            h = 2.0 - w
            if h <= 0.1 or w <= 0.1:
                return 1e10
            
            penalty = 0.0
            # Boundary
            for i in range(N):
                if xs[i] - rs[i] < 0:
                    penalty += 100 * (xs[i] - rs[i])**2
                if xs[i] + rs[i] > w:
                    penalty += 100 * (xs[i] + rs[i] - w)**2
                if ys[i] - rs[i] < 0:
                    penalty += 100 * (ys[i] - rs[i])**2
                if ys[i] + rs[i] > h:
                    penalty += 100 * (ys[i] + rs[i] - h)**2
            
            # Overlap
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        penalty += 100 * (min_d - dist)**2
            
            return -np.sum(rs) + penalty

        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((0, 1.0))  # r
        bounds.append((0.5, 1.5))  # w

        x0 = np.zeros(n_var)
        for i in range(N):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        x0[3*N] = w_init

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.fun < 1e9:  # Valid result
                v = res.x
                xs = v[0:3*N:3]
                ys = v[1:3*N:3]
                rs = v[2:3*N:3]
                w = v[3*N]
                h = 2.0 - w
                
                # Ensure radii non-negative
                rs = np.maximum(rs, 0)
                
                # Clamp positions
                for i in range(N):
                    xs[i] = max(rs[i], min(xs[i], w - rs[i]))
                    ys[i] = max(rs[i], min(ys[i], h - rs[i]))
                
                pos = np.column_stack([xs, ys])
                validate_and_update(pos, rs, w, h)
        except Exception:
            pass

    # Main loop over aspect ratios
    W_values = np.linspace(0.6, 1.4, 15)
    
    for W in W_values:
        if time.time() > deadline - 10.0:
            break
            
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Try multiple initializations
        initializations = []
        
        # Grid initialization
        for rows in range(3, 7):
            for cols in range(3, 8):
                if rows * cols < N:
                    continue
                ar_rect = W / H
                ar_grid = cols / rows
                if abs(ar_rect - ar_grid) > 2.0:
                    continue
                pos = compute_grid_positions(rows, cols, W, H)
                if pos is not None:
                    # Estimate radius
                    r_est = min(W / cols, H / rows) / 2 * 0.9
                    rad = np.full(N, r_est)
                    initializations.append((pos, rad))
        
        # Hexagonal initialization
        hex_pos, hex_rad = hex_layout(W, H)
        initializations.append((hex_pos, hex_rad))
        
        # Random initialization
        for _ in range(3):
            pos = np.random.uniform(0.1, max(0.2, W - 0.1), (N, 2))
            pos[:, 1] = np.random.uniform(0.1, max(0.2, H - 0.1), N)
            rad = np.full(N, 0.1)
            initializations.append((pos, rad))
        
        # Run physics simulation on each initialization
        for pos, rad in initializations:
            if time.time() > deadline - 5.0:
                break
            
            # Run physics simulation
            phys_pos, phys_rad = physics_simulate(pos, rad, W, H, steps=200)
            
            # Validate physics result
            validate_and_update(phys_pos, phys_rad, W, H)
            
            # Polish with SLSQP
            if time.time() < deadline - 5.0:
                run_slsqp(W, phys_pos, phys_rad)
    
    # Fallback to seed if no valid result
    if best_result is None:
        side = int(math.ceil(math.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
