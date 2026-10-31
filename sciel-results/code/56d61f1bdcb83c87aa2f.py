"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    
    def simulate(w, h, xs, ys, rs, steps=200, dt=0.005, damping=0.9):
        """
        Run physics simulation to resolve overlaps.
        Returns updated xs, ys.
        """
        xs = xs.copy()
        ys = ys.copy()
        vs_x = np.zeros(N)
        vs_y = np.zeros(N)
        
        for step in range(steps):
            fs_x = np.zeros(N)
            fs_y = np.zeros(N)
            
            # Circle-Circle forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist and dist > 1e-6:
                        # Repulsive force proportional to overlap
                        force_mag = (min_dist - dist)
                        # Normalize direction
                        fs_x[i] += force_mag * (dx / dist)
                        fs_y[i] += force_mag * (dy / dist)
                        fs_x[j] -= force_mag * (dx / dist)
                        fs_y[j] -= force_mag * (dy / dist)
                    elif dist <= 1e-6:
                        # Handle exact overlap with random nudge
                        fs_x[i] += np.random.uniform(-1, 1)
                        fs_y[i] += np.random.uniform(-1, 1)
                        fs_x[j] -= np.random.uniform(-1, 1)
                        fs_y[j] -= np.random.uniform(-1, 1)
                        
            # Wall forces
            for i in range(N):
                # Left
                if xs[i] < rs[i]:
                    fs_x[i] += (rs[i] - xs[i]) * 2.0 # Stronger wall force
                # Right
                if xs[i] > w - rs[i]:
                    fs_x[i] -= (xs[i] - (w - rs[i])) * 2.0
                # Bottom
                if ys[i] < rs[i]:
                    fs_y[i] += (rs[i] - ys[i]) * 2.0
                # Top
                if ys[i] > h - rs[i]:
                    fs_y[i] -= (ys[i] - (h - rs[i])) * 2.0
                    
            # Update velocities
            vs_x = vs_x * damping + fs_x * dt
            vs_y = vs_y * damping + fs_y * dt
            
            # Update positions
            xs += vs_x * dt
            ys += vs_y * dt
            
            # Hard constraint projection to keep inside (optional, helps stability)
            for i in range(N):
                if xs[i] < rs[i]: xs[i] = rs[i]
                if xs[i] > w - rs[i]: xs[i] = w - rs[i]
                if ys[i] < rs[i]: ys[i] = rs[i]
                if ys[i] > h - rs[i]: ys[i] = h - rs[i]
                
        return xs, ys

    def calculate_overlap_energy(xs, ys, rs, w, h):
        energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    energy += (min_dist - dist) ** 2
        # Wall violations
        for i in range(N):
            if xs[i] < rs[i]: energy += (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]: energy += (xs[i] - (w - rs[i])) ** 2
            if ys[i] < rs[i]: energy += (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]: energy += (ys[i] - (h - rs[i])) ** 2
        return energy

    best_value = 0.0
    best_solution = None
    
    np.random.seed(42)
    
    # Try different aspect ratios
    for w in np.linspace(0.8, 1.2, 9):
        h = 2.0 - w
        
        # Initialize circles in a grid
        cols = 5
        rows = 5
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        # Grid placement
        for i in range(N):
            col = i % cols
            row = i // cols
            xs[i] = (col + 0.5) / cols * w
            ys[i] = (row + 0.5) / rows * h
            # Initial small radius
            rs[i] = 0.02
            
        # Growth phase
        growth_step = 0.005
        max_growth_steps = 500
        
        for g_step in range(max_growth_steps):
            # Increase radii
            rs += growth_step
            
            # Check if radii exceed max possible (e.g., min(w,h)/2 for one circle, but we have many)
            # Simple heuristic: if average radius is too large, stop
            if np.mean(rs) > 0.15: 
                break
                
            # Simulate
            xs, ys = simulate(w, h, xs, ys, rs, steps=50, dt=0.01, damping=0.8)
            
            # Check energy
            energy = calculate_overlap_energy(xs, ys, rs, w, h)
            
            # If energy is too high, we might be stuck or growing too fast
            if energy > 0.01:
                # Reduce growth step and maybe add noise
                growth_step *= 0.9
                # Add small random noise to escape local minima
                xs += np.random.uniform(-0.001, 0.001, N)
                ys += np.random.uniform(-0.001, 0.001, N)
                # Re-simulate to settle
                xs, ys = simulate(w, h, xs, ys, rs, steps=100, dt=0.01, damping=0.8)
                
            # Verify validity
            if energy < 1e-5:
                val = np.sum(rs)
                if val > best_value:
                    best_value = val
                    best_solution = np.column_stack((xs, ys, rs))
            else:
                # If invalid, revert radii growth for this step? 
                # Or just keep going, hoping it resolves.
                # For robustness, if energy is very high, shrink radii slightly
                if energy > 0.1:
                    rs -= growth_step * 0.5
                    break

    # Final refinement: Local optimization of positions for the best solution found
    if best_solution is not None:
        xs, ys, rs = best_solution[:, 0], best_solution[:, 1], best_solution[:, 2]
        w_best = None
        # We need to know which w was used for best_solution. 
        # Since we didn't store w, let's re-run the logic or store it.
        # Let's modify the loop to store w.
        
    # Let's rewrite the loop structure to properly track the best w.
    
    best_value = 0.0
    best_solution = None
    
    for w in np.linspace(0.8, 1.2, 9):
        h = 2.0 - w
        
        cols = 5
        rows = 5
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            xs[i] = (col + 0.5) / cols * w
            ys[i] = (row + 0.5) / rows * h
            rs[i] = 0.02
            
        growth_step = 0.005
        
        for g_step in range(500):
            rs += growth_step
            if np.mean(rs) > 0.15:
                break
                
            xs, ys = simulate(w, h, xs, ys, rs, steps=50, dt=0.01, damping=0.8)
            energy = calculate_overlap_energy(xs, ys, rs, w, h)
            
            if energy < 1e-5:
                val = np.sum(rs)
                if val > best_value:
                    best_value = val
                    best_solution = np.column_stack((xs, ys, rs))
            else:
                if energy > 0.1:
                    break
                # Add noise and re-simulate
                xs += np.random.uniform(-0.001, 0.001, N)
                ys += np.random.uniform(-0.001, 0.001, N)
                xs, ys = simulate(w, h, xs, ys, rs, steps=100, dt=0.01, damping=0.8)
                energy = calculate_overlap_energy(xs, ys, rs, w, h)
                if energy < 1e-5:
                    val = np.sum(rs)
                    if val > best_value:
                        best_value = val
                        best_solution = np.column_stack((xs, ys, rs))

    if best_solution is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
# EVOLVE-BLOCK-END
