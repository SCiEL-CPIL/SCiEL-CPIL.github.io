"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii.
    Uses a physical simulation approach:
    1. Initialize circles in a grid.
    2. Iteratively grow radii and resolve overlaps using repulsive forces.
    """
    # Initialize positions in a grid
    side = int(np.ceil(np.sqrt(N)))
    # Ensure we have enough spots, e.g. 6x6 grid for 32 circles
    # We will pick the first N spots from a 6x6 grid
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    # Grid coordinates
    grid_x = np.linspace(0.5/side, 1 - 0.5/side, side)
    grid_y = np.linspace(0.5/side, 1 - 0.5/side, side)
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx < N:
                xs[idx] = grid_x[i]
                ys[idx] = grid_y[j]
                rs[idx] = 0.01 # Small initial radius
                idx += 1
    
    # Simulation parameters
    dt = 0.01
    damping = 0.5
    growth_rate = 0.005
    max_steps = 5000
    relax_steps = 20
    
    # Velocities
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    best_sum_r = np.sum(rs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()
    
    for step in range(max_steps):
        # 1. Grow radii
        rs += growth_rate
        
        # 2. Relax positions to resolve overlaps
        for _ in range(relax_steps):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    
                    if dist < min_dist:
                        if dist < 1e-6:
                            # Randomize if exactly same position
                            dx = np.random.randn() * 1e-3
                            dy = np.random.randn() * 1e-3
                            dist = np.sqrt(dx*dx + dy*dy)
                        
                        # Force proportional to overlap
                        force = (min_dist - dist) * 10.0
                        # Normalize direction
                        nx = dx / dist
                        ny = dy / dist
                        
                        fx[i] -= force * nx
                        fy[i] -= force * ny
                        fx[j] += force * nx
                        fy[j] += force * ny
            
            # Boundary repulsion
            for i in range(N):
                # Left wall
                if xs[i] - rs[i] < 0:
                    force = (0 - (xs[i] - rs[i])) * 10.0
                    fx[i] += force
                # Right wall
                if xs[i] + rs[i] > 1:
                    force = (1 - (xs[i] + rs[i])) * 10.0
                    fx[i] -= force
                # Bottom wall
                if ys[i] - rs[i] < 0:
                    force = (0 - (ys[i] - rs[i])) * 10.0
                    fy[i] += force
                # Top wall
                if ys[i] + rs[i] > 1:
                    force = (1 - (ys[i] + rs[i])) * 10.0
                    fy[i] -= force
            
            # Update velocities and positions
            vx = damping * vx + dt * fx
            vy = damping * vy + dt * fy
            
            xs += vx
            ys += vy
            
            # Hard clip to stay inside (safety net)
            xs = np.clip(xs, rs, 1 - rs)
            ys = np.clip(ys, rs, 1 - rs)
            
            # If any radius is too big to fit, shrink it? 
            # Actually, if a circle is pushed out, the force should pull it back.
            # But if r > 0.5, it can't fit. 
            # Max possible radius is 0.5 (if alone). With 32 circles, much smaller.
            # We can cap r at 0.5.
            rs = np.minimum(rs, 0.5)
            
        # Check validity and record best
        # Simple validity check: no overlaps, inside bounds
        valid = True
        for i in range(N):
            if xs[i] - rs[i] < -1e-5 or xs[i] + rs[i] > 1 + 1e-5:
                valid = False
                break
            if ys[i] - rs[i] < -1e-5 or ys[i] + rs[i] > 1 + 1e-5:
                valid = False
                break
            if rs[i] < -1e-5:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist < rs[i] + rs[j] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current_sum = np.sum(rs)
            if current_sum > best_sum_r:
                best_sum_r = current_sum
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_rs = rs.copy()
                
            # If we've reached a state where growth doesn't help much, stop?
            # Or just run to completion.
            
    # Final cleanup: ensure strict validity
    # The simulation might leave tiny overlaps due to floating point.
    # We can do a final "push apart" pass without growing.
    
    final_xs = best_xs.copy()
    final_ys = best_ys.copy()
    final_rs = best_rs.copy()
    
    # One last relaxation without growth to ensure no overlaps
    for _ in range(100):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = final_xs[j] - final_xs[i]
                dy = final_ys[j] - final_ys[i]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = final_rs[i] + final_rs[j]
                
                if dist < min_dist:
                    if dist < 1e-6:
                        dx = np.random.randn() * 1e-3
                        dy = np.random.randn() * 1e-3
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    force = (min_dist - dist) * 10.0
                    nx = dx / dist
                    ny = dy / dist
                    
                    fx[i] -= force * nx
                    fy[i] -= force * ny
                    fx[j] += force * nx
                    fy[j] += force * ny
        
        for i in range(N):
            if final_xs[i] - final_rs[i] < 0:
                force = (0 - (final_xs[i] - final_rs[i])) * 10.0
                fx[i] += force
            if final_xs[i] + final_rs[i] > 1:
                force = (1 - (final_xs[i] + final_rs[i])) * 10.0
                fx[i] -= force
            if final_ys[i] - final_rs[i] < 0:
                force = (0 - (final_ys[i] - final_rs[i])) * 10.0
                fy[i] += force
            if final_ys[i] + final_rs[i] > 1:
                force = (1 - (final_ys[i] + final_rs[i])) * 10.0
                fy[i] -= force
        
        # Update with small step
        final_xs += 0.01 * fx
        final_ys += 0.01 * fy
        
        final_xs = np.clip(final_xs, final_rs, 1 - final_rs)
        final_ys = np.clip(final_ys, final_rs, 1 - final_rs)
        
    # Construct output array
    result = np.zeros((N, 3))
    result[:, 0] = final_xs
    result[:, 1] = final_ys
    result[:, 2] = final_rs
    
    return result
# EVOLVE-BLOCK-END
