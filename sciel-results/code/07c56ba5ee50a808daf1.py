"""
Circle packing in perimeter-4 rectangle.
Structural change: Use a physics-based repulsion/attraction simulation to find a dense packing,
then optimize the aspect ratio of the container.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # Safety margin before 150s kill

def simulate_packing(w, h, max_iters=2000, dt=0.005, seed=0):
    """
    Simulate circles in a w x h box.
    Returns: (x, y, r) arrays
    """
    rng = np.random.default_rng(seed)
    
    # Initial guess: Grid-like placement
    # Try to fit N circles in w x h
    # Estimate average radius
    area = w * h
    avg_r = np.sqrt(area / (N * np.pi))
    
    # Initialize positions on a grid
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Place centers
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            # Add some jitter to break symmetry
            x[idx] = (j + 0.5) * (w / cols) + rng.uniform(-0.1, 0.1) * (w/cols)
            y[idx] = (i + 0.5) * (h / rows) + rng.uniform(-0.1, 0.1) * (h/rows)
            r[idx] = 0.5 * (w / cols) * 0.9  # Initial radius guess
            idx += 1
    
    # Ensure within bounds initially
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    # Simulation loop
    # We want to maximize sum(r) subject to non-overlap and bounds.
    # Energy function: -sum(r) + penalty(overlap) + penalty(bounds)
    # Force: Grad(energy)
    
    # Simple Verlet-like or Euler integration with strong damping
    v = np.zeros((N, 2))
    
    # Adaptive timestep or fixed? Fixed with damping.
    
    for it in range(max_iters):
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # 1. Repulsion between circles (Overlap penalty)
        # If dist < r_i + r_j, push apart.
        # Also, we want them to grow.
        # Strategy: 
        #   - Constant outward force on each circle (growth).
        #   - Strong repulsion if overlapping.
        #   - Wall repulsion if touching bounds.
        
        # Growth force: Simply increase r? Or force pushing out?
        # Let's use a potential energy approach.
        # E = sum(0.5 * k_overlap * overlap^2) + sum(0.5 * k_wall * wall_pen^2) - sum(r)
        
        # This is complex to differentiate. Let's use direct forces.
        
        # Growth: Add a small force in all directions? No, that doesn't make sense for circles.
        # Instead, let's treat radius as a variable that increases if there is space.
        # Heuristic: 
        # 1. Resolve collisions (push circles apart).
        # 2. Push circles away from walls.
        # 3. If a circle has "free space" around it, increase its radius.
        
        # This "grow if space" is hard to do locally.
        # Alternative: 
        # Fix positions, solve for max radii? No, positions move.
        
        # Let's stick to standard force-based:
        # F_i = sum_{j != i} F_repel(i,j) + F_wall(i)
        # And we want to maximize sum r.
        # How to couple r?
        # Let's assume all circles have the same radius? No, variable.
        # 
        # New Strategy: 
        # 1. Start with small radii.
        # 2. Iteratively:
        #    a. Move circles to minimize overlap (using current radii).
        #    b. Increase radii by a small amount if no overlaps.
        #    c. If overlaps, reduce radii or push apart.
        
        # This "greedy growth" often gets stuck.
        
        # Better Strategy: 
        # Use a soft constraint energy.
        # E = sum_{i<j} max(0, r_i + r_j - d_ij)^2 + sum_i max(0, r_i - x_i)^2 + ...
        # Minimize E - lambda * sum(r_i)
        
        # Let's implement gradient descent on this energy.
        # Variables: x, y, r
        # N=21, so 63 variables.
        
        pass # This loop was a placeholder, let's implement the gradient descent version instead.
    
    # Since the simple force sim is tricky with variable radii, let's use scipy.optimize on the whole configuration.
    # But that might be slow.
    
    # Let's try a different physics approach: 
    # "Soft" circles.
    # Force between i and j:
    # if d < r_i + r_j: F = k * (r_i + r_j - d) (repulsive)
    # Wall force:
    # if x < r: Fx = k * (r - x)
    # Growth:
    # We want to increase r. 
    # Let's add a "pressure" term. 
    # Actually, let's just maximize sum(r) using the forces as constraints.
    
    # Let's restart the simulation logic inside this function properly.
    
    # State: x, y, r
    # Forces on x,y: Repulsion from overlaps, Repulsion from walls.
    # Forces on r: 
    #   - Gradient of overlap penalty w.r.t r_i (negative, penalizes large r)
    #   - Gradient of wall penalty w.r.t r_i (negative)
    #   - +1 (Objective: maximize sum r)
    
    k_overlap = 50.0
    k_wall = 50.0
    k_growth = 1.0
    damping = 0.9
    dt = 0.01
    
    for it in range(max_iters):
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # 1. Inter-circle forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_sum = r[i] + r[j]
                
                if d < 1e-6:
                    dx = 0.1
                    dy = 0.1
                    d = np.sqrt(dx*dx + dy*dy)
                
                if d < r_sum:
                    overlap = r_sum - d
                    # Force magnitude
                    f_mag = k_overlap * overlap
                    
                    # Direction (unit vector from j to i)
                    ux = dx / d
                    uy = dy / d
                    
                    fx[i] += f_mag * ux
                    fy[i] += f_mag * uy
                    fx[j] -= f_mag * ux
                    fy[j] -= f_mag * uy
                    
                    # Force on radii
                    # d(overlap^2)/dr_i = 2 * overlap
                    fr[i] -= k_overlap * overlap
                    fr[j] -= k_overlap * overlap
        
        # 2. Wall forces
        for i in range(N):
            # Left
            if x[i] < r[i]:
                pen = r[i] - x[i]
                fx[i] += k_wall * pen
                fr[i] -= k_wall * pen
            # Right
            if x[i] > w - r[i]:
                pen = x[i] - (w - r[i])
                fx[i] -= k_wall * pen
                fr[i] -= k_wall * pen
            
            # Bottom
            if y[i] < r[i]:
                pen = r[i] - y[i]
                fy[i] += k_wall * pen
                fr[i] -= k_wall * pen
            # Top
            if y[i] > h - r[i]:
                pen = y[i] - (h - r[i])
                fy[i] -= k_wall * pen
                fr[i] -= k_wall * pen
        
        # 3. Growth force
        fr += k_growth
        
        # Update velocities
        v[:, 0] = (v[:, 0] + fx * dt) * damping
        v[:, 1] = (v[:, 1] + fy * dt) * damping
        
        # Update positions
        x += v[:, 0] * dt
        y += v[:, 1] * dt
        
        # Update radii
        r += fr * dt
        
        # Constrain radii to be positive
        r = np.maximum(r, 1e-6)
        
        # Constrain positions to be within bounds (hard constraint to prevent divergence)
        # Note: Soft forces should handle this, but hard clip prevents blow-up
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        # Check for convergence or time limit
        # (Handled by outer loop)
        
    return x, y, r

def evaluate_packing(x, y, r, w, h):
    """
    Check if packing is valid and return score.
    Score = sum(r) if valid, else 0.
    """
    # Check bounds
    if np.any(x < -1e-6) or np.any(x > w + 1e-6) or np.any(y < -1e-6) or np.any(y > h + 1e-6):
        return 0.0
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - 1e-6:
                return 0.0
    
    return np.sum(r)

def construct():
    start_time = time.time()
    
    # We need to find w, h such that w + h = 2 (Perimeter 4 => 2(w+h)=4 => w+h=2)
    # Let w = t, h = 2 - t. t in (0, 2).
    # Optimal t is likely near 1 (square) or slightly off.
    
    best_score = 0.0
    best_result = None
    
    # Try a few aspect ratios
    # Square: w=1, h=1
    # Rectangles: w=1.2, h=0.8; w=0.8, h=1.2; etc.
    
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.5, 0.66, 0.75, 1.33]
    
    for ar in aspect_ratios:
        if time.time() > DEADLINE - 5:
            break
            
        # Determine w and h
        # Let w = ar * h
        # ar*h + h = 2 => h(1+ar) = 2 => h = 2 / (1 + ar)
        h = 2.0 / (1.0 + ar)
        w = ar * h
        
        # Run simulation
        # Use multiple seeds/restarts if time permits
        max_restarts = 3
        for seed in range(max_restarts):
            if time.time() > DEADLINE - 5:
                break
            
            x, y, r = simulate_packing(w, h, max_iters=1000, dt=0.01, seed=seed)
            
            score = evaluate_packing(x, y, r, w, h)
            
            if score > best_score:
                best_score = score
                # Store result
                # Format: (n, 3) array of [x, y, r]
                best_result = np.column_stack((x, y, r))
                
            # Optional: Local refinement with scipy if score is good?
            # For now, just keep the best from simulation.
            
    if best_result is None:
        # Fallback: Grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        # Scale to fit perimeter 4 (w=1, h=1)
        return np.array(cells[:N])
        
    return best_result
