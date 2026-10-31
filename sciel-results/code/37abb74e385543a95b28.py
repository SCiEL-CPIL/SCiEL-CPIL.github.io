"""Physics-based circle packing using Jellies algorithm."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Random positions, small radii
    np.random.seed(42)
    x = np.random.uniform(0.05, 0.95, N)
    y = np.random.uniform(0.05, 0.95, N)
    r = np.full(N, 0.01)
    v = np.zeros((N, 2))
    
    # Precompute pairs for efficiency
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Simulation parameters
    k_rep = 1000.0  # Repulsion stiffness
    k_wall = 1000.0 # Wall stiffness
    damping = 0.95  # Velocity damping
    dt = 0.005     # Time step
    max_iter = 20000
    
    best_score = -np.inf
    best_state = None
    
    for step in range(max_iter):
        # Check time
        if time.time() > deadline:
            break
            
        # Reset forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # 1. Pairwise repulsion (Jellies force)
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            
            # Avoid division by zero
            if dist < 1e-6:
                continue
                
            sum_r = r[i] + r[j]
            
            # Force is proportional to overlap
            # If dist < sum_r, they overlap.
            # Jellies uses a linear force: F = k * (sum_r - dist)
            overlap = sum_r - dist
            if overlap > 0:
                force_mag = k_rep * overlap
                fx[i] += force_mag * (dx / dist)
                fy[i] += force_mag * (dy / dist)
                fx[j] -= force_mag * (dx / dist)
                fy[j] -= force_mag * (dy / dist)
                
        # 2. Wall constraints (Repulsion from boundary)
        # We want circles to stay inside [0,1]^2
        # If x[i] - r[i] < 0, push right
        # If x[i] + r[i] > 1, push left
        for i in range(N):
            # Left wall
            if x[i] - r[i] < 0:
                overlap = r[i] - x[i]
                fx[i] += k_wall * overlap
            # Right wall
            if x[i] + r[i] > 1:
                overlap = x[i] + r[i] - 1
                fx[i] -= k_wall * overlap
            # Bottom wall
            if y[i] - r[i] < 0:
                overlap = r[i] - y[i]
                fy[i] += k_wall * overlap
            # Top wall
            if y[i] + r[i] > 1:
                overlap = y[i] + r[i] - 1
                fy[i] -= k_wall * overlap
                
        # 3. Update velocities and positions (Velocity Verlet / Semi-implicit Euler)
        v[:, 0] = (v[:, 0] + fx * dt) * damping
        v[:, 1] = (v[:, 1] + fy * dt) * damping
        
        x += v[:, 0] * dt
        y += v[:, 1] * dt
        
        # 4. Grow radii
        # Grow all radii by a small amount.
        # The forces will push them apart if they overlap.
        # If the system is stable, the radii can grow.
        # We use a constant growth rate, but maybe decay it?
        # A constant growth rate with strong repulsion leads to a dynamic equilibrium
        # where radii grow until the "pressure" from repulsion balances the "pressure" from growth?
        # Actually, standard Jellies grows radii continuously. The positions adjust to accommodate.
        # We need to ensure we don't grow so fast that the system explodes.
        # Let's try a constant growth rate.
        growth_rate = 0.0005
        r += growth_rate
        
        # 5. Check validity and score
        # A valid packing requires:
        # 1. x[i] >= r[i], x[i] <= 1-r[i], etc. (No wall penetration)
        # 2. dist(i,j) >= r[i] + r[j] (No circle overlap)
        
        # For scoring, we want to maximize sum(r).
        # But if the configuration is invalid (overlapping), the score should be 0 or negative?
        # The task says "maximize the sum of their radii" for "disjoint interiors".
        # The evaluator measures the "raw value". If invalid, score is 0.
        # So we should only update best_state if the current state is VALID.
        
        # Check validity
        valid = True
        # Walls
        if np.any(x < r) or np.any(x > 1-r) or np.any(y < r) or np.any(y > 1-r):
            valid = False
        # Pairs
        if valid:
            for i, j in pairs:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    valid = False
                    break
        
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_state = (x.copy(), y.copy(), r.copy())
                
        # Optional: If we get stuck (low kinetic energy, no growth in score), we might need to perturb?
        # But Jellies usually works if parameters are tuned.
        
    # If we never found a valid state (unlikely with small initial r), return a trivial valid state
    if best_state is None:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x_best, y_best, r_best = best_state
    return np.column_stack((x_best, y_best, r_best))
# EVOLVE-BLOCK-END
