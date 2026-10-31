"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Multiple restarts to escape local minima
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialization: Jittered Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    # Add small random jitter to break symmetry
                    x += np.random.uniform(-0.1, 0.1) / side
                    y += np.random.uniform(-0.1, 0.1) / side
                    cells.append([x, y])
        
        # Pad if needed (should not happen for N=26, side=5 -> 25 cells, need 1 more? 
        # sqrt(26) is ~5.1. So side=5 gives 25 cells. We need 26.
        # Let's just use a random scatter if grid isn't enough, or extend grid.
        # Actually, for N=26, a 5x5 grid has 25 spots. 
        # Let's use a slightly denser random init or a 5x5 grid + 1 random.
        # Better: Just use random init for variety, or a specific pattern.
        # Let's stick to random init for robustness against grid traps.
        
        positions = np.random.rand(N, 2) * 0.8 + 0.1  # Keep away from edges initially
        radii = np.full(N, 0.05)
        
        # 2. Grow and Relax
        # We iteratively increase radii and relax positions
        max_steps = 500
        relax_steps = 200
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Increase radii
            # Heuristic: increase by a small amount
            growth = 0.002
            radii += growth
            
            # Relax positions
            positions, radii = _relax(positions, radii, relax_steps, deadline)
            
            # Check if we can grow more or if we are stuck
            # If overlap is still high after relaxation, we might be stuck.
            # For now, we just keep going.
            
        # 3. Polish with SLSQP
        if time.time() < deadline - 5.0:
            polished_pos, polished_rad = _polish(positions, radii, deadline)
            if polished_pos is not None:
                score = np.sum(polished_rad)
                if score > best_score:
                    best_score = score
                    best_circles = np.column_stack([polished_pos, polished_rad])
        
        # Early stopping if we hit a very high score
        if best_score > 1.25:
             break

    if best_circles is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles

def _relax(positions, radii, steps, deadline):
    """
    Force-directed relaxation to resolve overlaps and keep circles in bounds.
    """
    N = len(positions)
    positions = positions.copy()
    radii = radii.copy()
    
    # Precompute pairs
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    eta = 0.05  # Step size
    damping = 0.9
    velocities = np.zeros_like(positions)
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        for i, j in pairs:
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                if dist > 1e-8:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force_mag = overlap * 2.0
                    force_vec = (diff / dist) * force_mag
                else:
                    # If coincident, push apart randomly
                    force_vec = np.random.randn(2) * 1.0
                    force_mag = 1.0
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Wall repulsion
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                forces[i][0] += (r - x) * 2.0
            # Right wall
            if x > 1.0 - r:
                forces[i][0] -= (x - (1.0 - r)) * 2.0
            # Bottom wall
            if y < r:
                forces[i][1] += (r - y) * 2.0
            # Top wall
            if y > 1.0 - r:
                forces[i][1] -= (y - (1.0 - r)) * 2.0
        
        # Update velocities and positions
        velocities = damping * velocities + eta * forces
        positions += velocities
        
        # Clamp positions to [0, 1] to prevent numerical explosion
        positions = np.clip(positions, 0.0, 1.0)
        
        # Optional: Shrink radii slightly if they are too large for the space?
        # No, we want to maximize radii. The relaxation should handle it.
        # However, if a circle is forced out, we might need to shrink it.
        # Let's ensure radii don't exceed 0.5 (half the square)
        radii = np.clip(radii, 0.0, 0.5)
        
        # Reduce step size over time
        eta *= 0.999
        
    return positions, radii

def _polish(positions, radii, deadline):
    """
    Local optimization using SLSQP to maximize sum of radii.
    """
    N = len(positions)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = positions[i][0]
        x0[3*i+1] = positions[i][1]
        x0[3*i+2] = radii[i]
        
    # Objective: Minimize -sum(radii)
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii)
        
    # Constraints:
    # 1. Wall constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # 2. Pairwise: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    
    constraints = []
    
    # Wall constraints
    for i in range(N):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i] - x[3*i+2]
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]
        })
        
    # Pairwise constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
            })
            
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-6, 0.5)) # r
        
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success:
            x_opt = res.x
            pos = np.array([[x_opt[3*i], x_opt[3*i+1]] for i in range(N)])
            rad = np.array([x_opt[3*i+2] for i in range(N)])
            return pos, rad
        else:
            # Return original if optimization failed
            return positions, radii
            
    except Exception:
        return positions, radii

# EVOLVE-BLOCK-END
