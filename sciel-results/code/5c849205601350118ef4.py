"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 0
    best_result = None
    
    # Physics-based optimization with Lennard-Jones potential
    def lj_force(x, y, r1, r2, sigma=1.0):
        """Compute Lennard-Jones force between two circles."""
        dx = x[0] - y[0]
        dy = x[1] - y[1]
        dist = np.sqrt(dx**2 + dy**2) + 1e-10
        # LJ potential: 4*eps*((sigma/r)^12 - (sigma/r)^6)
        # Force: -dV/dr = 4*eps*(12*sigma^12/r^13 - 6*sigma^6/r^7)
        # We want repulsion when dist < r1+r2
        min_dist = r1 + r2
        if dist < min_dist:
            # Strong repulsion
            f = 100.0 * (min_dist - dist) / dist
            return np.array([f * dx, f * dy])
        else:
            return np.array([0.0, 0.0])
    
    def physics_simulate(init_pos, init_rad, width, height, steps=500, dt=0.01):
        """Run physics simulation with velocity-Verlet integration."""
        n = len(init_pos)
        pos = init_pos.copy().astype(float)
        vel = np.zeros_like(pos)
        rad = init_rad.copy().astype(float)
        
        # LJ parameters
        sigma = 0.1  # interaction range
        eps = 1.0    # energy scale
        
        for step in range(steps):
            # Compute forces
            force = np.zeros_like(pos)
            
            # Pairwise LJ forces
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = np.sqrt(dx**2 + dy**2) + 1e-10
                    min_dist = rad[i] + rad[j]
                    
                    if dist < min_dist + sigma:
                        # Repulsive force
                        overlap = (min_dist + sigma - dist) / (min_dist + sigma)
                        f = eps * overlap**2 / dist
                        force[i][0] += f * dx
                        force[i][1] += f * dy
                        force[j][0] -= f * dx
                        force[j][1] -= f * dy
            
            # Boundary forces (soft walls)
            for i in range(n):
                # Left wall
                if pos[i][0] < rad[i]:
                    force[i][0] += eps * (rad[i] - pos[i][0])
                # Right wall
                if pos[i][0] > width - rad[i]:
                    force[i][0] -= eps * (pos[i][0] - (width - rad[i]))
                # Bottom wall
                if pos[i][1] < rad[i]:
                    force[i][1] += eps * (rad[i] - pos[i][1])
                # Top wall
                if pos[i][1] > height - rad[i]:
                    force[i][1] -= eps * (pos[i][1] - (height - rad[i]))
            
            # Velocity-Verlet integration
            pos += vel * dt + 0.5 * force * dt**2
            vel_new = vel + force * dt
            vel = vel_new * 0.95  # damping
            
            # Keep circles inside bounds
            for i in range(n):
                pos[i][0] = np.clip(pos[i][0], rad[i], width - rad[i])
                pos[i][1] = np.clip(pos[i][1], rad[i], height - rad[i])
        
        return pos, rad
    
    def evaluate_solution(pos, rad, width, height):
        """Check if solution is valid and compute score."""
        n = len(pos)
        # Check bounds
        for i in range(n):
            if pos[i][0] < rad[i] - 1e-6 or pos[i][0] > width - rad[i] + 1e-6:
                return 0
            if pos[i][1] < rad[i] - 1e-6 or pos[i][1] > height - rad[i] + 1e-6:
                return 0
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < rad[i] + rad[j] - 1e-6:
                    return 0
        
        return np.sum(rad)
    
    def gradient_polish(pos, rad, width, height, max_iter=1000):
        """Polish solution using gradient descent on constraint violations."""
        n = len(pos)
        pos = pos.copy()
        rad = rad.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute constraint violations
            violations = []
            for i in range(n):
                # Boundary constraints
                if pos[i][0] < rad[i]:
                    violations.append((i, 'left', rad[i] - pos[i][0]))
                if pos[i][0] > width - rad[i]:
                    violations.append((i, 'right', pos[i][0] - (width - rad[i])))
                if pos[i][1] < rad[i]:
                    violations.append((i, 'bottom', rad[i] - pos[i][1]))
                if pos[i][1] > height - rad[i]:
                    violations.append((i, 'top', pos[i][1] - (height - rad[i])))
                
                # Overlap constraints
                for j in range(i+1, n):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = np.sqrt(dx**2 + dy**2) + 1e-10
                    if dist < rad[i] + rad[j]:
                        overlap = rad[i] + rad[j] - dist
                        violations.append((i, j, overlap))
            
            if not violations:
                break
            
            # Apply corrections
            for v in violations:
                if len(v) == 3 and isinstance(v[1], str):
                    # Boundary violation
                    i, side, amount = v
                    if side == 'left':
                        pos[i][0] += amount * 0.5
                    elif side == 'right':
                        pos[i][0] -= amount * 0.5
                    elif side == 'bottom':
                        pos[i][1] += amount * 0.5
                    elif side == 'top':
                        pos[i][1] -= amount * 0.5
                else:
                    # Overlap violation
                    i, j, overlap = v
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = np.sqrt(dx**2 + dy**2) + 1e-10
                    pos[i][0] += overlap * 0.5 * dx / dist
                    pos[i][1] += overlap * 0.5 * dy / dist
                    pos[j][0] -= overlap * 0.5 * dx / dist
                    pos[j][1] -= overlap * 0.5 * dy / dist
            
            # Keep inside bounds
            for i in range(n):
                pos[i][0] = np.clip(pos[i][0], rad[i], width - rad[i])
                pos[i][1] = np.clip(pos[i][1], rad[i], height - rad[i])
        
        return pos, rad
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.5]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Perimeter = 4, so 2*(w+h) = 4 => w+h = 2
        # w = aspect * h, so aspect*h + h = 2 => h = 2/(aspect+1)
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Initialize circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        init_pos = []
        init_rad = []
        
        for i in range(side):
            for j in range(side):
                if len(init_pos) >= N:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) * 0.95
                init_pos.append([x, y])
                init_rad.append(r)
        
        init_pos = np.array(init_pos[:N])
        init_rad = np.array(init_rad[:N])
        
        # Run physics simulation
        pos, rad = physics_simulate(init_pos, init_rad, w, h, steps=300)
        
        # Polish
        pos, rad = gradient_polish(pos, rad, w, h, max_iter=500)
        
        # Evaluate
        score = evaluate_solution(pos, rad, w, h)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, rad])
    
    if best_result is None:
        # Fallback to seed solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
