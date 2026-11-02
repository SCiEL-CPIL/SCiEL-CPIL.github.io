"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """Pack 26 circles with disjoint interiors in [0,1]^2, maximizing sum of radii."""
    np.random.seed(42)
    
    # Objective: maximize sum of radii subject to:
    # 1. Each circle (x_i, y_i, r_i) is inside [0,1]^2: r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i
    # 2. No two circles overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # 3. r_i > 0
    
    # Use simulated annealing
    def initial_placement():
        """Create initial non-overlapping placement."""
        # Start with a grid and then adjust
        side = 6  # 6x6 grid = 36 cells, use 26
        positions = []
        radii = []
        
        # Use a spiral or random placement with collision checking
        max_attempts = 10000
        for _ in range(N):
            placed = False
            for _ in range(max_attempts):
                # Try random position with random radius
                r = np.random.uniform(0.02, 0.12)
                x = np.random.uniform(r, 1 - r)
                y = np.random.uniform(r, 1 - r)
                
                # Check if this circle overlaps with existing ones
                overlap = False
                for px, py, pr in zip(positions, radii, radii):
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < r + pr + 1e-10:
                        overlap = True
                        break
                
                if not overlap:
                    positions.append(x)
                    radii.append(r)
                    placed = True
                    break
            
            if not placed:
                # Fallback: place at center with tiny radius
                positions.append(0.5)
                radii.append(0.01)
        
        return np.array(positions), np.array(radii)
    
    def total_radius(positions, radii):
        return np.sum(radii)
    
    def check_valid(positions, radii):
        """Check if placement is valid (no overlaps, all in bounds)."""
        for i in range(len(positions)):
            x, y, r = positions[i], radii[i], radii[i]
            if x < r - 1e-9 or x > 1 - r + 1e-9:
                return False
            if y < r - 1e-9 or y > 1 - r + 1e-9:
                return False
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                              (positions[i, 1] - positions[j, 1])**2)
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True
    
    def energy(positions, radii):
        """Compute energy (penalty for violations). Lower is better."""
        penalty = 0.0
        
        # Boundary violations
        for i in range(len(positions)):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            if x < r:
                penalty += (r - x) ** 2 * 1000
            if x > 1 - r:
                penalty += (x - (1 - r)) ** 2 * 1000
            if y < r:
                penalty += (r - y) ** 2 * 1000
            if y > 1 - r:
                penalty += (y - (1 - r)) ** 2 * 1000
        
        # Overlap violations
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                              (positions[i, 1] - positions[j, 1])**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        return penalty
    
    # Initialize
    pos_x, pos_y = [], []
    radii = []
    
    # Better initialization: use a structured approach
    # Divide square into regions and place circles
    n_per_side = 5
    cell_size = 1.0 / n_per_side
    
    for i in range(n_per_side):
        for j in range(n_per_side):
            if len(radii) >= N:
                break
            cx = (i + 0.5) * cell_size
            cy = (j + 0.5) * cell_size
            # Add some randomness
            cx += np.random.uniform(-0.1 * cell_size, 0.1 * cell_size)
            cy += np.random.uniform(-0.1 * cell_size, 0.1 * cell_size)
            r = 0.5 * cell_size * np.random.uniform(0.7, 0.9)
            
            # Clamp to valid range
            cx = np.clip(cx, r, 1 - r)
            cy = np.clip(cy, r, 1 - r)
            
            pos_x.append(cx)
            pos_y.append(cy)
            radii.append(r)
    
    # Ensure we have exactly N circles
    while len(radii) < N:
        r = 0.02
        x = np.random.uniform(r, 1 - r)
        y = np.random.uniform(r, 1 - r)
        pos_x.append(x)
        pos_y.append(y)
        radii.append(r)
    
    pos_x = np.array(pos_x[:N])
    pos_y = np.array(pos_y[:N])
    radii = np.array(radii[:N])
    
    positions = np.column_stack([pos_x, pos_y])
    
    # Simulated annealing
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    max_iters = 50000
    current_energy = energy(positions, radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_sum = total_radius(positions, radii)
    
    for it in range(max_iters):
        if T < T_min:
            break
        
        # Choose a move
        move_type = np.random.random()
        idx = np.random.randint(0, N)
        
        if move_type < 0.6:
            # Move a circle
            new_positions = positions.copy()
            dx = np.random.normal(0, 0.02)
            dy = np.random.normal(0, 0.02)
            new_positions[idx, 0] = np.clip(positions[idx, 0] + dx, radii[idx], 1 - radii[idx])
            new_positions[idx, 1] = np.clip(positions[idx, 1] + dy, radii[idx], 1 - radii[idx])
            
            new_energy = energy(new_positions, radii)
            delta = new_energy - current_energy
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                positions = new_positions
                current_energy = new_energy
                
                if total_radius(positions, radii) > best_sum:
                    best_positions = positions.copy()
                    best_radii = radii.copy()
                    best_sum = total_radius(positions, radii)
                    
        elif move_type < 0.85:
            # Try to grow a circle
            new_radii = radii.copy()
            growth = np.random.uniform(0, 0.01)
            new_radii[idx] = radii[idx] + growth
            
            # Check if still in bounds
            if new_radii[idx] > 0.5:
                new_radii[idx] = 0.5
            
            # Adjust position to stay in bounds
            new_positions = positions.copy()
            new_positions[idx, 0] = np.clip(positions[idx, 0], new_radii[idx], 1 - new_radii[idx])
            new_positions[idx, 1] = np.clip(positions[idx, 1], new_radii[idx], 1 - new_radii[idx])
            
            new_energy = energy(new_positions, new_radii)
            delta = new_energy - current_energy
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                positions = new_positions
                radii = new_radii
                current_energy = new_energy
                
                if total_radius(positions, radii) > best_sum:
                    best_positions = positions.copy()
                    best_radii = radii.copy()
                    best_sum = total_radius(positions, radii)
        else:
            # Shrink a circle (to allow others to grow)
            new_radii = radii.copy()
            shrink = np.random.uniform(0, 0.005)
            new_radii[idx] = max(0.01, radii[idx] - shrink)
            
            new_energy = energy(positions, new_radii)
            delta = new_energy - current_energy
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                radii = new_radii
                current_energy = new_energy
                
                if total_radius(positions, radii) > best_sum:
                    best_positions = positions.copy()
                    best_radii = radii.copy()
                    best_sum = total_radius(positions, radii)
        
        T *= cooling_rate
    
    # Use best solution found
    positions = best_positions
    radii = best_radii
    
    # Final verification and cleanup
    result = np.column_stack([positions, radii])
    
    # Ensure validity with small epsilon adjustment
    for i in range(N):
        x, y, r = result[i]
        # Clamp to valid range
        r = max(0.001, min(r, 0.5))
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        result[i] = [x, y, r]
    
    return result
# EVOLVE-BLOCK-END
