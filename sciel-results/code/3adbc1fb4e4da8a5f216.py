import numpy as np
import time

N = 26

def construct():
    # Simulated Annealing for Circle Packing
    # Variables: x, y, r for N circles
    # Objective: Maximize sum(r)
    # Constraints: 
    #   1. Circles inside [0,1]^2: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    #   2. No overlap: dist(i,j) >= r_i + r_j
    
    def initial_solution():
        # Grid initialization with random perturbation
        side = int(np.ceil(np.sqrt(N)))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        # Fill grid positions
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.8 # Slightly smaller to allow movement
                xs[count] = x
                ys[count] = y
                rs[count] = r
                count += 1
                if count == N:
                    break
            
        # Add some randomness to break symmetry
        xs += np.random.uniform(-0.05, 0.05, N)
        ys += np.random.uniform(-0.05, 0.05, N)
        rs += np.random.uniform(-0.02, 0.02, N)
        
        # Clip to valid bounds
        xs = np.clip(xs, 0, 1)
        ys = np.clip(ys, 0, 1)
        rs = np.clip(rs, 0.01, 0.5)
        
        # Enforce constraints by shrinking radii if overlapping or out of bounds
        # This ensures the initial state is valid
        for _ in range(100):
            valid = True
            for i in range(N):
                # Boundary check
                if xs[i] < rs[i] or xs[i] > 1 - rs[i] or ys[i] < rs[i] or ys[i] > 1 - rs[i]:
                    valid = False
                    # Shrink radius to fit boundary
                    max_r = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
                    if max_r < 0.01:
                        # Move center to center of square and set small radius
                        xs[i] = 0.5
                        ys[i] = 0.5
                        rs[i] = 0.01
                    else:
                        rs[i] = max_r * 0.99
                # Overlap check
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < rs[i] + rs[j]:
                        valid = False
                        # Shrink larger circle or both
                        # Simple strategy: shrink both slightly
                        scale = dist / (rs[i] + rs[j]) * 0.99
                        rs[i] *= scale
                        rs[j] *= scale
            if valid:
                break
                
        return np.column_stack((xs, ys, rs))

    def calculate_score(circles):
        # Returns sum of radii if valid, else -inf
        # Also returns a penalty for invalidity to guide search if needed, 
        # but for SA, we usually stick to valid moves or accept invalid with high penalty.
        # Let's use a penalty method: Score = Sum(r) - Penalty * Violation
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        sum_r = np.sum(r)
        penalty = 0.0
        
        # Boundary violations
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])
            if x[i] > 1 - r[i]:
                penalty += (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                penalty += (r[i] - y[i])
            if y[i] > 1 - r[i]:
                penalty += (y[i] - (1 - r[i]))
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    penalty += (r[i] + r[j] - dist)
                    
        # We want to maximize sum_r - large_penalty * penalty
        # If penalty > 0, the solution is invalid.
        # In SA, we define the "energy" as -Score. Lower energy is better.
        # Score = sum_r - 100 * penalty
        # Energy = -sum_r + 100 * penalty
        return sum_r - 100 * penalty

    def generate_neighbor(circles):
        # Pick a random circle and perturb it
        i = np.random.randint(N)
        new_circles = circles.copy()
        
        # Perturb position and radius
        # Step size can be adaptive, but fixed small steps work well
        new_circles[i, 0] += np.random.uniform(-0.05, 0.05)
        new_circles[i, 1] += np.random.uniform(-0.05, 0.05)
        new_circles[i, 2] += np.random.uniform(-0.01, 0.01)
        
        # Clip to hard bounds to prevent huge penalties from extreme values
        new_circles[i, 0] = np.clip(new_circles[i, 0], 0, 1)
        new_circles[i, 1] = np.clip(new_circles[i, 1], 0, 1)
        new_circles[i, 2] = np.clip(new_circles[i, 2], 0.01, 0.5)
        
        return new_circles

    def is_valid(circles):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        for i in range(N):
            if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6:
                return False
            if y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                return False
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r*sum_r - 1e-6:
                    return False
        return True

    # Main SA Loop
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Initial solution
    current = initial_solution()
    current_score = calculate_score(current)
    
    best = current.copy()
    best_score = current_score
    
    # SA Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.995
    
    # Track best valid solution
    if is_valid(best):
        best_valid = best.copy()
        best_valid_score = best_score
    else:
        best_valid = None
        best_valid_score = -np.inf

    while time.time() - start_time < time_limit:
        neighbor = generate_neighbor(current)
        neighbor_score = calculate_score(neighbor)
        
        delta = neighbor_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current = neighbor
            current_score = neighbor_score
            
            # Update best if valid
            if is_valid(current):
                if current_score > best_valid_score:
                    best_valid_score = current_score
                    best_valid = current.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Return best valid solution found
    if best_valid is not None:
        return best_valid
    else:
        # Fallback: Return initial grid if no valid solution found (should not happen)
        return initial_solution()
