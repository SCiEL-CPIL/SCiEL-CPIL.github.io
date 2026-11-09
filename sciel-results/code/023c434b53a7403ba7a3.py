# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in a unit square.
    Uses Simulated Annealing with a smooth penalty function for overlaps and bounds.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 600s limit
    
    def calculate_energy(params):
        """
        Calculate energy (cost to minimize).
        Energy = -sum(sides) + penalty_bounds + penalty_overlaps
        """
        sides = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        # Objective: Maximize sum of sides -> Minimize -sum(sides)
        energy = -np.sum(sides)
        
        # Penalty for bounds
        left = cx - sides / 2.0
        right = cx + sides / 2.0
        bottom = cy - sides / 2.0
        top = cy + sides / 2.0
        
        v_left = np.maximum(0.0, -left)
        v_right = np.maximum(0.0, right - 1.0)
        v_bottom = np.maximum(0.0, -bottom)
        v_top = np.maximum(0.0, top - 1.0)
        
        # High penalty for bounds to ensure validity
        bound_penalty = 100.0 * np.sum(v_left**2 + v_right**2 + v_bottom**2 + v_top**2)
        
        # Penalty for overlaps
        overlap_penalty = 0.0
        k_overlap = 100.0  # High scaling factor
        
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                
                if dx < t and dy < t:
                    pen = (t - dx) * (t - dy)
                    overlap_penalty += k_overlap * pen
        
        energy += bound_penalty + overlap_penalty
        return energy

    def init_config(seed):
        rng = np.random.default_rng(seed)
        params = np.zeros(3 * n)
        
        k = 3
        side = 1.0 / k
        for i in range(9):
            col = i % k
            row = i // k
            params[i] = side * 0.95
            params[n + i] = (col + 0.5) * side
            params[2*n + i] = (row + 0.5) * side
            
        params[9] = rng.uniform(0.1, 0.2)
        params[n + 9] = rng.uniform(0.1, 0.9)
        params[2*n + 9] = rng.uniform(0.1, 0.9)
        
        return params

    # Find best initial config
    best_init = None
    best_init_energy = np.inf
    
    for seed in range(20):
        p = init_config(seed)
        e = calculate_energy(p)
        if e < best_init_energy:
            best_init_energy = e
            best_init = p
            
    params = best_init.copy()
    current_energy = best_init_energy
    best_params = params.copy()
    best_energy = current_energy
    
    T = 1.0
    T_min = 1e-5
    cooling_rate = 0.99999
    
    sigma_s = 0.05
    sigma_p = 0.05
    
    while time.time() < deadline:
        i = np.random.randint(n)
        
        old_s = params[i]
        old_cx = params[n + i]
        old_cy = params[2*n + i]
        
        r = np.random.random()
        
        if r < 0.3:
            params[i] += np.random.normal(0, sigma_s)
            if params[i] < 1e-4: params[i] = 1e-4
            if params[i] > 0.6: params[i] = 0.6
        elif r < 0.7:
            params[n + i] += np.random.normal(0, sigma_p)
            params[2*n + i] += np.random.normal(0, sigma_p)
            if params[n + i] < 0: params[n + i] = 0
            if params[n + i] > 1: params[n + i] = 1
            if params[2*n + i] < 0: params[2*n + i] = 0
            if params[2*n + i] > 1: params[2*n + i] = 1
        else:
            params[i] += np.random.normal(0, sigma_s)
            params[n + i] += np.random.normal(0, sigma_p)
            params[2*n + i] += np.random.normal(0, sigma_p)
            
            if params[i] < 1e-4: params[i] = 1e-4
            if params[i] > 0.6: params[i] = 0.6
            if params[n + i] < 0: params[n + i] = 0
            if params[n + i] > 1: params[n + i] = 1
            if params[2*n + i] < 0: params[2*n + i] = 0
            if params[2*n + i] > 1: params[2*n + i] = 1
            
        new_energy = calculate_energy(params)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()
        else:
            params[i] = old_s
            params[n + i] = old_cx
            params[2*n + i] = old_cy
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        if T < 0.01:
            sigma_s = 0.01
            sigma_p = 0.01
        if T < 0.001:
            sigma_s = 0.005
            sigma_p = 0.005

    sides = best_params[:n].astype(np.float64)
    centers = best_params[n:].reshape(n, 2).astype(np.float64)
    
    # Validation and Repair
    tol = 1e-7
    valid = True
    for i in range(n):
        if sides[i] <= 0:
            valid = False
            break
        cx, cy = centers[i]
        if cx - sides[i]/2 < -tol or cx + sides[i]/2 > 1 + tol:
            valid = False
            break
        if cy - sides[i]/2 < -tol or cy + sides[i]/2 > 1 + tol:
            valid = False
            break
            
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Repair: Shrink sides and re-center if needed
        # This is a fallback, might lower score
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Ensure inside
            max_s = 2 * min(cx, 1-cx, cy, 1-cy)
            if max_s < 0:
                centers[i] = [0.5, 0.5]
                max_s = 1.0
            if s > max_s:
                sides[i] = max_s * 0.999
            # Ensure positive
            if sides[i] <= 0:
                sides[i] = 1e-4
                centers[i] = [0.5, 0.5]
                
        # Check overlaps again and shrink if necessary
        # Simple iterative shrink
        for _ in range(10):
            overlap_found = False
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2
                    if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                        # Shrink both
                        sides[i] *= 0.99
                        sides[j] *= 0.99
                        overlap_found = True
            if not overlap_found:
                break
                
        # Final bounds check
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < 0 or cx + s/2 > 1 or cy - s/2 < 0 or cy + s/2 > 1:
                max_s = 2 * min(cx, 1-cx, cy, 1-cy)
                if max_s > 0:
                    sides[i] = max_s * 0.999
                else:
                    sides[i] = 1e-4
                    centers[i] = [0.5, 0.5]

    return centers, sides
# EVOLVE-BLOCK-END
