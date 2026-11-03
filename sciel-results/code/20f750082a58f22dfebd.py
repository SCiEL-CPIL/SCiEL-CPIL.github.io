import numpy as np
import time

N = 26

def construct():
    """
    Pack 26 circles in a unit square maximizing sum of radii.
    Uses Simulated Annealing.
    """
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for overhead
    
    # Initial state: Grid of small circles
    def random_valid_init():
        # Place circles in a grid to ensure initial validity
        # 6x5 grid (30 cells), pick 26
        xs = np.linspace(0.1, 0.9, 6)
        ys = np.linspace(0.1, 0.9, 5)
        all_pos = []
        for i in range(6):
            for j in range(5):
                all_pos.append((xs[i], ys[j]))
        # Shuffle and take 26
        np.random.shuffle(all_pos)
        init_x = [p[0] for p in all_pos[:26]]
        init_y = [p[1] for p in all_pos[:26]]
        init_r = [0.05] * 26
        return np.array(init_x), np.array(init_y), np.array(init_r)

    x, y, r = random_valid_init()
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = np.sum(r)
    
    # SA Parameters
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iterations = 500000
    
    def get_penalty(x, y, r):
        # Wall penalty
        # dist to wall
        dx = np.minimum(x, 1.0 - x)
        dy = np.minimum(y, 1.0 - y)
        d_wall = np.minimum(dx, dy)
        wall_viol = np.maximum(r - d_wall, 0.0)
        pen = np.sum(wall_viol**2)
        
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(r_sum - d, 0.0)
        # Ignore diagonal
        np.fill_diagonal(overlap, 0.0)
        pen += np.sum(overlap**2)
        return pen

    def get_score(x, y, r):
        return -np.sum(r) + 100.0 * get_penalty(x, y, r)

    current_score = get_score(x, y, r)
    
    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Random move
        idx = np.random.randint(N)
        move_type = np.random.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.5:
            # Move position
            step = T * 10
            new_x[idx] += np.random.normal(0, step)
            new_y[idx] += np.random.normal(0, step)
            new_x[idx] = np.clip(new_x[idx], 0, 1)
            new_y[idx] = np.clip(new_y[idx], 0, 1)
        elif move_type < 0.9:
            # Change radius
            step = T * 5
            new_r[idx] += np.random.normal(0, step)
            new_r[idx] = np.clip(new_r[idx], 0.001, 0.5)
        else:
            # Move and change radius
            step_pos = T * 10
            step_r = T * 5
            new_x[idx] += np.random.normal(0, step_pos)
            new_y[idx] += np.random.normal(0, step_pos)
            new_x[idx] = np.clip(new_x[idx], 0, 1)
            new_y[idx] = np.clip(new_y[idx], 0, 1)
            new_r[idx] += np.random.normal(0, step_r)
            new_r[idx] = np.clip(new_r[idx], 0.001, 0.5)
            
        new_score = get_score(new_x, new_y, new_r)
        delta = new_score - current_score
        
        # Acceptance probability
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x, y, r = new_x, new_y, new_r
            current_score = new_score
            
            # Track best valid solution
            pen = get_penalty(x, y, r)
            if pen < 1e-8:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final cleanup: ensure the returned solution is strictly valid
    # If best solution had any penalty, it might be slightly invalid due to floating point
    # We can shrink radii by a tiny epsilon to be safe
    eps = 1e-9
    best_r = best_r - eps
    # Ensure r > 0
    best_r = np.maximum(best_r, 1e-6)
    
    result = np.column_stack((best_x, best_y, best_r))
    return result
