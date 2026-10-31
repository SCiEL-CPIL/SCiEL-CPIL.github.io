import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_value = 0
    best_result = None
    
    def is_valid(x, y, r, w):
        h = 2.0 - w
        for i in range(N):
            if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j]:
                    return False
        return True

    def calc_energy(x, y, r, w):
        h = 2.0 - w
        # Boundary penalty
        bound_pen = 0.0
        # Vectorized boundary check is hard with simple sums, loop is fine for N=21
        for i in range(N):
            if x[i] < r[i]: bound_pen += (r[i] - x[i])**2
            elif x[i] > w - r[i]: bound_pen += (x[i] - (w - r[i]))**2
            if y[i] < r[i]: bound_pen += (r[i] - y[i])**2
            elif y[i] > h - r[i]: bound_pen += (y[i] - (h - r[i]))**2
            
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        sum_r = r[:, None] + r[None, :]
        overlap = np.maximum(0, sum_r - d)
        overlap_pen = np.sum(overlap**2)
        
        obj = -np.sum(r)
        E = obj + 50.0 * overlap_pen + 50.0 * bound_pen
        return E

    def run_sa(seed, time_limit):
        np.random.seed(seed)
        
        # Initialization: Grid
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    x[count] = (i + 0.5) / side
                    y[count] = (j + 0.5) / side
                    r[count] = 0.5 / side
                    count += 1
        
        w = 1.0
        h = 1.0
        
        current_E = calc_energy(x, y, r, w)
        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
        best_E = current_E
        
        T = 0.5
        alpha = 0.999
        run_start = time.time()
        
        while time.time() < run_start + time_limit:
            T *= alpha
            if T < 0.001:
                break
                
            # Move
            if np.random.random() < 0.05:
                # Change w
                dw = np.random.normal(0, 0.05)
                new_w = w + dw
                if new_w < 0.5 or new_w > 1.5:
                    continue
                new_x, new_y, new_r = x, y, r
                new_E = calc_energy(new_x, new_y, new_r, new_w)
            else:
                k = np.random.randint(N)
                new_x = x.copy()
                new_y = y.copy()
                new_r = r.copy()
                
                new_x[k] += np.random.normal(0, 0.05)
                new_y[k] += np.random.normal(0, 0.05)
                new_r[k] += np.random.normal(0, 0.005)
                
                if new_r[k] < 0.01:
                    new_r[k] = 0.01
                    
                new_E = calc_energy(new_x, new_y, new_r, w)
                
            delta_E = new_E - current_E
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                x, y, r, w = new_x, new_y, new_r, (new_w if 'new_w' in locals() else w)
                current_E = new_E
                
                if current_E < best_E:
                    best_E = current_E
                    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
        
        # Return best found in this run
        # Check validity
        if is_valid(best_x, best_y, best_r, best_w):
            return best_x, best_y, best_r, best_w
        else:
            # Try to fix by shrinking
            scale = 1.0
            for _ in range(50):
                if is_valid(best_x, best_y, best_r * scale, best_w):
                    return best_x, best_y, best_r * scale, best_w
                scale *= 0.95
            return None

    # Main loop
    run_id = 0
    while time.time() < deadline:
        # Allocate time for each run, e.g., 5 seconds
        time_left = deadline - time.time()
        if time_left < 2: # Not enough time for a full run
            break
        run_time = min(5.0, time_left - 1.0)
        
        res = run_sa(run_id, run_time)
        if res is not None:
            x, y, r, w = res
            val = np.sum(r)
            if val > best_value:
                best_value = val
                # Construct result array
                best_result = np.zeros((N, 3))
                best_result[:, 0] = x
                best_result[:, 1] = y
                best_result[:, 2] = r
                
        run_id += 1
        
    if best_result is None:
        # Fallback to seed-like result
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        
    return best_result
