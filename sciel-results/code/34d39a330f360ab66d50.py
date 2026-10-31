import numpy as np
import random

N = 21

def construct():
    best_score = -1
    best_result = None
    
    # Try a few aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for w_ratio in aspect_ratios:
        W = w_ratio
        H = 2.0 - W
        
        # Initialize with a grid
        cols = 4
        rows = 5
        x_init = []
        y_init = []
        for i in range(cols):
            for j in range(rows):
                if len(x_init) >= N:
                    break
                x_init.append((i + 0.5) * W / cols)
                y_init.append((j + 0.5) * H / rows)
        
        if len(x_init) < N:
            x_init.append(W/2)
            y_init.append(H/2)
            
        x = np.array(x_init[:N])
        y = np.array(y_init[:N])
        
        def calc_radii(x, y, W, H):
            r = np.zeros(N)
            # Vectorized distance calculation
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            d = np.sqrt(dx**2 + dy**2)
            
            # Wall constraints
            r_wall = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
            
            # Neighbor constraints: min over j != i of d[i,j]/2
            np.fill_diagonal(d, np.inf)
            r_neigh = np.min(d / 2.0, axis=1)
            
            r = np.minimum(r_wall, r_neigh)
            return r
        
        # Simulated Annealing
        T = 0.1
        T_min = 1e-6
        alpha = 0.9995
        iterations = 20000
        
        x_curr = x.copy()
        y_curr = y.copy()
        r_curr = calc_radii(x_curr, y_curr, W, H)
        score_curr = np.sum(r_curr)
        
        best_local_score = score_curr
        best_local_x = x_curr.copy()
        best_local_y = y_curr.copy()
        
        for it in range(iterations):
            if T < T_min:
                break
                
            k = random.randint(0, N-1)
            old_x = x_curr[k]
            old_y = y_curr[k]
            
            # Perturb
            step = 0.05 * (0.5 + 0.5 * (T / 0.1))
            x_curr[k] += random.uniform(-step, step)
            y_curr[k] += random.uniform(-step, step)
            
            # Clamp
            x_curr[k] = max(0.0, min(W, x_curr[k]))
            y_curr[k] = max(0.0, min(H, y_curr[k]))
            
            # Recalculate radii
            r_new = calc_radii(x_curr, y_curr, W, H)
            score_new = np.sum(r_new)
            
            delta = score_new - score_curr
            if delta > 0:
                score_curr = score_new
                if score_new > best_local_score:
                    best_local_score = score_new
                    best_local_x = x_curr.copy()
                    best_local_y = y_curr.copy()
            else:
                if random.random() < np.exp(delta / T):
                    score_curr = score_new
                else:
                    x_curr[k] = old_x
                    y_curr[k] = old_y
            
            T *= alpha
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_result = np.column_stack((best_local_x, best_local_y, calc_radii(best_local_x, best_local_y, W, H)))
            
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
