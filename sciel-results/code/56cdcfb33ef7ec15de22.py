import numpy as np
import time

N = 26

def construct():
    def initial_solution():
        side = int(np.ceil(np.sqrt(N)))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.5
                xs[count] = x
                ys[count] = y
                rs[count] = r
                count += 1
                if count == N:
                    break
        
        xs += np.random.uniform(-0.01, 0.01, N)
        ys += np.random.uniform(-0.01, 0.01, N)
        
        xs = np.clip(xs, rs, 1 - rs)
        ys = np.clip(ys, rs, 1 - rs)
        
        return np.column_stack((xs, ys, rs))

    def is_valid(x, y, r, tol=1e-8):
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r*sum_r - tol:
                    return False
        return True

    def repair(x, y, r, max_iters=20, step_size=0.02):
        x_new = x.copy()
        y_new = y.copy()
        
        for _ in range(max_iters):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Boundary forces
            for i in range(N):
                if x_new[i] < r[i]:
                    fx[i] += (r[i] - x_new[i]) * 10
                elif x_new[i] > 1 - r[i]:
                    fx[i] -= (x_new[i] - (1 - r[i])) * 10
                if y_new[i] < r[i]:
                    fy[i] += (r[i] - y_new[i]) * 10
                elif y_new[i] > 1 - r[i]:
                    fy[i] -= (y_new[i] - (1 - r[i])) * 10
            
            # Overlap forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_new[i] - x_new[j]
                    dy = y_new[i] - y_new[j]
                    dist_sq = dx*dx + dy*dy
                    sum_r = r[i] + r[j]
                    
                    if dist_sq < sum_r*sum_r:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-6:
                            angle = np.random.uniform(0, 2*np.pi)
                            push = (sum_r - dist) * 0.5
                            fx[i] += np.cos(angle) * push * 10
                            fy[i] += np.sin(angle) * push * 10
                            fx[j] -= np.cos(angle) * push * 10
                            fy[j] -= np.sin(angle) * push * 10
                        else:
                            push = (sum_r - dist) * 0.5
                            fx[i] += (dx / dist) * push * 10
                            fy[i] += (dy / dist) * push * 10
                            fx[j] -= (dx / dist) * push * 10
                            fy[j] -= (dy / dist) * push * 10
            
            x_new += fx * step_size
            y_new += fy * step_size
            
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            
            if is_valid(x_new, y_new, r):
                return x_new, y_new
        
        return None

    start_time = time.time()
    time_limit = 110
    
    current = initial_solution()
    x, y, r = current[:, 0], current[:, 1], current[:, 2]
    
    best_score = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    delta_r = 0.01
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        indices = np.random.permutation(N)
        
        for i in indices:
            r_new = r.copy()
            r_new[i] += delta_r
            
            if r_new[i] > 0.5:
                r_new[i] = 0.5
            
            # Quick check: if new radius makes it impossible to fit in square
            if r_new[i] > 0.5:
                continue
                
            # Try repair
            result = repair(x, y, r_new)
            
            if result is not None:
                x_new, y_new = result
                new_score = np.sum(r_new)
                
                if new_score > best_score:
                    x, y, r = x_new, y_new, r_new
                    best_score = new_score
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                    improved = True
                    no_improve_count = 0
                    # After a successful move, the landscape changes, so we might want to restart the index loop or just continue
                    # Continuing is fine, but the other circles might now be able to grow too.
        
        if not improved:
            no_improve_count += 1
            delta_r *= 0.9
            if delta_r < 1e-5:
                break
        else:
            no_improve_count = 0
    
    return np.column_stack((best_x, best_y, best_r))
