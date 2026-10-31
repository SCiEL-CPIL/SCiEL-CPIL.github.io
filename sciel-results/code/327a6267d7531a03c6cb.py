import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_config = None
    
    # Initial simple grid as a fallback/baseline
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    base_config = np.array(cells[:N])
    best_score = np.sum(base_config[:, 2])
    best_config = base_config.copy()
    
    def get_score(config):
        x, y, r = config[:, 0], config[:, 1], config[:, 2]
        # Check boundaries
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9) or np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return -np.inf
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)

    def repair_config(config):
        x = config[:, 0].copy()
        y = config[:, 1].copy()
        r = config[:, 2].copy()
        
        # Clamp to boundaries
        for i in range(N):
            r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = max(r[i], 1e-6)
        
        # Iteratively resolve overlaps
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 1.0:
                break
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Shrink both radii by half the overlap
                        shrink = overlap / 2.0
                        r[i] -= shrink
                        r[j] -= shrink
                        
                        # Re-clamp to boundaries
                        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
                        r[j] = min(r[j], x[j], 1 - x[j], y[j], 1 - y[j])
                        
                        r[i] = max(r[i], 1e-6)
                        r[j] = max(r[j], 1e-6)
                        changed = True
            
            if not changed:
                break
                
        return np.column_stack((x, y, r))

    def run_sa(x0, y0, r0, steps=5000):
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        T = 0.1
        alpha = 0.998
        step_size = 0.05
        
        def energy(x, y, r):
            e = -np.sum(r)
            
            for i in range(N):
                if x[i] < r[i]:
                    e += 100 * (r[i] - x[i])**2
                if x[i] > 1 - r[i]:
                    e += 100 * (x[i] - (1 - r[i]))**2
                if y[i] < r[i]:
                    e += 100 * (r[i] - y[i])**2
                if y[i] > 1 - r[i]:
                    e += 100 * (y[i] - (1 - r[i]))**2
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        e += 100 * overlap * overlap
            
            return e

        current_energy = energy(x, y, r)
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
                
            i = np.random.randint(N)
            
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dr = np.random.normal(0, step_size * 0.5)
            
            new_x = x[i] + dx
            new_y = y[i] + dy
            new_r = r[i] + dr
            
            if new_r < 1e-4 or new_r > 0.5:
                continue
            if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
                continue

            old_x, old_y, old_r = x[i], y[i], r[i]
            x[i], y[i], r[i] = new_x, new_y, new_r
            new_energy = energy(x, y, r)
            
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current_energy = new_energy
            else:
                x[i], y[i], r[i] = old_x, old_y, old_r
            
            T *= alpha
            
        return np.column_stack((x, y, r))

    def polish(config):
        x0 = config[:, 0]
        y0 = config[:, 1]
        r0 = config[:, 2]
        
        vars_ = np.concatenate([x0, y0, r0])
        
        def objective(v):
            r = v[2::3]
            return -np.sum(r)
        
        constraints = []
        
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3] - v[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i*3] - v[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3+1] - v[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i*3+1] - v[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3+2] - 1e-6})
            
        for i in range(N):
            for j in range(i + 1, N):
                def overlap_fun(v, i=i, j=j):
                    xi = v[i*3]
                    yi = v[i*3+1]
                    ri = v[i*3+2]
                    xj = v[j*3]
                    yj = v[j*3+1]
                    rj = v[j*3+2]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                constraints.append({'type': 'ineq', 'fun': overlap_fun})
                
        bounds = []
        for i in range(N):
            bounds.append((1e-6, 1.0))
            bounds.append((1e-6, 1.0))
            bounds.append((1e-6, 0.5))
            
        try:
            res = minimize(objective, vars_, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 50, 'ftol': 1e-9})
            if res.success:
                v_opt = res.x
                x_opt = v_opt[0::3]
                y_opt = v_opt[1::3]
                r_opt = v_opt[2::3]
                return np.column_stack((x_opt, y_opt, r_opt))
        except:
            pass
        return config

    while time.time() < deadline - 5.0:
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.1, N)
        
        config = np.column_stack((x, y, r))
        
        sa_config = run_sa(x, y, r, steps=5000)
        
        polished_config = polish(sa_config)
        
        valid_config = repair_config(polished_config)
        
        score = get_score(valid_config)
        
        if score > best_score:
            best_score = score
            best_config = valid_config.copy()
            
        if time.time() > deadline - 10.0:
            break

    return best_config
