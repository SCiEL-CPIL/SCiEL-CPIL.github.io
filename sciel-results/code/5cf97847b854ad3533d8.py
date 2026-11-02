"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin (5s buffer before 120s kill)
    
    best_value = -np.inf
    best_params = None  # [w, x1,y1,r1, ..., x21,y21,r21]
    
    def compute_penalty(params):
        """Compute penalty for constraint violations and negative sum of radii."""
        w_opt = params[0]
        h_opt = 2.0 - w_opt
        if w_opt <= 0.05 or h_opt <= 0.05:
            return 1e10
        x = params[1:]
        penalty = 0.0
        # Bounds
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-8:
                penalty += 1000.0
            if xi - ri < -1e-5:
                penalty += 500.0 * (xi - ri + 1e-5)**2
            if xi + ri > w_opt + 1e-5:
                penalty += 500.0 * (xi + ri - w_opt - 1e-5)**2
            if yi - ri < -1e-5:
                penalty += 500.0 * (yi - ri + 1e-5)**2
            if yi + ri > h_opt + 1e-5:
                penalty += 500.0 * (yi + ri - h_opt - 1e-5)**2
        # Disjointness
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq + 1e-12)
                    penalty += 500.0 * (min_dist - dist)**2
        return penalty - np.sum(x[2::3])
    
    def is_valid(params, tol=1e-6):
        """Check if params represent a valid packing."""
        w_opt = params[0]
        h_opt = 2.0 - w_opt
        if w_opt <= 0.05 or h_opt <= 0.05:
            return False
        x = params[1:]
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-8:
                return False
            if xi - ri < -tol or xi + ri > w_opt + tol or yi - ri < -tol or yi + ri > h_opt + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                if dist_sq < (ri + rj - tol)**2:
                    return False
        return True
    
    def update_best(params):
        nonlocal best_value, best_params
        if time.time() - start_time > deadline:
            return
        if is_valid(params):
            val = np.sum(params[1::3])
            if val > best_value:
                best_value = val
                best_params = params.copy()
    
    def make_grid_init(rows, cols, w, h, r_base=0.09, jitter=0.0):
        """Create grid-based initial configuration."""
        x0 = np.zeros(3*N)
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x0[3*count] = (i + 0.5) * w / cols + np.random.randn()*jitter
                x0[3*count+1] = (j + 0.5) * h / rows + np.random.randn()*jitter
                x0[3*count+2] = r_base + np.random.randn()*jitter*0.5
                count += 1
        return np.concatenate(([w], x0))
    
    def simulated_annealing(params0, max_iter=5000, T_start=0.1, T_end=0.001):
        """Simulated annealing on the joint parameter space."""
        params = params0.copy()
        current_obj = compute_penalty(params)
        best_local = params.copy()
        best_local_obj = current_obj
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iter))
        
        for it in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            # Propose perturbation
            new_params = params.copy()
            # Perturb aspect ratio
            new_params[0] += np.random.randn() * 0.05
            new_params[0] = np.clip(new_params[0], 0.1, 1.9)
            # Perturb circle parameters
            for i in range(N):
                new_params[1+3*i] += np.random.randn() * 0.03   # x
                new_params[2+3*i] += np.random.randn() * 0.03   # y
                new_params[3+3*i] += np.random.randn() * 0.02   # r
                new_params[3+3*i] = max(new_params[3+3*i], 0.01)
            
            new_obj = compute_penalty(new_params)
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                params = new_params
                current_obj = new_obj
                if current_obj < best_local_obj:
                    best_local = params.copy()
                    best_local_obj = current_obj
            
            T *= cooling
        
        return best_local
    
    def polish(params, maxiter=300):
        """Final high-precision polish using Nelder-Mead."""
        try:
            result = minimize(compute_penalty, params, method='Nelder-Mead',
                             options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'adaptive': True})
            return result.x
        except Exception:
            return params
    
    # Multiple diverse initializations
    inits = []
    # 7x3 grid
    inits.append(make_grid_init(3, 7, 1.4, 0.6, 0.09))
    # 5x5 grid (21 circles)
    inits.append(make_grid_init(5, 5, 1.0, 1.0, 0.09))
    # 6x4 grid
    inits.append(make_grid_init(4, 6, 1.2, 0.8, 0.09))
    # 3x7 with different aspect
    inits.append(make_grid_init(3, 7, 1.5, 0.5, 0.085))
    # 7x3 with different aspect
    inits.append(make_grid_init(3, 7, 1.3, 0.7, 0.095))
    # 4x5 grid
    inits.append(make_grid_init(5, 4, 0.8, 1.2, 0.09))
    # Random perturbation
    inits.append(make_grid_init(3, 7, 1.4, 0.6, 0.09, jitter=0.02))
    inits.append(make_grid_init(3, 7, 1.35, 0.65, 0.09, jitter=0.015))
    
    # Phase 1: Simulated annealing from each initialization
    for init in inits:
        if time.time() - start_time > deadline:
            break
        sa_result = simulated_annealing(init, max_iter=3000, T_start=0.08, T_end=0.002)
        update_best(sa_result)
    
    # Phase 2: Polish the best result found
    if best_params is not None:
        if time.time() - start_time < deadline - 10:
            polished = polish(best_params, maxiter=500)
            update_best(polished)
        
        # Try polishing from a few other good candidates
        for init in inits[:4]:
            if time.time() - start_time > deadline - 5:
                break
            sa_result = simulated_annealing(init, max_iter=2000, T_start=0.05, T_end=0.001)
            update_best(sa_result)
            if is_valid(sa_result):
                val = np.sum(sa_result[1::3])
                if val > best_value * 0.98:
                    polished = polish(sa_result, maxiter=300)
                    update_best(polished)
    
    # Fallback: use grid if no valid solution found
    if best_params is None:
        w, h = 1.4, 0.6
        r = 0.099
        best_params = np.zeros(3*N + 1)
        best_params[0] = w
        for i in range(7):
            for j in range(3):
                idx = i*3 + j
                best_params[1+3*idx] = (i + 0.5) * w / 7
                best_params[2+3*idx] = (j + 0.5) * h / 3
                best_params[3+3*idx] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_params[1+3*i]
        result[i, 1] = best_params[2+3*i]
        result[i, 2] = best_params[3+3*i]
    
    return result
# EVOLVE-BLOCK-END
