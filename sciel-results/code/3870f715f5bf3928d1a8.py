import numpy as np
import time
from scipy.optimize import minimize

N = 32
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_solution = None
    
    # Helper to check time
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)

    # Penalty weight
    LAMBDA = 100.0

    def objective_and_grad(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Objective: -sum(r)
        f = -np.sum(r)
        grad = np.zeros(3*N)
        grad[2::3] -= 1.0
        
        # Boundary penalties
        # x - r >= 0 -> viol = r - x
        # 1 - x - r >= 0 -> viol = x + r - 1
        # y - r >= 0 -> viol = r - y
        # 1 - y - r >= 0 -> viol = y + r - 1
        
        viol_x_left = np.maximum(0, r - x)
        viol_x_right = np.maximum(0, x + r - 1.0)
        viol_y_bottom = np.maximum(0, r - y)
        viol_y_top = np.maximum(0, y + r - 1.0)
        
        # Sum of squared violations
        f += LAMBDA * (np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2))
        
        # Gradients for boundary penalties
        # d/dx (r-x)^2 = -2(r-x) if r>x else 0
        # d/dx (x+r-1)^2 = 2(x+r-1) if x+r>1 else 0
        # d/dr (r-x)^2 = 2(r-x) if r>x else 0
        # d/dr (x+r-1)^2 = 2(x+r-1) if x+r>1 else 0
        
        mask_xl = viol_x_left > 0
        mask_xr = viol_x_right > 0
        mask_yb = viol_y_bottom > 0
        mask_yt = viol_y_top > 0
        
        grad[0::3][mask_xl] += LAMBDA * (-2 * viol_x_left[mask_xl])
        grad[2::3][mask_xl] += LAMBDA * (2 * viol_x_left[mask_xl])
        
        grad[0::3][mask_xr] += LAMBDA * (2 * viol_x_right[mask_xr])
        grad[2::3][mask_xr] += LAMBDA * (2 * viol_x_right[mask_xr])
        
        grad[1::3][mask_yb] += LAMBDA * (-2 * viol_y_bottom[mask_yb])
        grad[2::3][mask_yb] += LAMBDA * (2 * viol_y_bottom[mask_yb])
        
        grad[1::3][mask_yt] += LAMBDA * (2 * viol_y_top[mask_yt])
        grad[2::3][mask_yt] += LAMBDA * (2 * viol_y_top[mask_yt])
        
        # Pairwise penalties
        # d_ij^2 - (r_i+r_j)^2 >= 0
        # viol = (r_i+r_j) - d_ij
        # Penalty = viol^2 if viol > 0 else 0
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                rad_sum = r[i] + r[j]
                
                viol = rad_sum - dist
                if viol > 0:
                    # f += LAMBDA * viol^2
                    f += LAMBDA * viol * viol
                    
                    # Gradient w.r.t x_i, y_i, r_i, x_j, y_j, r_j
                    # d/dx_i (r_i+r_j - sqrt((x_i-x_j)^2 + ...))^2
                    # = 2 * viol * (-1) * d/dx_i dist
                    # d/dx_i dist = (x_i - x_j) / dist
                    # So d/dx_i penalty = -2 * viol * (x_i - x_j) / dist
                    # d/dx_j penalty = -2 * viol * (x_j - x_i) / dist = 2 * viol * (x_i - x_j) / dist
                    # d/dr_i penalty = 2 * viol * 1
                    # d/dr_j penalty = 2 * viol * 1
                    
                    factor = 2 * LAMBDA * viol / dist
                    
                    grad[3*i]   += -factor * dx
                    grad[3*i+1] += -factor * dy
                    grad[3*i+2] += 2 * LAMBDA * viol
                    grad[3*j]   += factor * dx
                    grad[3*j+1] += factor * dy
                    grad[3*j+2] += 2 * LAMBDA * viol
        
        return f, grad

    def objective_only(z):
        return objective_and_grad(z)[0]

    def grad_only(z):
        return objective_and_grad(z)[1]

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds_int = []
    for _ in range(N):
        bounds_int.append((0, 1)) # x
        bounds_int.append((0, 1)) # y
        bounds_int.append((0, 0.5)) # r

    def run_optimizer(z0):
        res = minimize(
            objective_only, 
            z0, 
            jac=grad_only,
            method='L-BFGS-B', 
            bounds=bounds_int,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        return res

    # Initialization Strategies
    
    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        # 6x6 grid
        xs, ys = np.meshgrid(np.linspace(0.5/6, 1-0.5/6, 6), np.linspace(0.5/6, 1-0.5/6, 6))
        x = xs.flatten()
        y = ys.flatten()
        # Take first 32
        x = x[:N]
        y = y[:N]
        r = np.full(N, 0.08)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def init_random(seed):
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def init_hexagonal():
        # 4 rows of 8
        x = []
        y = []
        for i in range(4):
            offset = 0 if i % 2 == 0 else 0.0625
            for j in range(8):
                x.append(offset + j * 0.125 + 0.0625)
                y.append(0.15 + i * 0.216)
        
        x = np.array(x)
        y = np.array(y)
        r = np.full(N, 0.06)
        
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    initializations = [
        init_grid,
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
        init_hexagonal,
    ]
    
    for i, init_func in enumerate(initializations):
        if time_remaining() < 15:
            break
            
        try:
            z0 = init_func()
            # Add small noise to break symmetry
            if i > 0:
                noise = np.random.uniform(-0.01, 0.01, 3*N)
                z0 += noise
                z0 = np.clip(z0, 0, 1)
                z0[2::3] = np.maximum(z0[2::3], 0.01)

            res = run_optimizer(z0)
            
            # Check validity and score
            # We need to ensure the solution is actually valid (no overlaps)
            # L-BFGS-B minimizes the penalty, but might leave small violations if lambda is not huge.
            # We check the raw score and validity.
            
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            # Validity check
            valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6 or y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        if dist_sq < (r[i]+r[j])**2 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_solution = res.x
        except Exception as e:
            pass

    # If we have time, do a polish on the best solution using SLSQP with constraints
    # This ensures strict constraint satisfaction
    if best_solution is not None and time_remaining() > 10:
        try:
            def cons_func_int(z):
                x = z[0::3]
                y = z[1::3]
                r = z[2::3]
                
                c_bounds = np.concatenate([
                    x - r,
                    1.0 - x - r,
                    y - r,
                    1.0 - y - r
                ])
                
                c_pairwise = []
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                        rad_sum = r[i] + r[j]
                        c_pairwise.append(dist_sq - rad_sum**2)
                
                return np.concatenate([c_bounds, c_pairwise])

            constraints = [
                {'type': 'ineq', 'fun': cons_func_int}
            ]

            def obj_func_int(z):
                r = z[2::3]
                return -np.sum(r)

            res = minimize(
                obj_func_int, 
                best_solution, 
                method='SLSQP', 
                bounds=bounds_int, 
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success:
                # Verify again
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                valid = True
                for i in range(N):
                    if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6 or y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                            if dist_sq < (r[i]+r[j])**2 - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_solution = res.x
        except:
            pass

    # Format output
    if best_solution is None:
        # Fallback to grid
        z0 = init_grid()
        best_solution = z0
        
    x = best_solution[0::3]
    y = best_solution[1::3]
    r = best_solution[2::3]
    
    result = np.column_stack((x, y, r))
    return result
