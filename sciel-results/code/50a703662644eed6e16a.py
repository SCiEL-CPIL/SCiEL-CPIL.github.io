import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 32
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)

    def unpack(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def pack(x, y, r):
        z = np.zeros(3 * N)
        z[0::3] = x
        z[1::3] = r # Wait, interleaved is x, y, r? 
        # Let's stick to x, y, r interleaved: [x1, y1, r1, x2, y2, r2, ...]
        z[1::3] = y
        z[2::3] = r
        return z

    def calc_constraints(z):
        x, y, r = unpack(z)
        # Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
        c_bounds = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
        
        # Pairwise constraints: dist^2 - (r_i + r_j)^2 >= 0
        c_pairwise = np.zeros((N * (N - 1)) // 2)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum = r[i] + r[j]
                c_pairwise[idx] = dist_sq - rad_sum**2
                idx += 1
        
        return np.concatenate([c_bounds, c_pairwise])

    def obj_func(z):
        r = z[2::3]
        return -np.sum(r)

    def penalty_objective(z, penalty_weight):
        x, y, r = unpack(z)
        # Base objective
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        bound_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                     np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        obj += penalty_weight * np.sum(bound_viol**2)
        
        # Penalty for overlap violations
        # dist < r_i + r_j  =>  r_i + r_j - dist > 0
        # We use squared distance to avoid sqrt, but penalty on dist is better for smoothness?
        # Let's use (r_i + r_j)^2 - dist^2. If positive, they overlap.
        # Actually, (r_i + r_j) - dist is the linear violation.
        # Using squared linear violation is smoother.
        
        # Efficient pairwise distance calculation
        # Points: (N, 2)
        pts = np.column_stack((x, y))
        dists = cdist(pts, pts)
        # Upper triangle
        iu = np.triu_indices(N, k=1)
        dists_u = dists[iu]
        r_sum = r[iu[0]] + r[iu[1]]
        
        # Violation is max(0, r_sum - dist)
        viol = np.maximum(0, r_sum - dists_u)
        obj += penalty_weight * np.sum(viol**2)
        
        return obj

    def run_local_opt(z0, method='SLSQP', maxiter=200, ftol=1e-10):
        bounds = []
        for _ in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
            
        if method == 'SLSQP':
            constraints = [{'type': 'ineq', 'fun': calc_constraints}]
            res = minimize(
                obj_func, 
                z0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
        elif method == 'Nelder-Mead':
            # Use penalty method for NM
            # Start with moderate penalty
            res = minimize(
                lambda z: penalty_objective(z, 100.0), 
                z0, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={'maxiter': maxiter * 2, 'xatol': 1e-8, 'fatol': 1e-8}
            )
        return res

    def is_valid(z, tol=1e-6):
        x, y, r = unpack(z)
        if np.any(r < 0) or np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
            return False
        if np.any(r > x + tol) or np.any(r > 1-x + tol) or np.any(r > y + tol) or np.any(r > 1-y + tol):
            return False
            
        pts = np.column_stack((x, y))
        dists = cdist(pts, pts)
        iu = np.triu_indices(N, k=1)
        dists_u = dists[iu]
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(dists_u < r_sum - tol):
            return False
        return True

    def make_valid(z):
        # Simple projection: shrink radii if they touch boundaries or each other
        x, y, r = unpack(z)
        # Boundary check
        r = np.minimum(r, x)
        r = np.minimum(r, 1-x)
        r = np.minimum(r, y)
        r = np.minimum(r, 1-y)
        r = np.maximum(r, 0)
        
        # Pairwise check - iterative shrink
        for _ in range(10):
            pts = np.column_stack((x, y))
            dists = cdist(pts, pts)
            iu = np.triu_indices(N, k=1)
            dists_u = dists[iu]
            r_sum = r[iu[0]] + r[iu[1]]
            
            # Find violations
            viol = r_sum - dists_u
            if np.all(viol <= 1e-9):
                break
            
            # Shrink radii proportionally to violation
            # This is a heuristic
            shrink = np.zeros(N)
            for k in range(len(viol)):
                if viol[k] > 1e-9:
                    i = iu[0][k]
                    j = iu[1][k]
                    # Distribute shrinkage
                    s = viol[k] / 2
                    shrink[i] = min(shrink[i] + s, r[i])
                    shrink[j] = min(shrink[j] + s, r[j])
            
            if np.all(shrink == 0):
                break
            r -= shrink
            
        return pack(x, y, r)

    # Initializations
    def init_hex():
        # 4 rows of 8
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.06)
        
        idx = 0
        for i in range(4):
            offset = 0 if i % 2 == 0 else 0.0625
            for j in range(8):
                x[idx] = offset + j * 0.125 + 0.0625
                y[idx] = 0.15 + i * 0.216
                idx += 1
        
        return pack(x, y, r)

    def init_grid():
        # 6x6 grid, take 32
        side = 6
        xs, ys = np.meshgrid(np.linspace(0.5/6, 1-0.5/6, side), np.linspace(0.5/6, 1-0.5/6, side))
        x = xs.flatten()[:N]
        y = ys.flatten()[:N]
        r = np.full(N, 0.08)
        return pack(x, y, r)

    def init_random(seed):
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        return pack(x, y, r)

    def init_rect_4x8():
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.06)
        idx = 0
        for i in range(4):
            for j in range(8):
                x[idx] = (j + 0.5) / 8.0
                y[idx] = (i + 0.5) / 4.0
                idx += 1
        return pack(x, y, r)

    def init_rect_5x7():
        # 5 rows, 7 cols = 35, take 32
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.06)
        idx = 0
        for i in range(5):
            for j in range(7):
                if idx >= N: break
                x[idx] = (j + 0.5) / 7.0
                y[idx] = (i + 0.5) / 5.0
                idx += 1
        return pack(x, y, r)

    initializations = [
        init_hex,
        init_grid,
        init_rect_4x8,
        init_rect_5x7,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
    ]

    best_score = -np.inf
    best_solution = None

    # Phase 1: Coarse Search with Nelder-Mead (Penalty)
    for i, init_func in enumerate(initializations):
        if time_remaining() < 15: break
        
        z0 = init_func()
        
        # Run NM
        try:
            res_nm = run_local_opt(z0, method='Nelder-Mead', maxiter=100)
            if res_nm.success or res_nm.status == 0:
                z_nm = res_nm.x
                # Polish with SLSQP
                res_sl = run_local_opt(z_nm, method='SLSQP', maxiter=100, ftol=1e-8)
                if res_sl.success:
                    z_sol = res_sl.x
                    score = -res_sl.fun
                    if score > best_score:
                        # Check validity
                        if is_valid(z_sol):
                            best_score = score
                            best_solution = z_sol
        except Exception as e:
            pass

    # Phase 2: Fine Polish on Best Solution
    if best_solution is not None and time_remaining() > 10:
        try:
            res_polish = run_local_opt(best_solution, method='SLSQP', maxiter=500, ftol=1e-12)
            if res_polish.success:
                z_polished = res_polish.x
                score = -res_polish.fun
                if score > best_score:
                    if is_valid(z_polished):
                        best_score = score
                        best_solution = z_polished
        except Exception as e:
            pass

    # Phase 3: Try to expand radii further
    if best_solution is not None and time_remaining() > 5:
        z_current = best_solution.copy()
        x, y, r = unpack(z_current)
        
        # Try uniform scaling of radii by 1.01, 1.02...
        for scale in [1.01, 1.02, 1.05]:
            if time_remaining() < 5: break
            r_new = r * scale
            z_new = pack(x, y, r_new)
            
            try:
                res_exp = run_local_opt(z_new, method='SLSQP', maxiter=100, ftol=1e-10)
                if res_exp.success:
                    z_sol = res_exp.x
                    score = -res_exp.fun
                    if score > best_score:
                        if is_valid(z_sol):
                            best_score = score
                            best_solution = z_sol
            except Exception as e:
                pass

    # Final Fallback
    if best_solution is None:
        best_solution = init_hex()
        
    x, y, r = unpack(best_solution)
    # Ensure strict validity for return
    # If invalid, shrink
    if not is_valid(best_solution, tol=1e-8):
        best_solution = make_valid(best_solution)
        x, y, r = unpack(best_solution)

    result = np.column_stack((x, y, r))
    return result
