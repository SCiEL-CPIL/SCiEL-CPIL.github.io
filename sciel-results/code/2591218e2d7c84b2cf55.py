import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 610  # safety margin before 630s kill
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid(centers, sides, tol=1e-7):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True
    
    def score(centers, sides):
        if not is_valid(centers, sides):
            return -np.inf
        return np.sum(sides)
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if centers is None or sides is None:
            return
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return
        if np.any(sides <= 0):
            return
        s = score(centers, sides)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # ---- Generate diverse initial configurations ----
    
    def make_config_2x5():
        """2 rows of 5 squares, all same size 0.2"""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        for i in range(n):
            r = i // 5
            col = i % 5
            c[i, 0] = (col + 0.5) * 0.2
            c[i, 1] = (r + 0.5) * 0.5
            s[i] = 0.2
        return c, s
    
    def make_config_3232():
        """3 top, 2 middle, 3 lower-middle, 2 bottom - varying sizes"""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        # Row 0: 3 squares of size 1/3
        for i in range(3):
            c[i, 0] = (i + 0.5) / 3.0
            c[i, 1] = 5/6
            s[i] = 1/3
        # Row 1: 2 squares of size 0.5
        for i in range(2):
            c[3+i, 0] = (i + 0.5) * 0.5
            c[3+i, 1] = 0.5
            s[3+i] = 0.5
        # Row 2: 3 squares of size 1/3
        for i in range(3):
            c[5+i, 0] = (i + 0.5) / 3.0
            c[5+i, 1] = 1/6
            s[5+i] = 1/3
        # Row 3: 2 squares of size 0.25
        for i in range(2):
            c[8+i, 0] = 0.125 + i * 0.25
            c[8+i, 1] = 0.125
            s[8+i] = 0.25
        return c, s
    
    def make_config_known():
        """The known 3.0 solution: 2 large + 8 small"""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        c[0] = [0.25, 0.75]
        s[0] = 0.5
        c[1] = [0.75, 0.75]
        s[1] = 0.5
        for r in range(2):
            for col in range(4):
                idx = 2 + r*4 + col
                c[idx, 0] = 0.125 + col*0.25
                c[idx, 1] = 0.125 + r*0.25
                s[idx] = 0.25
        return c, s
    
    def make_config_corner_cluster():
        """4 corners + center cluster"""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        # 4 corner squares of size 0.3
        corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for i, (cx, cy) in enumerate(corners):
            c[i] = [cx, cy]
            s[i] = 0.3
        # 6 center squares of size ~0.2
        for i in range(6):
            r = i // 3
            col = i % 3
            c[4+i, 0] = 0.3 + col * 0.2
            c[4+i, 1] = 0.3 + r * 0.2
            s[4+i] = 0.2
        return c, s
    
    def make_config_staircase():
        """Staircase pattern with decreasing sizes"""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        sizes = [0.4, 0.35, 0.3, 0.3, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2]
        positions = [
            (0.2, 0.8), (0.65, 0.8), (0.15, 0.45), (0.5, 0.45),
            (0.85, 0.5), (0.15, 0.15), (0.4, 0.15), (0.65, 0.15),
            (0.85, 0.15), (0.3, 0.7)
        ]
        for i in range(n):
            c[i] = positions[i]
            s[i] = sizes[i]
        return c, s
    
    # Objective function with smooth constraints
    def objective(params, beta=80.0):
        """
        params: [cx1, cy1, ..., cx10, cy10, log_s1, ..., log_s10]
        Uses log-sides for positivity.
        """
        centers = params[:2*n].reshape(n, 2)
        log_sides = params[2*n:]
        sides = np.exp(log_sides)
        
        # Clip sides to reasonable range
        sides = np.clip(sides, 1e-6, 0.5)
        
        obj = -np.sum(sides)
        
        # Boundary penalty (smooth quadratic)
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Left: cx - s/2 >= 0
            v = max(0.0, -(cx - s/2))
            obj += 50.0 * v * v
            # Right: cx + s/2 <= 1
            v = max(0.0, (cx + s/2) - 1.0)
            obj += 50.0 * v * v
            # Bottom: cy - s/2 >= 0
            v = max(0.0, -(cy - s/2))
            obj += 50.0 * v * v
            # Top: cy + s/2 <= 1
            v = max(0.0, (cy + s/2) - 1.0)
            obj += 50.0 * v * v
        
        # Overlap penalty: for each pair, need dx >= t OR dy >= t
        # Violation if dx < t AND dy < t
        tol = 1e-7
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t <= 0:
                    continue
                ox = max(0.0, t - dx)
                oy = max(0.0, t - dy)
                if ox > 0 and oy > 0:
                    # Smooth product penalty
                    obj += 50.0 * ox * oy
        
        return obj
    
    def objective_v2(params, beta=100.0):
        """
        Alternative: epigraph-style with auxiliary slack variables.
        params: [cx1..cx10, cy1..cy10, log_s1..log_s10, slack1..slack10]
        slack_i >= 0 ensures boundary feasibility.
        """
        centers = params[:2*n].reshape(n, 2)
        log_sides = params[2*n:4*n]
        slacks = params[4*n:]
        sides = np.exp(log_sides)
        sides = np.clip(sides, 1e-6, 0.5)
        
        obj = -np.sum(sides)
        
        # Boundary: use slack to absorb violations smoothly
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            slack = max(0.0, slacks[i])
            # Left
            v = max(0.0, -(cx - s/2) + slack*0.01)
            obj += 30.0 * v * v
            # Right
            v = max(0.0, (cx + s/2) - 1.0 + slack*0.01)
            obj += 30.0 * v * v
            # Bottom
            v = max(0.0, -(cy - s/2) + slack*0.01)
            obj += 30.0 * v * v
            # Top
            v = max(0.0, (cy + s/2) - 1.0 + slack*0.01)
            obj += 30.0 * v * v
            # Slack penalty (encourage zero slack)
            obj += 0.1 * slack * slack
        
        # Overlap penalty
        tol = 1e-7
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t <= 0:
                    continue
                ox = max(0.0, t - dx)
                oy = max(0.0, t - dy)
                if ox > 0 and oy > 0:
                    obj += 50.0 * ox * oy
        
        return obj
    
    # Build list of initial configurations
    configs = []
    
    c0, s0 = make_config_known()
    configs.append(np.concatenate([c0.flatten(), np.log(s0)]))
    
    c1, s1 = make_config_2x5()
    configs.append(np.concatenate([c1.flatten(), np.log(s1)]))
    
    c2, s2 = make_config_3232()
    configs.append(np.concatenate([c2.flatten(), np.log(s2)]))
    
    c3, s3 = make_config_corner_cluster()
    configs.append(np.concatenate([c3.flatten(), np.log(s3)]))
    
    c4, s4 = make_config_staircase()
    configs.append(np.concatenate([c4.flatten(), np.log(s4)]))
    
    # Add perturbed versions of the best known
    for sigma in [0.005, 0.01, 0.02, 0.03]:
        c_pert = c0 + np.random.normal(0, sigma, (n, 2))
        s_pert = s0 + np.random.normal(0, sigma, n)
        s_pert = np.clip(s_pert, 0.05, 0.6)
        configs.append(np.concatenate([c_pert.flatten(), np.log(s_pert)]))
    
    # Random valid-ish starts
    for _ in range(3):
        c_rand = np.random.uniform(0.1, 0.9, (n, 2))
        s_rand = np.random.uniform(0.1, 0.3, n)
        configs.append(np.concatenate([c_rand.flatten(), np.log(s_rand)]))
    
    # Run optimization from each start
    for idx, init in enumerate(configs):
        if elapsed() > 570:
            break
        try:
            res = minimize(objective, init, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < 0:
                c_opt = res.x[:2*n].reshape(n, 2)
                s_opt = np.exp(res.x[2*n:])
                s_opt = np.clip(s_opt, 1e-6, 0.5)
                update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Polish pass on best solution with tighter tolerance
    if best_centers is not None and elapsed() < 585:
        try:
            init_polish = np.concatenate([best_centers.flatten(), np.log(best_sides)])
            res_polish = minimize(objective, init_polish, method='L-BFGS-B',
                                   options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_polish.fun < 0:
                c_polish = res_polish.x[:2*n].reshape(n, 2)
                s_polish = np.exp(res_polish.x[2*n:])
                s_polish = np.clip(s_polish, 1e-6, 0.5)
                update_best(c_polish, s_polish)
        except Exception:
            pass
    
    # Second polish with objective_v2
    if best_centers is not None and elapsed() < 595:
        try:
            init_v2 = np.concatenate([best_centers.flatten(), np.log(best_sides), np.zeros(n)])
            res_v2 = minimize(objective_v2, init_v2, method='L-BFGS-B',
                               options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res_v2.fun < 0:
                c_v2 = res_v2.x[:2*n].reshape(n, 2)
                s_v2 = np.exp(res_v2.x[2*n:4*n])
                s_v2 = np.clip(s_v2, 1e-6, 0.5)
                update_best(c_v2, s_v2)
        except Exception:
            pass
    
    # Final ultra-fine polish
    if best_centers is not None and elapsed() < 605:
        try:
            init_final = np.concatenate([best_centers.flatten(), np.log(best_sides)])
            res_final = minimize(objective, init_final, method='L-BFGS-B',
                                  options={'maxiter': 8000, 'ftol': 1e-17, 'gtol': 1e-15})
            if res_final.fun < 0:
                c_final = res_final.x[:2*n].reshape(n, 2)
                s_final = np.exp(res_final.x[2*n:])
                s_final = np.clip(s_final, 1e-6, 0.5)
                update_best(c_final, s_final)
        except Exception:
            pass
    
    # Fallback
    if best_centers is None:
        best_centers = c0
        best_sides = s0
    
    # Final validity check
    if not is_valid(best_centers, best_sides):
        # Try to fix by shrinking
        best_sides = best_sides * 0.99
        best_centers = best_centers * 0.99 + 0.005
    
    return best_centers, best_sides
