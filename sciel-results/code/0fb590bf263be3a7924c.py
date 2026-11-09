import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin
    
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
        s = score(centers, sides)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
            best_sides = sides.copy()

    # 1. Analytic Solution: 2 large (0.5), 8 small (0.25)
    c_ana = np.zeros((n, 2))
    s_ana = np.zeros(n)
    c_ana[0] = [0.25, 0.75]; s_ana[0] = 0.5
    c_ana[1] = [0.75, 0.75]; s_ana[1] = 0.5
    for r in range(2):
        for col in range(4):
            idx = 2 + r*4 + col
            c_ana[idx] = [0.125 + col*0.25, 0.125 + r*0.25]
            s_ana[idx] = 0.25
    update_best(c_ana, s_ana)
    
    # 2. Alternative Analytic: 3x3 grid of 1/3 (9 squares) + 1 small?
    # This is hard to fit. Skip.
    
    # Objective for DE
    def de_objective(params):
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]
        
        if np.any(sides < 0):
            return 1e6
            
        obj = -np.sum(sides)
        
        # Boundary penalty
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            v_l = max(0, -(cx - s/2))
            v_r = max(0, (cx + s/2) - 1)
            v_b = max(0, -(cy - s/2))
            v_t = max(0, (cy + s/2) - 1)
            obj += 1000 * (v_l**2 + v_r**2 + v_b**2 + v_t**2)
            
        # Overlap penalty
        tol = 1e-7
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t <= 0:
                    continue
                ox = max(0, t - dx)
                oy = max(0, t - dy)
                if ox > 0 and oy > 0:
                    obj += 1000 * ox * oy
                    
        return obj

    # Run DE
    bounds = [(0, 1)] * (2*n) + [(0, 1)] * n
    
    try:
        # Use seed for reproducibility, but try multiple seeds if time permits?
        # For now, one good run.
        res = differential_evolution(de_objective, bounds, maxiter=150, popsize=15, tol=1e-6, disp=False, seed=42)
        
        if elapsed() < deadline - 30:
             params_opt = res.x
             c_opt = params_opt[:2*n].reshape(n, 2)
             s_opt = np.maximum(params_opt[2*n:], 1e-6)
             update_best(c_opt, s_opt)
             
             # Polish with SLSQP
             if is_valid(c_opt, s_opt):
                 res_polish = minimize(de_objective, params_opt, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-12})
                 if res_polish.success:
                     params_pol = res_polish.x
                     c_pol = params_pol[:2*n].reshape(n, 2)
                     s_pol = np.maximum(params_pol[2*n:], 1e-6)
                     update_best(c_pol, s_pol)
                     
    except Exception:
        pass

    # Try a few random restarts with SLSQP if time remains
    if elapsed() < deadline - 10:
        for _ in range(5):
            if elapsed() > deadline - 5: break
            # Random init
            c_rand = np.random.rand(n, 2)
            s_rand = np.random.rand(n) * 0.3 + 0.1
            # Scale to fit?
            params_rand = np.concatenate([c_rand.flatten(), s_rand])
            try:
                res_r = minimize(de_objective, params_rand, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-10})
                if res_r.success:
                    c_r = res_r.x[:2*n].reshape(n, 2)
                    s_r = np.maximum(res_r.x[2*n:], 1e-6)
                    update_best(c_r, s_r)
            except:
                pass

    if best_centers is None:
        best_centers = c_ana
        best_sides = s_ana

    return best_centers, best_sides
