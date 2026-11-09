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

    # 2. Alternative Analytic: Try a 3x3 grid of 1/3 squares (9 squares) + 1 small
    # 9 squares of side 1/3 fill the unit square exactly.
    # The 10th square must be tiny.
    # Sum = 9*(1/3) + epsilon = 3 + epsilon.
    # This is better than 2.0!
    c_ana2 = np.zeros((n, 2))
    s_ana2 = np.zeros(n)
    for i in range(3):
        for j in range(3):
            idx = i*3 + j
            c_ana2[idx] = [1/6 + j/3, 1/6 + i/3]
            s_ana2[idx] = 1/3
    # The 10th square? It doesn't fit.
    # We can shrink the 9 squares slightly to make room for the 10th.
    # Let side of 9 squares be s, and 10th be t.
    # 3*s + t <= 1 (if 10th is in a corner, say)
    # Maximize 9*s + t subject to 3*s + t <= 1, s>0, t>0.
    # t = 1 - 3*s. Sum = 9*s + 1 - 3*s = 1 + 6*s.
    # Max s such that t > 0 => s < 1/3.
    # As s -> 1/3, sum -> 1 + 6*(1/3) = 3.
    # So we can get arbitrarily close to 3.
    # Let's set s = 1/3 - 1e-6, t = 1 - 3*(1/3 - 1e-6) = 3e-6.
    s_val = 1/3 - 1e-6
    t_val = 1 - 3*s_val
    c_ana2[9] = [t_val/2, t_val/2] # Place in corner?
    # Wait, the 9 squares form a 3x3 grid.
    # If we shrink them, there is a gap.
    # The gap is a cross shape.
    # A small square can fit in one of the corner gaps?
    # Actually, if we shrink the 9 squares, they are separated by gaps.
    # The 10th square can fit in the center? No, center is surrounded.
    # It can fit in a corner of the unit square?
    # If the 9 squares are centered at 1/6, 1/2, 5/6.
    # Shrink them to s.
    # The corner square (0,0) is at [1/6 - s/2, 1/6 + s/2].
    # Distance to 0 is 1/6 - s/2.
    # So a square of side 2*(1/6 - s/2) = 1/3 - s can fit in the corner?
    # No, the corner of the unit square is [0, 1/3 - s] x [0, 1/3 - s].
    # So t = 1/3 - s.
    # Sum = 9*s + (1/3 - s) = 8*s + 1/3.
    # Max s such that t > 0 => s < 1/3.
    # As s -> 1/3, sum -> 8/3 + 1/3 = 3.
    # Let's try s = 0.33, t = 0.0033.
    s_val = 0.33
    t_val = 1/3 - s_val
    c_ana2[9] = [t_val/2, t_val/2]
    s_ana2[9] = t_val
    # Check validity
    if is_valid(c_ana2, s_ana2):
        update_best(c_ana2, s_ana2)

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
        # Use seed for reproducibility
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
