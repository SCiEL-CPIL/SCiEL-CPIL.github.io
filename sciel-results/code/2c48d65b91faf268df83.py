import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin

    best_centers = None
    best_sides = None
    best_score = -np.inf

    def penalty_and_obj(params):
        # params: [cx1, cy1, s1, ..., cx10, cy10, s10]
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        # Objective: Maximize sum(s) -> Minimize -sum(s)
        obj = -np.sum(s)
        
        # Boundary Penalty
        # Square i: [cx_i - s_i/2, cx_i + s_i/2]
        # Violation if cx_i - s_i/2 < 0 or cx_i + s_i/2 > 1
        # Penalty: max(0, s_i/2 - cx_i)^2 + max(0, cx_i + s_i/2 - 1)^2
        # Same for y
        
        # Vectorized calculation
        # left = cx - s/2, right = cx + s/2
        # viol_x = max(0, -left) + max(0, right - 1)
        # This is sum of squares.
        
        left_x = cx - 0.5 * s
        right_x = cx + 0.5 * s
        left_y = cy - 0.5 * s
        right_y = cy + 0.5 * s
        
        pen_x = np.sum(np.maximum(0, -left_x)**2) + np.sum(np.maximum(0, right_x - 1)**2)
        pen_y = np.sum(np.maximum(0, -left_y)**2) + np.sum(np.maximum(0, right_y - 1)**2)
        
        # Overlap Penalty
        # For each pair i, j
        # u = (s_i + s_j)/2 - |cx_i - cx_j|
        # v = (s_i + s_j)/2 - |cy_i - cy_j|
        # Penalty = softplus(softmin(u, v))^2
        
        # We can compute this in a loop for clarity, or vectorize.
        # n=10, so 45 pairs. Loop is fine.
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                u = 0.5 * (s[i] + s[j]) - np.abs(cx[i] - cx[j])
                v = 0.5 * (s[i] + s[j]) - np.abs(cy[i] - cy[j])
                
                # Softmin of u, v
                # If u,v are large negative, softmin is large negative.
                # We want min(u,v) approx.
                # Use log-sum-exp trick for min?
                # min(u,v) = -max(-u, -v)
                # max(a,b) = log(exp(a)+exp(b)) - log(2) approx?
                # Let's use a simple smooth min:
                # sm = -1/alpha * log((exp(-alpha*u) + exp(-alpha*v))/2)
                alpha = 10.0
                # Clip to avoid overflow
                u_c = np.clip(u, -50, 50)
                v_c = np.clip(v, -50, 50)
                sm = -1.0/alpha * np.log((np.exp(-alpha*u_c) + np.exp(-alpha*v_c))/2.0)
                
                # Softplus of sm
                # sp = log(1 + exp(sm))
                sm_c = np.clip(sm, -50, 50)
                sp = np.log(1.0 + np.exp(sm_c))
                
                pen_overlap += sp**2
        
        total_penalty = 100.0 * (pen_x + pen_y + pen_overlap)
        
        return obj + total_penalty

    def is_valid(centers, sides):
        # Check if valid
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0: return False
            if cx - s/2 < -1e-7 or cx + s/2 > 1 + 1e-7: return False
            if cy - s/2 < -1e-7 or cy + s/2 > 1 + 1e-7: return False
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1 = centers[i]
                s1 = sides[i]
                cx2, cy2 = centers[j]
                s2 = sides[j]
                t_x = (s1 + s2)/2
                t_y = (s1 + s2)/2
                if np.abs(cx1 - cx2) < t_x - 1e-7 and np.abs(cy1 - cy2) < t_y - 1e-7:
                    return False
        return True

    def get_score(centers, sides):
        return np.sum(sides)

    # Initial guesses
    inits = []
    
    # 1. 4x4 grid, pick 10 cells
    k = 4
    side = 1.0/k
    idx = np.arange(16)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    # Pick first 10
    init1 = np.zeros(30)
    init1[0::3] = cx[:10]
    init1[1::3] = cy[:10]
    init1[2::3] = side
    inits.append(init1)
    
    # 2. 3x3 grid + 1 random
    k = 3
    side = 1.0/k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    init2 = np.zeros(30)
    init2[0::3] = cx
    init2[1::3] = cy
    init2[2::3] = side * 0.9 # Shrink slightly to allow movement
    # 10th square
    init2[27] = 0.5 # cx
    init2[28] = 0.5 # cy
    init2[29] = 0.1 # s
    inits.append(init2)
    
    # 3. Random
    np.random.seed(42)
    for _ in range(20):
        init = np.random.uniform(0.1, 0.9, size=30)
        # Set sides to be smallish initially
        init[2::3] = np.random.uniform(0.1, 0.3, size=10)
        inits.append(init)

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.01, 0.5)] * 10

    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(penalty_and_obj, init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                s = res.x[2::3]
                centers = np.column_stack((cx, cy))
                sides = s
                if is_valid(centers, sides):
                    score = get_score(centers, sides)
                    if score > best_score:
                        best_score = score
                        best_centers = centers
                        best_sides = sides
        except Exception as e:
            pass

    # Fallback: simple grid if nothing found (should not happen)
    if best_centers is None:
        k = 4
        side = 1.0/k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.column_stack((cx, cy))
        best_sides = np.full(10, side)

    return best_centers, best_sides
