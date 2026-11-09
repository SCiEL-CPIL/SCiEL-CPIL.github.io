import numpy as np
import time
from scipy.optimize import minimize, linprog

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
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

def _smooth_penalty(params, n=10, tol=1e-7, beta=50.0):
    """Smooth penalty for non-linear optimization."""
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = 0.0
    
    # Boundary penalties (smooth)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += 1e6 * (1.0 - s)
            continue
        v = -tol - (cx - s/2)
        if v > 0:
            pen += 1e6 * v
        v = (cx + s/2) - (1 + tol)
        if v > 0:
            pen += 1e6 * v
        v = -tol - (cy - s/2)
        if v > 0:
            pen += 1e6 * v
        v = (cy + s/2) - (1 + tol)
        if v > 0:
            pen += 1e6 * v
            
    # Overlap penalties (smooth)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0 - tol
            
            vx = t - abs(dx)
            vy = t - abs(dy)
            
            sx = np.maximum(0, vx)
            sy = np.maximum(0, vy)
            pen += 1e5 * (sx * sy)
            
    return pen

def _objective(params, n=10, tol=1e-7):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = _smooth_penalty(params, n, tol)
    
    obj = -np.sum(sides) + pen
    return obj

def _lp_solve(ordering, n=10):
    """
    Solve LP for square packing with given ordering constraints.
    Variables: x_i (left edge), y_i (bottom edge), s_i (side)
    Objective: maximize sum(s_i)
    """
    # Variable indices: 0..n-1 for x_i, n..2n-1 for y_i, 2n..3n-1 for s_i
    nv = 3 * n
    
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c_obj = np.zeros(nv)
    for i in range(n):
        c_obj[2*n + i] = -1.0
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Boundary constraints
    for i in range(n):
        # x_i >= 0 -> -x_i <= 0
        row = np.zeros(nv)
        row[i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
        # x_i + s_i <= 1
        row = np.zeros(nv)
        row[i] = 1.0
        row[2*n + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
        # y_i >= 0 -> -y_i <= 0
        row = np.zeros(nv)
        row[n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
        # y_i + s_i <= 1
        row = np.zeros(nv)
        row[n + i] = 1.0
        row[2*n + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
        # s_i >= 0.05 (small minimum to avoid degeneracy)
        row = np.zeros(nv)
        row[2*n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(-0.05)
    
    # Ordering constraints
    for (i, j), ord_val in ordering.items():
        row = np.zeros(nv)
        if ord_val == 0:  # i left of j: x_i + s_i <= x_j
            row[i] = 1.0
            row[2*n + i] = 1.0
            row[j] = -1.0
        elif ord_val == 1:  # j left of i: x_j + s_j <= x_i
            row[j] = 1.0
            row[2*n + j] = 1.0
            row[i] = -1.0
        elif ord_val == 2:  # i below j: y_i + s_i <= y_j
            row[n + i] = 1.0
            row[2*n + i] = 1.0
            row[n + j] = -1.0
        elif ord_val == 3:  # j below i: y_j + s_j <= y_i
            row[n + j] = 1.0
            row[2*n + j] = 1.0
            row[n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    if not A_ub_rows:
        return None
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, 
                      bounds=[(0, 1.0)] * (2*n) + [(0.05, 1.0)] * n,
                      method='highs')
        if res.success:
            x_vals = res.x[:n]
            y_vals = res.x[n:2*n]
            s_vals = res.x[2*n:3*n]
            
            centers = np.column_stack([x_vals + s_vals/2, y_vals + s_vals/2])
            sides = s_vals
            
            if _is_valid(centers, sides):
                return centers, sides
    except:
        pass
    
    return None

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # --- LP-based initialization with multiple ordering patterns ---
    
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Pattern 1: 4 corners large, 6 small in cross
    # Large: 0(BL), 1(BR), 2(TL), 3(TR)
    # Small: 4-7 in central vertical strip, 8-9 in side gaps
    ordering1 = {}
    # Large-large: diagonal separation
    ordering1[(0, 1)] = 0  # 0 left of 1
    ordering1[(0, 2)] = 3  # 2 above 0 (y_2 + s_2 <= y_0 is wrong, we want y_0 + s_0 <= y_2)
    # Let's be precise:
    # 0: BL, 1: BR, 2: TL, 3: TR
    # 0 vs 1: 0 left of 1 -> x_0 + s_0 <= x_1 (ord 0)
    # 0 vs 2: 0 below 2 -> y_0 + s_0 <= y_2 (ord 2)
    # 0 vs 3: 0 left of 3 AND 0 below 3. Pick one: 0 below 3 (ord 2)
    # 1 vs 2: 2 above 1? No, 1 is BR, 2 is TL. They are diagonal. 1 right of 2 -> x_2 + s_2 <= x_1 (ord 1 for pair (2,1) but we store (i,j) with i<j)
    # Pair (1,2): 1 is BR, 2 is TL. 2 left of 1 -> x_2 + s_2 <= x_1 (ord 1)
    # 1 vs 3: 1 below 3 -> y_1 + s_1 <= y_3 (ord 2)
    # 2 vs 3: 2 left of 3 -> x_2 + s_2 <= x_3 (ord 0)
    
    ordering1[(0, 1)] = 0
    ordering1[(0, 2)] = 2
    ordering1[(0, 3)] = 2
    ordering1[(1, 2)] = 1
    ordering1[(1, 3)] = 2
    ordering1[(2, 3)] = 0
    
    # Small in central strip: 4,5,6,7 at x=0.5, y=0.125, 0.375, 0.625, 0.875
    # They are stacked vertically
    ordering1[(4, 5)] = 2  # 4 below 5
    ordering1[(5, 6)] = 2  # 5 below 6
    ordering1[(6, 7)] = 2  # 6 below 7
    
    # Side gaps: 8 left, 9 right
    ordering1[(8, 9)] = 0  # 8 left of 9
    
    # Large vs Small
    # Large 0 (BL) vs Small 4 (bottom center): 0 left of 4
    ordering1[(0, 4)] = 0
    # Large 0 (BL) vs Small 5 (center): 0 left of 5? No, 5 is higher. 0 below 5?
    # Let's just say Large are in corners, Small are in center.
    # 0 (BL) vs 4,5,6,7 (center strip): 0 left of them?
    # x_0 + s_0 <= x_4. If s_0=0.375, x_0=0, x_4=0.375. Yes.
    for j in [4, 5, 6, 7]:
        ordering1[(0, j)] = 0
        ordering1[(1, j)] = 1  # j left of 1
        ordering1[(2, j)] = 0  # 2 left of j? No, 2 is TL. x_2 + s_2 <= x_j. If x_2=0, s_2=0.375, x_j=0.375. Yes.
        ordering1[(3, j)] = 1  # j left of 3
    
    # 8 (left gap) vs Small: 8 left of 4,5,6,7
    for j in [4, 5, 6, 7]:
        ordering1[(8, j)] = 0
    # 9 (right gap) vs Small: 4,5,6,7 left of 9
    for j in [4, 5, 6, 7]:
        ordering1[(j, 9)] = 1
    
    # 8 vs Large: 8 is in left gap, y=0.5.
    # 8 vs 0 (BL): 0 below 8? y_0 + s_0 <= y_8. 0.375 <= 0.375. Yes.
    ordering1[(0, 8)] = 2
    # 8 vs 1 (BR): 1 right of 8. x_8 + s_8 <= x_1.
    ordering1[(1, 8)] = 1
    # 8 vs 2 (TL): 8 below 2. y_8 + s_8 <= y_2.
    ordering1[(2, 8)] = 2
    # 8 vs 3 (TR): 3 right of 8 and above 8.
    ordering1[(3, 8)] = 1
    
    # 9 vs Large: 9 is in right gap, y=0.5.
    ordering1[(0, 9)] = 0  # 0 left of 9
    ordering1[(1, 9)] = 2  # 1 below 9
    ordering1[(2, 9)] = 0  # 2 left of 9
    ordering1[(3, 9)] = 2  # 3 above 9 -> y_3 + s_3 <= y_9 is wrong. 9 below 3.
    # Wait, 9 is y=0.5, 3 is y=0.8125. So 9 below 3.
    # Pair (3,9): 9 below 3 -> y_9 + s_9 <= y_3. This is ord 3 for (3,9)? No.
    # Ord 2: i below j. i=3, j=9. y_3 + s_3 <= y_9. Wrong.
    # Ord 3: j below i. i=3, j=9. y_9 + s_9 <= y_3. Correct.
    ordering1[(3, 9)] = 3
    
    lp_res1 = _lp_solve(ordering1, n)
    if lp_res1 is not None:
        update_best(lp_res1[0], lp_res1[1])

    # Pattern 2: 2 large top, 8 small bottom grid
    ordering2 = {}
    # Large: 0 (TL), 1 (TR)
    ordering2[(0, 1)] = 0
    # Small: 2-9 in 2x4 grid
    # Row 0: 2,3,4,5. Row 1: 6,7,8,9
    for r in range(2):
        for c in range(4):
            i = 2 + r*4 + c
            for c2 in range(c+1, 4):
                j = 2 + r*4 + c2
                ordering2[(i, j)] = 0  # i left of j
    for i in range(2, 6):
        for j in range(6, 10):
            ordering2[(i, j)] = 2  # i below j
    
    # Large vs Small: Large above Small
    for i in [0, 1]:
        for j in range(2, 10):
            ordering2[(i, j)] = 3  # j below i
    
    lp_res2 = _lp_solve(ordering2, n)
    if lp_res2 is not None:
        update_best(lp_res2[0], lp_res2[1])

    # Pattern 3: 5 large, 5 small?
    # Let's try a different heuristic: 1 large center, 9 around?
    # Or just use the seed config as a start for optimization.
    
    # Seed Config 1: 2 large 0.5, 8 small 0.25
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    c1[0] = [0.25, 0.75]; s1[0] = 0.5
    c1[1] = [0.75, 0.75]; s1[1] = 0.5
    idx = 2
    for r in range(2):
        for c in range(4):
            c1[idx] = [(c + 0.5)*0.25, 0.125 + r*0.25]
            s1[idx] = 0.25
            idx += 1
    update_best(c1, s1)
    
    # Seed Config 2: 4 corners 0.375, 6 small 0.25
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    a = 0.375
    b = 0.25
    c2[0] = [a/2, a/2]; s2[0] = a
    c2[1] = [1-a/2, a/2]; s2[1] = a
    c2[2] = [a/2, 1-a/2]; s2[2] = a
    c2[3] = [1-a/2, 1-a/2]; s2[3] = a
    c2[4] = [0.5, 0.125]; s2[4] = b
    c2[5] = [0.5, 0.375]; s2[5] = b
    c2[6] = [0.5, 0.625]; s2[6] = b
    c2[7] = [0.5, 0.875]; s2[7] = b
    c2[8] = [0.1875, 0.5]; s2[8] = b
    c2[9] = [0.8125, 0.5]; s2[9] = b
    update_best(c2, s2)

    # Optimization
    np.random.seed(42)
    starts = [(c1.copy(), s1.copy()), (c2.copy(), s2.copy())]
    
    if lp_res1 is not None:
        starts.append((lp_res1[0].copy(), lp_res1[1].copy()))
    if lp_res2 is not None:
        starts.append((lp_res2[0].copy(), lp_res2[1].copy()))
    
    # Add random perturbations
    for c_init, s_init in starts:
        for _ in range(5):
            c_r = c_init.copy() + np.random.normal(0, 0.05, (n, 2))
            s_r = s_init.copy() + np.random.normal(0, 0.02, n)
            s_r = np.clip(s_r, 0.1, 0.6)
            starts.append((c_r, s_r))

    for c_init, s_init in starts:
        if time.time() > deadline:
            break
        params_init = np.concatenate([c_init.flatten(), s_init])
        
        try:
            # Use Powell for better handling of non-smooth objectives
            res = minimize(_objective, params_init, args=(n,), method='Powell',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
            
            # Polish with COBYLA if valid
            if _is_valid(c_opt, s_opt) and time.time() < deadline - 30:
                res_polish = minimize(_objective, params_opt, args=(n,), method='COBYLA',
                                      options={'maxiter': 300, 'rhobeg': 1e-6})
                params_pol = res_polish.x
                c_pol = params_pol[:2*n].reshape(n, 2)
                s_pol = np.maximum(params_pol[2*n:], 1e-6)
                update_best(c_pol, s_pol)
                
        except Exception:
            pass

    if best_centers is None:
        best_centers = c1
        best_sides = s1

    # Final high-precision polish on the best result
    if time.time() < deadline - 60:
        try:
            params_best = np.concatenate([best_centers.flatten(), best_sides])
            res_final = minimize(_objective, params_best, args=(n,), method='Powell',
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            params_final = res_final.x
            c_final = params_final[:2*n].reshape(n, 2)
            s_final = np.maximum(params_final[2*n:], 1e-6)
            update_best(c_final, s_final)
        except:
            pass

    return best_centers, best_sides
