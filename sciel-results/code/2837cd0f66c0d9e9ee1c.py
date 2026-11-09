# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from itertools import combinations

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t > 0:
                if dx < t and dy < t:
                    return False
    return True

def _smooth_penalty(centers, sides, penalty_weight=1e8, smooth_eps=1e-4):
    """
    Smooth penalty using soft-min/soft-max for differentiable optimization.
    Uses log-sum-exp style smoothing for the max(0, violation) terms.
    """
    n = len(sides)
    penalty = 0.0
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Smooth boundary violations: softplus(x) ≈ max(0, x)
        violations = [
            -(cx - s/2),      # cx - s/2 < 0  =>  -(cx - s/2) > 0
            (cx + s/2 - 1),   # cx + s/2 > 1  =>  cx + s/2 - 1 > 0
            -(cy - s/2),
            (cy + s/2 - 1),
        ]
        for v in violations:
            # softplus: log(1 + exp(x)) ≈ max(0, x) for large |x|
            if v > 20:
                penalty += v
            elif v < -20:
                penalty += 0.0
            else:
                penalty += np.log1p(np.exp(v))
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0
            # Smooth overlap: max(0, t - dx) * max(0, t - dy)
            ox = t - dx
            oy = t - dy
            # Use smooth max via softplus-like
            if ox > 20:
                sx = ox
            elif ox < -20:
                sx = 0.0
            else:
                sx = np.log1p(np.exp(ox))
            if oy > 20:
                sy = oy
            elif oy < -20:
                sy = 0.0
            else:
                sy = np.log1p(np.exp(oy))
            penalty += sx * sy
    
    return penalty_weight * penalty

def _objective(x, n=10, penalty_weight=1e8):
    """
    Objective: minimize -sum(sides) + penalty
    x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}]
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    obj = -np.sum(sides) + _penalty(centers, sides, penalty_weight)
    return obj

def _local_search(centers, sides, max_iters=2000, deadline=None, rng_seed=42):
    """
    Enhanced local search with:
    1. Random perturbations with adaptive step size
    2. Coordinate descent for fine-tuning
    3. Pair adjustments
    4. Grow-and-reposition strategy
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    rng = np.random.RandomState(rng_seed)
    step_size = 0.05  # Adaptive step size
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Adaptive step: shrink if no improvement for a while
        if it % 200 == 0 and it > 0:
            step_size *= 0.7
        
        # Strategy: pick a random square and try to improve
        i = rng.randint(n)
        
        improved = False
        
        # Try changing side length with various step sizes
        for trial in range(8):
            scale = 1 + rng.uniform(-0.03, 0.08)
            s_new = best_s[i] * scale
            s_new = max(0.01, min(0.5, s_new))
            
            # Try adjusting position
            cx_new = best_c[i, 0] + rng.uniform(-step_size, step_size)
            cy_new = best_c[i, 1] + rng.uniform(-step_size, step_size)
            
            # Clamp to valid range
            cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
            cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
            
            c_new = best_c.copy()
            s_new_arr = best_s.copy()
            c_new[i] = [cx_new, cy_new]
            s_new_arr[i] = s_new
            
            if _is_valid(c_new, s_new_arr):
                score = np.sum(s_new_arr)
                if score > best_score + 1e-12:
                    best_score = score
                    best_c = c_new
                    best_s = s_new_arr
                    improved = True
        
        # Try pair adjustments
        if not improved and rng.random() < 0.4:
            j = rng.randint(n)
            if j != i:
                for _ in range(4):
                    c_new = best_c.copy()
                    s_new_arr = best_s.copy()
                    
                    for idx in [i, j]:
                        s_new = s_new_arr[idx] * (1 + rng.uniform(-0.02, 0.05))
                        s_new = max(0.01, min(0.5, s_new))
                        cx_new = c_new[idx, 0] + rng.uniform(-step_size*0.8, step_size*0.8)
                        cy_new = c_new[idx, 1] + rng.uniform(-step_size*0.8, step_size*0.8)
                        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
                        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
                        c_new[idx] = [cx_new, cy_new]
                        s_new_arr[idx] = s_new
                    
                    if _is_valid(c_new, s_new_arr):
                        score = np.sum(s_new_arr)
                        if score > best_score + 1e-12:
                            best_score = score
                            best_c = c_new
                            best_s = s_new_arr
        
        # Coordinate descent: fine-tune one coordinate at a time
        if rng.random() < 0.3:
            for coord in range(2 * n + n):
                if deadline is not None and time.time() > deadline:
                    break
                
                if coord < 2 * n:
                    # Position coordinate
                    idx = coord // 2
                    axis = coord % 2
                    for delta in [-0.005, 0.005, -0.001, 0.001]:
                        c_new = best_c.copy()
                        c_new[idx, axis] += delta
                        s = best_s[idx]
                        c_new[idx, axis] = max(s/2, min(1 - s/2, c_new[idx, axis]))
                        
                        if _is_valid(c_new, best_s):
                            score = np.sum(best_s)
                            if score > best_score + 1e-12:
                                best_score = score
                                best_c = c_new
                else:
                    # Side length coordinate
                    idx = coord - 2 * n
                    for scale in [1.01, 0.99, 1.02, 0.98]:
                        s_new = best_s[idx] * scale
                        s_new = max(0.01, min(0.5, s_new))
                        s_new_arr = best_s.copy()
                        s_new_arr[idx] = s_new
                        
                        if _is_valid(best_c, s_new_arr):
                            score = np.sum(s_new_arr)
                            if score > best_score + 1e-12:
                                best_score = score
                                best_s = s_new_arr
    
    return best_c, best_s

def _gradient_ascent(centers, sides, max_iters=500, deadline=None):
    """
    Use gradient ascent on a smooth objective to improve the configuration.
    Uses the smooth penalty formulation for better gradient estimates.
    """
    n = len(sides)
    x = np.concatenate([centers.flatten(), sides])
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    def smooth_obj(x_vec):
        """Negative of smooth objective (we minimize this)."""
        c = x_vec[:2*n].reshape(n, 2)
        s = x_vec[2*n:]
        # Penalize invalid side lengths
        s_pen = 0.0
        for i in range(n):
            if s[i] <= 0:
                s_pen += (s[i] - 0.01)**2
            if s[i] > 0.5:
                s_pen += (s[i] - 0.5)**2
        return -np.sum(s) + _smooth_penalty(c, s, penalty_weight=1e6) + s_pen
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient numerically with central differences
        eps = 1e-4
        grad = np.zeros_like(x)
        
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            f_plus = smooth_obj(x_plus)
            
            x_minus = x.copy()
            x_minus[k] -= eps
            f_minus = smooth_obj(x_minus)
            
            grad[k] = (f_plus - f_minus) / (2 * eps)
        
        # Adaptive learning rate
        lr = 0.0005 * (1.0 / (1.0 + it * 0.001))
        x_new = x - lr * grad
        
        # Project to valid bounds
        for i in range(n):
            s = x_new[2*n + i]
            s = max(0.01, min(0.5, s))
            x_new[2*n + i] = s
            x_new[i] = max(s/2, min(1 - s/2, x_new[i]))
            x_new[n+i] = max(s/2, min(1 - s/2, x_new[n+i]))
        
        x = x_new
        centers_new = x[:2*n].reshape(n, 2)
        sides_new = x[2*n:]
        
        if _is_valid(centers_new, sides_new):
            score = np.sum(sides_new)
            if score > best_score + 1e-12:
                best_score = score
                best_c = centers_new
                best_s = sides_new
    
    return best_c, best_s

def _try_fixed_orientation_lp(sides, orientations, n):
    """
    Given fixed side lengths and orientation choices, solve LP for centers.
    """
    c_obj = np.zeros(2*n)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        row = np.zeros(2*n)
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        row = np.zeros(2*n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
        
        row = np.zeros(2*n)
        row[n+i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        row = np.zeros(2*n)
        row[n+i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
    
    # Disjunctive constraints
    for i in range(n):
        for j in range(i+1, n):
            o = orientations.get((i,j), 0)
            sep = (sides[i] + sides[j]) / 2.0
            row = np.zeros(2*n)
            if o == 0:  # i left of j
                row[i] = 1.0
                row[j] = -1.0
                A_ub.append(row)
                b_ub.append(-sep)
            elif o == 1:  # i right of j
                row[i] = -1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(-sep)
            elif o == 2:  # i below j
                row[n+i] = 1.0
                row[n+j] = -1.0
                A_ub.append(row)
                b_ub.append(-sep)
            else:  # i above j
                row[n+i] = -1.0
                row[n+j] = 1.0
                A_ub.append(row)
                b_ub.append(-sep)
    
    if not A_ub:
        return np.zeros((n, 2)), True
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, 1)] * (2*n), method='highs')
    if res.status == 0:
        centers = res.x.reshape(n, 2)
        return centers, True
    return None, False

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
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
    
    # Fallback: 2x5 grid
    fallback_c = np.zeros((n, 2))
    fallback_s = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        fallback_c[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        fallback_s[i] = 0.2
    update_best(fallback_c, fallback_s)
    
    # Generate diverse initial configurations
    configs = []
    
    # 2x5 grid with equal sides
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c1[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s1[i] = 0.2
    configs.append((c1, s1))
    
    # 1x10 strip
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        c2[i] = [(i + 0.5) * 0.1, 0.5]
        s2[i] = 0.1
    configs.append((c2, s2))
    
    # 5x2 grid
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    for i in range(n):
        r = i % 2
        c = i // 2
        c3[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s3[i] = 0.2
    configs.append((c3, s3))
    
    # 3x3 grid + 1
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(9):
        r = i // 3
        c = i % 3
        c4[i] = [(c + 0.5) / 3.0, (r + 0.5) / 3.0]
        s4[i] = 1.0/3.0
    c4[9] = [0.5, 0.5]
    s4[9] = 0.1
    configs.append((c4, s4))
    
    # Random initializations with greedy placement
    rng = np.random.RandomState(123)
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Random sides
        s_rand = rng.uniform(0.08, 0.35, n)
        
        # Try to place using greedy approach
        c_rand = np.zeros((n, 2))
        placed = np.zeros(n, dtype=bool)
        
        # Sort by side length descending
        order = np.argsort(-s_rand)
        
        for idx in order:
            s = s_rand[idx]
            found = False
            for _ in range(200):
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                c_rand[idx] = [cx, cy]
                
                ok = True
                for j in range(n):
                    if placed[j] and j != idx:
                        dx = abs(c_rand[idx, 0] - c_rand[j, 0])
                        dy = abs(c_rand[idx, 1] - c_rand[j, 1])
                        t = (s_rand[idx] + s_rand[j]) / 2.0
                        if dx < t and dy < t:
                            ok = False
                            break
                if ok:
                    placed[idx] = True
                    found = True
                    break
            
            if not found:
                s_rand[idx] = max(0.05, s_rand[idx] * 0.85)
        
        if _is_valid(c_rand, s_rand):
            configs.append((c_rand, s_rand))
    
    # For each config, run local search
    for c_init, s_init in configs:
        if time.time() > deadline:
            break
        
        update_best(c_init, s_init)
        
        # Run local search
        c_ls, s_ls = _local_search(c_init, s_init, max_iters=500, deadline=deadline, rng_seed=42)
        update_best(c_ls, s_ls)
        
        # Also try gradient ascent
        if time.time() < deadline - 60:
            c_ga, s_ga = _gradient_ascent(c_ls, s_ls, max_iters=200, deadline=deadline)
            update_best(c_ga, s_ga)
    
    # Strategy: LP relaxation with different M values
    M_values = [1.0, 1.5, 2.0]
    
    for M in M_values:
        if time.time() > deadline:
            break
        
        num_pairs = n * (n - 1) // 2
        total_vars = 2 * n + n + 4 * num_pairs
        
        c_obj = np.zeros(total_vars)
        for i in range(n):
            c_obj[2 * n + i] = -1.0
        
        A_ub = []
        b_ub = []
        
        for i in range(n):
            row = np.zeros(total_vars)
            row[i] = -1.0
            row[2 * n + i] = -0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            row = np.zeros(total_vars)
            row[i] = 1.0
            row[2 * n + i] = 0.5
            A_ub.append(row)
            b_ub.append(1.0)
            
            row = np.zeros(total_vars)
            row[n + i] = -1.0
            row[2 * n + i] = -0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            row = np.zeros(total_vars)
            row[n + i] = 1.0
            row[2 * n + i] = 0.5
            A_ub.append(row)
            b_ub.append(1.0)
        
        pair_counter = 0
        for i in range(n):
            for j in range(i + 1, n):
                z_base = 2 * n + n + 4 * pair_counter
                
                row = np.zeros(total_vars)
                row[i] = 1.0
                row[j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 0] = M
                A_ub.append(row)
                b_ub.append(M)
                
                row = np.zeros(total_vars)
                row[i] = -1.0
                row[j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 1] = M
                A_ub.append(row)
                b_ub.append(M)
                
                row = np.zeros(total_vars)
                row[n + i] = 1.0
                row[n + j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 2] = M
                A_ub.append(row)
                b_ub.append(M)
                
                row = np.zeros(total_vars)
                row[n + i] = -1.0
                row[n + j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 3] = M
                A_ub.append(row)
                b_ub.append(M)
                
                pair_counter += 1
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0.0, 1.0)] * (2 * n)
        bounds += [(0.01, 0.5)] * n
        bounds += [(0.0, 1.0)] * (4 * num_pairs)
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            x = res.x
            centers_lp = x[:2 * n].reshape(n, 2)
            sides_lp = x[2 * n:3 * n]
            
            orientations = {}
            pair_counter = 0
            for i in range(n):
                for j in range(i + 1, n):
                    z_base = 3 * n + 4 * pair_counter
                    z_vals = x[z_base:z_base + 4]
                    o = int(np.argmax(z_vals))
                    orientations[(i, j)] = o
                    pair_counter += 1
            
            update_best(centers_lp, sides_lp)
            
            c_polished, ok = _try_fixed_orientation_lp(sides_lp, orientations, n)
            if ok and _is_valid(c_polished, sides_lp):
                update_best(c_polished, sides_lp)
            
            c_ls, s_ls = _local_search(centers_lp, sides_lp, max_iters=300, deadline=deadline, rng_seed=99)
            update_best(c_ls, s_ls)
    
    # Final polish: high-precision local search with very small steps
    if best_centers is not None and time.time() < deadline - 30:
        c_final, s_final = _local_search(best_centers, best_sides, max_iters=2000, deadline=deadline, rng_seed=777)
        update_best(c_final, s_final)
        
        # Additional coordinate-descent-style polish
        if time.time() < deadline - 15:
            c_polish = best_centers.copy()
            s_polish = best_sides.copy()
            for _ in range(50):
                if time.time() > deadline:
                    break
                for i in range(n):
                    for axis in range(2):
                        for delta in [-0.0005, 0.0005, -0.0001, 0.0001]:
                            c_test = c_polish.copy()
                            c_test[i, axis] += delta
                            s = s_polish[i]
                            c_test[i, axis] = max(s/2, min(1 - s/2, c_test[i, axis]))
                            if _is_valid(c_test, s_polish):
                                c_polish = c_test
                    # Try growing
                    s_test = s_polish.copy()
                    s_test[i] = min(0.5, s_polish[i] * 1.005)
                    if _is_valid(c_polish, s_test):
                        s_polish = s_test
                update_best(c_polish, s_polish)
    
    if best_centers is None:
        best_centers = fallback_c
        best_sides = fallback_s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
