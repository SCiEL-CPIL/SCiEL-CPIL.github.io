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

def _objective(x, n=10):
    """
    Objective function for optimization.
    x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}]
    Returns: -sum(s_i) + penalty_terms
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Negative sum of sides (we minimize)
    obj = -np.sum(sides)
    
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < 0:
            obj += 1e6 * (cx - s/2)**2
        if cx + s/2 > 1:
            obj += 1e6 * (cx + s/2 - 1)**2
        if cy - s/2 < 0:
            obj += 1e6 * (cy - s/2)**2
        if cy + s/2 > 1:
            obj += 1e6 * (cy + s/2 - 1)**2
    
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0
            if abs(dx) < t and abs(dy) < t:
                overlap_x = t - abs(dx)
                overlap_y = t - abs(dy)
                obj += 1e6 * (overlap_x * overlap_y)
    
    return obj

def _lp_pack_fixed_sides(sides, orientations):
    """
    Given fixed side lengths and orientation choices, solve LP for centers.
    orientations: dict mapping (i,j) -> 0,1,2,3
    Returns (centers, success)
    """
    n = len(sides)
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

def _local_search(centers, sides, max_iters=500, deadline=None):
    """
    Local search: try moving individual squares to improve sum of sides.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = rng.randint(n)
        
        # Try changing side length
        s_new = best_s[i] * (1 + rng.uniform(-0.05, 0.1))
        s_new = max(0.01, min(0.5, s_new))
        
        # Try adjusting position
        cx_new = best_c[i, 0] + rng.uniform(-0.05, 0.05)
        cy_new = best_c[i, 1] + rng.uniform(-0.05, 0.05)
        
        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
        
        c_new = best_c.copy()
        s_new_arr = best_s.copy()
        c_new[i] = [cx_new, cy_new]
        s_new_arr[i] = s_new
        
        if _is_valid(c_new, s_new_arr):
            score = np.sum(s_new_arr)
            if score > best_score:
                best_score = score
                best_c = c_new
                best_s = s_new_arr
    
    return best_c, best_s

def _differential_evolution(centers, sides, deadline, pop_size=20, max_gen=100):
    """
    Differential evolution for global exploration.
    """
    n = len(sides)
    rng = np.random.RandomState(77)
    
    # Initialize population
    pop_c = np.zeros((pop_size, n, 2))
    pop_s = np.zeros((pop_size, n))
    
    for k in range(pop_size):
        if k == 0:
            pop_c[k] = centers.copy()
            pop_s[k] = sides.copy()
        else:
            # Perturb from best or random
            base = pop_c[rng.randint(k)] if k > 0 else centers
            pop_c[k] = base + rng.uniform(-0.1, 0.1, (n, 2))
            pop_s[k] = sides * (1 + rng.uniform(-0.2, 0.2, n))
            pop_s[k] = np.clip(pop_s[k], 0.01, 0.5)
            # Clamp positions
            for i in range(n):
                pop_c[k, i, 0] = max(pop_s[k, i]/2, min(1 - pop_s[k, i]/2, pop_c[k, i, 0]))
                pop_c[k, i, 1] = max(pop_s[k, i]/2, min(1 - pop_s[k, i]/2, pop_c[k, i, 1]))
    
    best_c = centers.copy()
    best_s = sides.copy()
    best_score = np.sum(sides)
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        for k in range(pop_size):
            if time.time() > deadline:
                break
            
            # Select three distinct random individuals
            idxs = rng.choice(pop_size, 3, replace=False)
            a, b, d = idxs
            
            # Mutation
            trial_c = pop_c[a] + 0.5 * (pop_c[b] - pop_c[d])
            trial_s = pop_s[a] + 0.5 * (pop_s[b] - pop_s[d])
            trial_s = np.clip(trial_s, 0.01, 0.5)
            
            # Clamp positions
            for i in range(n):
                trial_c[i, 0] = max(trial_s[i]/2, min(1 - trial_s[i]/2, trial_c[i, 0]))
                trial_c[i, 1] = max(trial_s[i]/2, min(1 - trial_s[i]/2, trial_c[i, 1]))
            
            # Selection
            if _is_valid(trial_c, trial_s):
                trial_score = np.sum(trial_s)
                if trial_score > np.sum(pop_s[k]):
                    pop_c[k] = trial_c
                    pop_s[k] = trial_s
                    if trial_score > best_score:
                        best_score = trial_score
                        best_c = trial_c.copy()
                        best_s = trial_s.copy()
    
    return best_c, best_s

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
    
    # Primary seed: known optimal 3-packing (2 large + 8 small)
    # Two large squares of side 0.5 in corners, 8 small of side 0.25
    c_seed = np.zeros((n, 2))
    s_seed = np.zeros(n)
    
    # Square 0: large, bottom-left
    c_seed[0] = [0.25, 0.25]
    s_seed[0] = 0.5
    
    # Square 1: large, top-right
    c_seed[1] = [0.75, 0.75]
    s_seed[1] = 0.5
    
    # Squares 2-9: small, side 0.25
    small_positions = [
        [0.75, 0.25],  # bottom-right
        [0.25, 0.75],  # top-left
        [0.5, 0.5],    # center
        [0.125, 0.5],  # left-middle
        [0.875, 0.5],  # right-middle
        [0.5, 0.125],  # bottom-middle
        [0.5, 0.875],  # top-middle
        [0.125, 0.125] # extra
    ]
    
    for i in range(8):
        c_seed[i+2] = small_positions[i]
        s_seed[i+2] = 0.25
    
    update_best(c_seed, s_seed)
    
    # Fallback: 2x5 grid
    fallback_c = np.zeros((n, 2))
    fallback_s = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        fallback_c[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        fallback_s[i] = 0.2
    update_best(fallback_c, fallback_s)
    
    # Run local search from the seed
    c_ls, s_ls = _local_search(c_seed, s_seed, max_iters=300, deadline=deadline)
    update_best(c_ls, s_ls)
    
    # Run differential evolution for global exploration
    c_de, s_de = _differential_evolution(c_ls, s_ls, deadline, pop_size=15, max_gen=50)
    update_best(c_de, s_de)
    
    # More local search from DE result
    c_ls2, s_ls2 = _local_search(c_de, s_de, max_iters=300, deadline=deadline)
    update_best(c_ls2, s_ls2)
    
    # Try LP relaxation with different M values
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
            
            c_polished, ok = _lp_pack_fixed_sides(sides_lp, orientations)
            if ok and _is_valid(c_polished, sides_lp):
                update_best(c_polished, sides_lp)
            
            c_ls3, s_ls3 = _local_search(centers_lp, sides_lp, max_iters=100, deadline=deadline)
            update_best(c_ls3, s_ls3)
    
    if best_centers is None:
        best_centers = fallback_c
        best_sides = fallback_s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
