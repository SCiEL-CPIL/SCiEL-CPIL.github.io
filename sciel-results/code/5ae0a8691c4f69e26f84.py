import numpy as np
import itertools
import random
import time
import scipy.optimize
import scipy.sparse

def construct(q, n, radius):
    total_words = q ** n
    
    # If the space is small enough, try to solve exactly using MILP
    if total_words <= 5000:
        try:
            code = solve_covering_code_milp(q, n, radius)
            if code is not None:
                return code
        except Exception:
            pass
            
    # Fallback to heuristic
    return solve_covering_code_heuristic(q, n, radius)

def solve_covering_code_milp(q, n, radius):
    words = [list(p) for p in itertools.product(range(q), repeat=n)]
    N = len(words)
    words_arr = np.array(words)
    
    # Compute Hamming distance matrix
    # dist[i, j] = number of positions where words[i] != words[j]
    dist_mat = np.zeros((N, N), dtype=np.int8)
    for pos in range(n):
        # (N, 1) != (1, N) -> (N, N)
        diff = (words_arr[:, pos][:, None] != words_arr[:, pos][None, :]).astype(np.int8)
        dist_mat += diff
        
    # Create mask for covering constraint: dist <= radius
    mask = dist_mat <= radius
    
    # Build sparse matrix
    A_sparse = scipy.sparse.csr_matrix(mask)
    
    # Objective: minimize sum(x)
    c = np.ones(N)
    
    # Constraints: A @ x >= 1
    cons = scipy.optimize.LinearConstraint(A_sparse, 1, np.inf)
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * N
    
    # Integrality: x is integer
    integrality = np.ones(N)
    
    res = scipy.optimize.milp(c, constraints=cons, integrality=integrality, bounds=bounds)
    
    if res.success and res.x is not None:
        x = res.x
        # Threshold to handle floating point errors
        indices = np.where(x > 0.5)[0]
        if len(indices) > 0:
            return [words[i] for i in indices]
            
    return None

def solve_covering_code_heuristic(q, n, radius):
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    N = len(all_words)
    
    # Precompute balls as sets of indices for faster lookup?
    # Or use bitsets.
    # For N=2187, bitsets are good.
    
    balls = [0] * N
    for i, w in enumerate(all_words):
        mask = 0
        for j, v in enumerate(all_words):
            # Distance check
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                mask |= (1 << j)
        balls[i] = mask
        
    all_ones_mask = (1 << N) - 1
    
    def greedy_cover(uncovered_mask):
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            for i in range(N):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    def remove_redundants(selected):
        if not selected:
            return []
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    def local_search_2_1(selected):
        if len(selected) < 3:
            return None
        for _ in range(10):
            i1, i2 = random.sample(range(len(selected)), 2)
            c1 = selected[i1]
            c2 = selected[i2]
            rest_mask = 0
            for i in range(len(selected)):
                if i != i1 and i != i2:
                    rest_mask |= balls[selected[i]]
            uncovered = all_ones_mask & ~rest_mask
            if uncovered == 0:
                new_selected = [x for x in selected if x != c1 and x != c2]
                new_selected = remove_redundants(new_selected)
                if len(new_selected) < len(selected):
                    return new_selected
                continue
            best_idx = -1
            max_cover = 0
            for k in range(N):
                if k == c1 or k == c2:
                    continue
                c = (balls[k] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = k
            if best_idx != -1 and (balls[best_idx] & uncovered) == uncovered:
                new_selected = [x for x in selected if x != c1 and x != c2]
                new_selected.append(best_idx)
                new_selected = remove_redundants(new_selected)
                if len(new_selected) < len(selected):
                    return new_selected
        return None

    best_code = None
    best_size = N + 1
    start_time = time.time()
    time_limit = 110.0
    
    initial_selected = greedy_cover(all_ones_mask)
    initial_selected = remove_redundants(initial_selected)
    
    if len(initial_selected) < best_size:
        best_size = len(initial_selected)
        best_code = [all_words[i] for i in initial_selected]
        
    current_selected = initial_selected[:]
    
    while time.time() - start_time < time_limit:
        improved = False
        new_sel = local_search_2_1(current_selected)
        if new_sel:
            current_selected = new_sel
            improved = True
            if len(current_selected) < best_size:
                best_size = len(current_selected)
                best_code = [all_words[i] for i in current_selected]
        
        if improved:
            continue
            
        # Perturb
        perturb_amount = random.randint(2, 4)
        if len(current_selected) <= perturb_amount:
            current_selected = greedy_cover(all_ones_mask)
            current_selected = remove_redundants(current_selected)
        else:
            indices_to_remove = random.sample(range(len(current_selected)), perturb_amount)
            indices_to_remove_set = set(indices_to_remove)
            covered_mask = 0
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    covered_mask |= balls[current_selected[i]]
            uncovered_mask = all_ones_mask & ~covered_mask
            repair_selected = greedy_cover(uncovered_mask)
            new_selected = []
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    new_selected.append(current_selected[i])
            new_selected.extend(repair_selected)
            new_selected = remove_redundants(new_selected)
            if len(new_selected) < best_size:
                best_size = len(new_selected)
                best_code = [all_words[i] for i in new_selected]
            if len(new_selected) <= len(current_selected) or (random.random() < 0.05 and len(new_selected) < len(current_selected) + 3):
                current_selected = new_selected

    return best_code
