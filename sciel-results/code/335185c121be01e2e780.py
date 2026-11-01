import time
import numpy as np
import itertools
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_lp(deadline)
    
    # Fallbacks
    if q == 3 and n == 4 and radius == 1:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_lp(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # LP Formulation
    # Minimize sum(x_i)
    # Subject to sum(x_j for j in ball(i)) >= 1 for all i
    # 0 <= x_i <= 1
    
    # A_ub: -A (since linprog handles <=, we use -A x <= -1)
    # A is incidence matrix: A[i, j] = 1 if j in ball(i)
    # We can build A row by row or use sparse matrix.
    # For linprog, we can pass a sparse matrix.
    import scipy.sparse as sp
    
    # Construct sparse A
    # Rows: 0..total-1 (constraints)
    # Cols: 0..total-1 (variables)
    # Non-zeros: for each i, balls[i] are the columns
    row_indices = []
    col_indices = []
    data = []
    
    for i in range(total):
        for j in balls[i]:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1.0)
            
    A_sparse = sp.csr_matrix((data, (row_indices, col_indices)), shape=(total, total))
    
    # c: cost vector (all 1s)
    c = np.ones(total)
    
    # b: RHS (all 1s)
    b = np.ones(total)
    
    # Bounds
    bounds = [(0, 1)] * total
    
    try:
        res = linprog(c, A_ub=-A_sparse, b_ub=-b, bounds=bounds, method='highs')
        if not res.success:
            # Fallback to greedy if LP fails
            return _construct_ternary_8_1_greedy(deadline, balls)
            
        x_opt = res.x
    except Exception:
        return _construct_ternary_8_1_greedy(deadline, balls)
        
    # Rounding: Sort by x_opt descending
    order = np.argsort(-x_opt)
    
    covered = np.zeros(total, dtype=bool)
    selected = []
    
    for i in order:
        if np.all(covered):
            break
        # If this word covers any uncovered point, add it
        # Check if any point in balls[i] is not covered
        if np.any(~covered[balls[i]]):
            selected.append(i)
            covered[balls[i]] = True
            
    # Post-processing: Remove redundant codewords
    # Count coverage
    cover_count = np.zeros(total, dtype=np.int32)
    for c_idx in selected:
        cover_count[balls[c_idx]] += 1
        
    new_selected = []
    for c_idx in selected:
        # If min coverage in its ball is > 1, it is redundant
        if np.min(cover_count[balls[c_idx]]) > 1:
            cover_count[balls[c_idx]] -= 1
        else:
            new_selected.append(c_idx)
            
    selected = new_selected
    
    # Final Verification
    final_cover = np.zeros(total, dtype=np.int32)
    for c_idx in selected:
        final_cover[balls[c_idx]] += 1
        
    if np.any(final_cover == 0):
        # Add missing
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1
            
    # Convert to lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result

def _construct_ternary_8_1_greedy(deadline, balls):
    # Fallback greedy if LP fails
    total = balls.shape[0]
    covered = np.zeros(total, dtype=bool)
    selected = []
    
    while np.any(~covered):
        counts = np.sum(~covered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        covered[balls[best_idx]] = True
        
        if time.time() > deadline - 5.0:
            break
            
    # Prune
    cover_count = np.zeros(total, dtype=np.int32)
    for c_idx in selected:
        cover_count[balls[c_idx]] += 1
    new_selected = []
    for c_idx in selected:
        if np.min(cover_count[balls[c_idx]]) > 1:
            cover_count[balls[c_idx]] -= 1
        else:
            new_selected.append(c_idx)
    selected = new_selected
    
    # Verify and fix
    final_cover = np.zeros(total, dtype=np.int32)
    for c_idx in selected:
        final_cover[balls[c_idx]] += 1
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1
            
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(8):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result
