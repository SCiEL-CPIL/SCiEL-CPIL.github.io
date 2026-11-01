import time
import numpy as np
import random
import scipy.sparse
import scipy.optimize

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Build balls array: balls[w] = [w, neighbors...]
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
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    # Try to solve using ILP if available
    try:
        # Construct sparse matrix A
        # A[j, i] = 1 if j is in balls[i]
        # We can build this by iterating over centers i and adding 1 to A[j, i] for j in balls[i]
        # Or iterate over j and add 1 to A[j, i] for i in inv[j]
        
        # Let's build row indices, col indices, data
        row_indices = []
        col_indices = []
        data = []
        
        # For each center i (column), it covers balls[i] (rows)
        for i in range(total):
            for j in balls[i]:
                row_indices.append(j)
                col_indices.append(i)
                data.append(1)
        
        A = scipy.sparse.csr_matrix(
            (data, (row_indices, col_indices)), 
            shape=(total, total), 
            dtype=np.float64
        )
        
        # ILP: min x^T * 1
        # s.t. A x >= 1
        # x in {0, 1}
        
        c = np.ones(total, dtype=np.float64)
        lb = np.zeros(total, dtype=np.float64)
        ub = np.ones(total, dtype=np.float64)
        
        # Constraints: A x >= 1
        # scipy.optimize.milp expects constraints in form A_ub x <= b_ub or A_eq x == b_eq
        # We have A x >= 1  =>  -A x <= -1
        A_ub = -A
        b_ub = -np.ones(total, dtype=np.float64)
        
        # Run MILP with time limit
        # options={'time_limit': 100.0}
        # Note: milp might not support time_limit directly in all versions, 
        # but HiGHS does. We can also just run it and hope it's fast.
        # 6561 vars, 6561 constraints, ~110k non-zeros. This is very small.
        # It should solve in seconds.
        
        res = scipy.optimize.milp(
            c=c,
            constraints=[scipy.optimize.LinearConstraint(A_ub, ub=b_ub)],
            integrality=np.ones(total, dtype=np.int8),
            bounds=[(0, 1)] * total
        )
        
        if res.success:
            x = res.x
            # Round to nearest integer
            x = np.round(x).astype(int)
            selected = np.where(x == 1)[0]
            
            # Verify validity
            cover_count = np.zeros(total, dtype=np.int32)
            for c_idx in selected:
                cover_count[balls[c_idx]] += 1
            if np.min(cover_count) >= 1:
                # Convert to list of digit lists
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
        
    except Exception as e:
        pass # Fallback to greedy

    # Fallback: Greedy + Local Search
    return _greedy_fallback(balls, total, n, deadline)

def _greedy_fallback(balls, total, n, deadline):
    # Reuse the logic from the previous program but optimized
    # This is a safety net
    import numpy as np
    
    # Build inverse mapping
    inv = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv[w].append(c)
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    def greedy_construct():
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=np.int32)
        
        while uncovered.sum() > 0:
            counts = np.zeros(total, dtype=np.int32)
            unc_idx = np.where(uncovered > 0)[0]
            if len(unc_idx) == 0:
                break
            for w in unc_idx:
                for c in inv[w]:
                    counts[c] += 1
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            for w in balls[best_idx]:
                cover_count[w] += 1
                if cover_count[w] == 1:
                    uncovered[w] = 0
            
            if time.time() > deadline - 10.0:
                break
        
        if uncovered.sum() > 0:
            return None
        return selected

    def prune(selected, cover_count):
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count

    while time.time() < deadline - 5.0:
        selected = greedy_construct()
        if selected is None:
            continue
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        selected, cover_count = prune(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        if time.time() > deadline - 8.0:
            break

    if best_set is None:
        best_set = list(range(total))
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
