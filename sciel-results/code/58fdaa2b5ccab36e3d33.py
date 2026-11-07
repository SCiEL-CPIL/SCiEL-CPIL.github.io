"""ILP-based construction for large integer sets with no two-other-product divisibility."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    start_time = time.time()
    
    all_nums = list(range(2, 257))
    n = len(all_nums)
    num_to_idx = {x: i for i, x in enumerate(all_nums)}
    
    # Precompute divisibility: div[a][b] = True if a divides b
    # a divides b means b % a == 0
    div = [[False] * n for _ in range(n)]
    for i, a in enumerate(all_nums):
        for j, b in enumerate(all_nums):
            if b % a == 0:
                div[i][j] = True
    
    # For each pair (b, c) with b != c, find all a that divide b*c
    # We need: for every triple (a, b, c) distinct, NOT (a | b*c)
    # Equivalent: for every triple (a, b, c) distinct, at least one of a, b, c is excluded
    # i.e., x_a + x_b + x_c <= 2 for all distinct a, b, c where a | b*c
    
    # Precompute conflict triples: list of (i, j, k) where i|j*k, i!=j, i!=k, j!=k
    # But this is too many. Instead, let's think differently.
    
    # Actually, the constraint is: for all distinct a, b, c in S, a does not divide b*c.
    # This is equivalent to: for all distinct a, b, c, it is NOT the case that (a | b*c AND all in S).
    # So for each triple (a, b, c) with a | b*c, we need x_a + x_b + x_c <= 2.
    
    # Let's enumerate all such triples. For a in 2..256, b in 2..256, c in 2..256,
    # a != b, a != c, b != c, and b*c % a == 0.
    # This is O(n^3) = 255^3 ≈ 16.6M, which is feasible.
    
    # But storing all constraints might be large. Let's estimate:
    # For each a, how many (b,c) pairs with b*c % a == 0?
    # It's roughly (255/a)^2 * 255 or so. Total could be a few million.
    
    # Let's build the constraint matrix in a sparse format for scipy.optimize.milp.
    
    # Variables: x_i for i in 0..n-1, binary.
    # Objective: maximize sum(x_i)
    # Constraints: for each conflict triple (i, j, k): x_i + x_j + x_k <= 2
    
    # Let's collect all conflict triples.
    conflict_triples = []
    for i, a in enumerate(all_nums):
        for j, b in enumerate(all_nums):
            if i == j:
                continue
            for k, c in enumerate(all_nums):
                if i == k or j == k:
                    continue
                if (b * c) % a == 0:
                    conflict_triples.append((i, j, k))
    
    # This might be too many. Let's count first.
    # Actually, let's be smarter. For fixed a and b, which c satisfy b*c % a == 0?
    # b*c % a == 0 iff c % (a / gcd(a, b)) == 0.
    # Let g = gcd(a, b), then a/g must divide c.
    # So c must be a multiple of a/g.
    
    # Let's rebuild more efficiently.
    conflict_triples = []
    for i, a in enumerate(all_nums):
        for j, b in enumerate(all_nums):
            if i == j:
                continue
            g = gcd(a, b)
            m = a // g  # c must be a multiple of m
            # c in 2..256, c != a, c != b, c is multiple of m
            for c_val in range(m, 257, m):
                if c_val < 2:
                    continue
                k = num_to_idx[c_val]
                if i == k or j == k:
                    continue
                conflict_triples.append((i, j, k))
    
    # Now we have all conflict triples. Let's see how many.
    # This could still be large. Let's proceed with scipy.optimize.milp.
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Number of variables
        n_vars = n
        
        # Objective: maximize sum(x_i) => minimize -sum(x_i)
        c_obj = -np.ones(n_vars)
        
        # Constraints: for each conflict triple (i, j, k): x_i + x_j + x_k <= 2
        # This is a linear constraint: x_i + x_j + x_k <= 2
        # In matrix form: A @ x <= b
        
        if len(conflict_triples) == 0:
            # No constraints, take all
            return list(all_nums)
        
        m_cons = len(conflict_triples)
        
        # Build sparse matrix A
        from scipy.sparse import lil_matrix, csr_matrix
        
        A = lil_matrix((m_cons, n_vars), dtype=np.int8)
        b_ub = np.full(m_cons, 2, dtype=np.int8)
        
        for row, (i, j, k) in enumerate(conflict_triples):
            A[row, i] = 1
            A[row, j] = 1
            A[row, k] = 1
        
        A = A.tocsr()
        
        # Bounds: 0 <= x_i <= 1
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        # Integer: all binary
        integrality = np.ones(n_vars, dtype=np.int8)
        
        # Solve
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A, ub=b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.success and res.x is not None:
            x = res.x
            # x should be close to 0 or 1
            result_idx = [i for i in range(n_vars) if x[i] > 0.5]
            result = [all_nums[i] for i in result_idx]
            
            # Validate
            result_set = set(result)
            result_list = list(result_set)
            valid = True
            n_res = len(result_list)
            for i in range(n_res):
                a = result_list[i]
                for j in range(n_res):
                    if i == j:
                        continue
                    b = result_list[j]
                    for k in range(n_res):
                        if i == k or j == k:
                            continue
                        c = result_list[k]
                        if (b * c) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return [int(x) for x in result]
        else:
            # Fallback to greedy
            pass
    
    except Exception:
        pass
    
    # Fallback: greedy construction
    def _can_add(S_list, S_set, x):
        n_s = len(S_list)
        # 1. x divides a*b for distinct a,b in S
        for i in range(n_s):
            a = S_list[i]
            for j in range(i+1, n_s):
                b = S_list[j]
                if (a * b) % x == 0:
                    return False
        # 2. a divides x*b for distinct a,b in S
        for i in range(n_s):
            a = S_list[i]
            for j in range(n_s):
                if i == j:
                    continue
                b = S_list[j]
                if (x * b) % a == 0:
                    return False
        return True
    
    def is_prime(n_val):
        if n_val < 2: return False
        if n_val == 2: return True
        if n_val % 2 == 0: return False
        for i in range(3, int(n_val**0.5)+1, 2):
            if n_val % i == 0: return False
        return True
    
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    for _ in range(30):
        random.shuffle(all_nums)
        orders.append(all_nums.copy())
    
    best_set = set()
    best_size = 0
    
    for order in orders:
        if time.time() > deadline:
            break
        S = set()
        S_list = []
        for x in order:
            if time.time() > deadline:
                break
            if _can_add(S_list, S, x):
                S.add(x)
                S_list.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
