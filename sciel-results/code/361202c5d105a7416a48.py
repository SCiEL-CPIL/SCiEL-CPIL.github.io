import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint

def construct(q, n, radius):
    """
    Construct a q-ary covering code of length n and radius r.
    Uses Greedy Set Cover initialization, Simple Local Search, and MILP for refinement.
    """
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    N = q ** n  # 6561
    start_time = time.time()
    TIME_LIMIT = 115

    # Precompute neighbors
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i-1] * 3

    neighbors = [[] for _ in range(N)]
    for i in range(N):
        neighbors[i].append(i)
        w = i
        for p in range(n):
            p3 = pow3[p]
            digit = (w // p3) % 3
            base = w - digit * p3
            for new_d in range(3):
                if new_d == digit:
                    continue
                j = base + new_d * p3
                neighbors[i].append(j)

    # Precompute all codewords for final return
    all_words = [list(w) for w in itertools.product(range(3), repeat=8)]

    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    code_indices = []
    in_code = np.zeros(N, dtype=bool)
    
    # Gain calculation: number of uncovered neighbors
    gain = np.zeros(N, dtype=np.int32)
    for c in range(N):
        for j in neighbors[c]:
            if uncovered[j]:
                gain[c] += 1
                
    while np.any(uncovered):
        best_c = np.argmax(gain)
        if gain[best_c] == 0: break
        code_indices.append(best_c)
        in_code[best_c] = True
        for j in neighbors[best_c]:
            if uncovered[j]:
                uncovered[j] = False
                for c_prime in neighbors[j]:
                    gain[c_prime] -= 1
        if time.time() - start_time > TIME_LIMIT - 5:
            break

    # Ensure validity
    if not np.all(uncovered == False):
        for i in range(N):
            if uncovered[i]:
                code_indices.append(i)
                in_code[i] = True
                for j in neighbors[i]:
                    uncovered[j] = False

    # 2. Simple Local Search (Remove redundant words)
    count_cover = np.zeros(N, dtype=np.int32)
    for c in code_indices:
        for j in neighbors[c]:
            count_cover[j] += 1
            
    improved = True
    while improved and (time.time() - start_time < TIME_LIMIT - 20):
        improved = False
        for c in range(N):
            if not in_code[c]: continue
            removable = True
            for j in neighbors[c]:
                if count_cover[j] < 2:
                    removable = False
                    break
            if removable:
                in_code[c] = False
                for j in neighbors[c]:
                    count_cover[j] -= 1
                improved = True

    # Get current best solution
    best_solution = [i for i in range(N) if in_code[i]]
    best_size = len(best_solution)
    
    # 3. MILP Refinement
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5
    if remaining_time > 5:
        # Build sparse matrix A
        # A[j, i] = 1 if i in neighbors[j]
        row_idx = []
        col_idx = []
        data = []
        for j in range(N):
            for i in neighbors[j]:
                row_idx.append(j)
                col_idx.append(i)
                data.append(1)
        
        A = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
        
        # Objective
        c_obj = np.ones(N)
        
        # Bounds
        lb = np.zeros(N)
        ub = np.ones(N)
        
        # Integrality
        integrality = np.ones(N)
        
        # Constraints: A @ x >= 1
        constraints = LinearConstraint(A, 1, np.inf)
        
        try:
            res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=(lb, ub), options={'time_limit': remaining_time})
            
            # Check if we have a solution
            if res.x is not None:
                x = np.round(res.x).astype(int)
                milp_solution = [i for i in range(N) if x[i] == 1]
                
                # Verify validity of MILP solution
                if len(milp_solution) > 0:
                    covered = np.zeros(N, dtype=bool)
                    for c_idx in milp_solution:
                        for j in neighbors[c_idx]:
                            covered[j] = True
                    if np.all(covered):
                        if len(milp_solution) < best_size:
                            best_solution = milp_solution
                            best_size = len(milp_solution)
        except Exception:
            pass # Fallback to heuristic

    # Final construction
    return [all_words[i] for i in best_solution]
