import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Specific to the task: q=3, n=8, radius=1
    # Total words = 3^8 = 6561
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh
        
    # Build the incidence matrix A (total_words x total_words)
    # A[i, j] = 1 if word i is covered by codeword j (i.e., i in neighbors[j])
    # Note: neighbors[j] contains all words covered by j.
    # So for each j, for each i in neighbors[j], A[i, j] = 1.
    
    # Using sparse matrix for efficiency? SciPy milp accepts sparse or dense.
    # 6561x6561 is ~43M floats. 344MB. Might be tight but okay.
    # Let's use a sparse representation if possible, or just build it.
    # Actually, milp in SciPy 1.13+ supports sparse matrices for A.
    # Let's try to build a sparse matrix to be safe on memory and speed.
    
    from scipy.sparse import lil_matrix, csr_matrix
    
    # A is row=point, col=codeword
    A = lil_matrix((total_words, total_words), dtype=np.int8)
    
    for j in range(total_words):
        for i in neighbors[j]:
            A[i, j] = 1
            
    A = A.tocsr()
    
    # Objective: minimize sum of x_j
    c = np.ones(total_words, dtype=np.float64)
    
    # Constraints: A @ x >= 1 (every point covered)
    # milp requires A @ x <= b or A @ x >= b or ==
    # We can use LinearConstraint(A, lb=1, ub=np.inf)
    # Wait, milp interface:
    # A: 2D array
    # c: 1D array
    # integrality: array of 0, 1, -1
    # bounds: Bounds(lb, ub)
    # constraints: list of LinearConstraint
    
    integrality = np.ones(total_words, dtype=np.int8) # all binary
    
    bounds = Bounds(lb=0, ub=1)
    
    # Constraint: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(total_words), ub=np.inf)
    # Note: For sparse A, this should work.
    
    constraint = LinearConstraint(A, lb=np.ones(total_words, dtype=np.float64), ub=np.inf)
    
    # Time limit for solver
    # SciPy milp options for HiGHS:
    # time_limit: float
    # We have about 115s total. Precomputation took some time.
    # Let's check time.
    time_elapsed = time.time() - start_time
    remaining_time = deadline - time_elapsed - 5.0 # 5s margin
    
    if remaining_time < 10.0:
        # Fallback to greedy if not enough time for ILP setup/solve
        # Simple greedy
        cov_count = np.zeros(total_words, dtype=np.int32)
        is_code = np.zeros(total_words, dtype=bool)
        code_list = []
        potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        uncovered_remaining = total_words
        
        while uncovered_remaining > 0 and time.time() < deadline - 2.0:
            c_idx = np.argmax(potential)
            if potential[c_idx] == 0:
                break
            code_list.append(c_idx)
            is_code[c_idx] = True
            potential[c_idx] = 0
            
            for x in neighbors[c_idx]:
                if cov_count[x] == 0:
                    cov_count[x] = 1
                    uncovered_remaining -= 1
                    potential[neighbors[x]] -= 1
                else:
                    cov_count[x] += 1
                    
        # Verify and return
        if np.all(cov_count > 0):
            return [list(digits_of[i]) for i in code_list]
        else:
            # Emergency fallback: all words
            return [list(digits_of[i]) for i in range(total_words)]

    try:
        # Run ILP
        # HiGHS options
        options = {
            "time_limit": remaining_time,
            "presolve": True,
            "mip_rel_gap": 0.0 # We want exact if possible, but might take too long.
                               # Actually, for covering codes, finding ANY valid solution is the first goal.
                               # Then minimize.
                               # If we set gap > 0, it might stop early with a suboptimal but valid solution.
                               # Let's allow a small gap to ensure we get a solution within time.
                               # But the score is the number of codewords.
                               # If we don't find the optimum, we return the incumbent.
                               # The incumbent is always valid.
                               # So we just need to run as long as possible.
        }
        
        # Note: milp returns a result object.
        # res.x contains the solution.
        # res.status: 0 is optimal, 1 is feasible (incumbent found), etc.
        
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds, options=options)
        
        if res.status == 0 or res.status == 1:
            # res.status 0: Optimal
            # res.status 1: Feasible (incumbent)
            # res.x is the solution vector
            x_sol = res.x
            code_indices = [i for i in range(total_words) if x_sol[i] > 0.5]
            
            # Verify validity
            cov_count = np.zeros(total_words, dtype=np.int32)
            for idx in code_indices:
                cov_count[neighbors[idx]] += 1
                
            if np.all(cov_count > 0):
                return [list(digits_of[i]) for i in code_indices]
            else:
                # If ILP returned something invalid (shouldn't happen with >= 1 constraint), fallback
                pass
        else:
            # No feasible solution found? Should not happen as all-words is feasible.
            # Maybe timeout before finding any?
            # Fallback to greedy
            pass
            
    except Exception as e:
        # Fallback to greedy on error
        pass

    # Fallback Greedy if ILP failed or didn't return valid
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words
    
    while uncovered_remaining > 0 and time.time() < deadline - 2.0:
        c_idx = np.argmax(potential)
        if potential[c_idx] == 0:
            break
        code_list.append(c_idx)
        is_code[c_idx] = True
        potential[c_idx] = 0
        
        for x in neighbors[c_idx]:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1
                
    # Verify
    if np.all(cov_count > 0):
        return [list(digits_of[i]) for i in code_list]
    else:
        # Last resort
        return [list(digits_of[i]) for i in range(total_words)]
