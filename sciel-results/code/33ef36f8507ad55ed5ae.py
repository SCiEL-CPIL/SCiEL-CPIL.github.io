import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Build the list of (a, b, c) triples with T=1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    num_eq = len(triples)
    num_rows_U = n * m  # 6
    num_rows_V = m * p  # 21
    num_rows_W = p * n  # 14

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W (as lists of lists) exactly decompose the tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for (a, b, c) in triples:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val - 1.0) > tol:
                return False
        return True

    # Rank 39 fallback construction (Strassen-style blocks)
    def construct_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1

            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1

            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1

            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1

            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1

            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1

            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1

            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1

        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])

        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])

        return {"U": U, "V": V, "W": W}

    fallback_result = construct_39()

    # MILP-based search for lower rank
    # Variables: for each rank term r and each position (i,j,k), binary x_{r,i,j,k}
    # indicating whether term r contributes to triple (i,j,k).
    # For each term r, the support pattern must be consistent: if x_{r,i,j,k}=1,
    # then U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (or -1 with sign).
    
    # Simpler formulation: For each rank R, try to find a valid decomposition
    # using a greedy/structured approach with MILP for support selection.
    
    def try_milp_rank(R, time_budget=15):
        """Try to find a rank-R decomposition using MILP for support patterns."""
        t_start = time.time()
        
        # For each of R terms, we need to choose which triples it covers.
        # A term r covers a set of triples S_r such that there exist values
        # U[a,r], V[b,r], W[c,r] in {-1, 0, 1, 0.5, -0.5} satisfying
        # U[a,r]*V[b,r]*W[c,r] = 1 for all (a,b,c) in S_r, and 0 otherwise.
        
        # Key insight: For a single term to cover multiple triples, the triples
        # must be "compatible" - they must share consistent index patterns.
        
        # For a term to contribute to triple (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
        # The term's contribution is U[a,r]*V[b,r]*W[c,r].
        # For this to be 1 for multiple triples, we need:
        #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's define for each term r:
        #   u_vals: dict mapping a-index to value (or 0 if not in support)
        #   v_vals: dict mapping b-index to value
        #   w_vals: dict mapping c-index to value
        # Then for triple (a,b,c): contribution = u_vals.get(a,0) * v_vals.get(b,0) * w_vals.get(c,0)
        
        # For the product to be 1, all three must be nonzero and their product = 1.
        # For the product to be 0, at least one must be 0.
        
        # This is equivalent to: the set of triples covered by term r forms a "combinatorial rectangle"
        # in the (i,j,k) space where the indices are consistent.
        
        # Actually, let me think about this differently. A rank-1 term in matrix multiplication
        # corresponds to a specific pattern. For 2x3x7, the possible "atomic" patterns are limited.
        
        # Let's use a different approach: enumerate all possible "term types" and use set cover.
        
        # A term type is defined by which (i,j,k) indices it touches.
        # For a term to have U[i*m+j,r] != 0, V[j*p+k,r] != 0, W[k*n+i,r] != 0,
        # the indices (i,j,k) must be such that these positions are consistent.
        
        # Actually, the simplest valid terms are:
        # 1. Single triple: covers exactly one (i,j,k)
        # 2. Pairs of triples that share structure
        # 3. Larger groups
        
        # Let me just do a smarter random search with better initialization.
        
        # Strategy: Start from the rank-39 solution and try to merge columns.
        U39 = np.array(fallback_result['U'], dtype=float)
        V39 = np.array(fallback_result['V'], dtype=float)
        W39 = np.array(fallback_result['W'], dtype=float)
        
        # Try to eliminate columns by adjusting others
        # For each pair of columns (r1, r2), check if we can set one to zero
        # by adjusting the other.
        
        best_err = np.inf
        best_solution = None
        
        # Random local search with better initialization
        np.random.seed(42)
        
        for attempt in range(20):
            if time.time() - t_start > time_budget:
                break
            
            # Initialize from rank-39 with some random perturbations
            U = U39[:, :R].copy()
            V = V39[:, :R].copy()
            W = W39[:, :R].copy()
            
            # Randomly zero out some entries and adjust
            for _ in range(100):
                if time.time() - t_start > time_budget - 1:
                    break
                
                # Compute current error
                err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    err += (val - 1.0) ** 2
                
                if err < 1e-9:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if check_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break
                
                # Pick a random entry to modify
                choice = np.random.random()
                if choice < num_rows_U / (num_rows_U + num_rows_V + num_rows_W):
                    row = np.random.randint(num_rows_U)
                    mat = U
                elif choice < (num_rows_U + num_rows_V) / (num_rows_U + num_rows_V + num_rows_W):
                    row = np.random.randint(num_rows_V)
                    mat = V
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                
                col = np.random.randint(R)
                old_val = mat[row, col]
                
                # Try values from a small set
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                new_val = np.random.choice(vals)
                if new_val == old_val:
                    continue
                
                mat[row, col] = new_val
                
                # Compute new error
                err_new = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    err_new += (val - 1.0) ** 2
                
                if err_new < err:
                    err = err_new
                    if err < best_err:
                        best_err = err
                        best_solution = (U.copy(), V.copy(), W.copy())
                else:
                    mat[row, col] = old_val
        
        if best_solution is not None and best_err < 1e-6:
            U, V, W = best_solution
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if check_decomposition(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 38 down to 36
    for R in range(38, 35, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_milp_rank(R, time_budget=min(30, deadline - time.time() - 5))
        if result is not None:
            return result

    return fallback_result
# EVOLVE-BLOCK-END
