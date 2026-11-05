import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) with matching j
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def compute_error(U, V, W):
        X = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((X - T) ** 2)

    def is_valid(U, V, W):
        # Check if entries are half-integers and decomposition is exact
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        X = np.einsum('ar,br,cr->abc', U, V, W)
        return np.array_equal(X, T)

    # Strassen-based block decomposition for 3x5x7
    # Split: A is 3x5, B is 5x7
    # Use blocks: A = [A1 (3x2) | A2 (3x3)], B = [B1 (2x7); B2 (3x7)]
    # Or better: use Strassen on 2x2 blocks where possible
    
    def get_strassen_decomposition():
        """
        Build a Strassen-inspired decomposition for 3x5x7.
        
        Strategy: Partition the computation into blocks and apply Strassen 
        where 2x2x2 sub-problems exist.
        
        A (3x5) * B (5x7) = C (3x7)
        
        Let's partition:
        - A = [A_top (2x5); A_bottom (1x5)]  -- or other splits
        - Actually, let's try: A rows split 2+1, A cols split 2+3
        - B rows split 2+3, B cols split 3+4
        
        Better approach: Use the known result that rank(3x5x7) <= 99 
        via a specific block decomposition.
        
        Let me construct it explicitly:
        Split A as 3x5 into blocks, B as 5x7 into blocks.
        
        One effective split:
        A = [A1 (2x2) A2 (2x3); A3 (1x2) A4 (1x3)]
        B = [B1 (2x3) B2 (2x4); B3 (3x3) B4 (3x4)]
        
        C = A*B has blocks:
        C11 = A1*B1 + A2*B3 (2x3)
        C12 = A1*B2 + A2*B4 (2x4)
        C21 = A3*B1 + A4*B3 (1x3)
        C22 = A3*B2 + A4*B4 (1x4)
        
        For C11 (2x3): can use standard 2x3 = 6 mults or Strassen-like
        For C12 (2x4): standard 8 mults
        For C21 (1x3): 3 mults
        For C22 (1x4): 4 mults
        Total standard: 6+8+3+4 = 21... that's not right, we need to count the intermediate products.
        
        Actually, the rank counts the number of bilinear forms (scalar multiplications in the algorithm).
        
        Let me use a different known construction. The key insight from the inspiration is 
        that Strassen on 2x2 sub-blocks reduces from 105 to 99.
        
        Let me construct a specific decomposition:
        
        Partition:
        A: rows [0,1] and [2], cols [0,1] and [2,3,4]
        B: rows [0,1] and [2,3,4], cols [0,1,2] and [3,4,5,6]
        
        So:
        A11 = A[0:2, 0:2] (2x2)
        A12 = A[0:2, 2:5] (2x3)
        A21 = A[2:3, 0:2] (1x2)
        A22 = A[2:3, 2:5] (1x3)
        
        B11 = B[0:2, 0:3] (2x3)
        B12 = B[0:2, 3:7] (2x4)
        B21 = B[2:5, 0:3] (3x3)
        B22 = B[2:5, 3:7] (3x4)
        
        C11 = A11*B11 + A12*B21 (2x3)
        C12 = A11*B12 + A12*B22 (2x4)
        C21 = A21*B11 + A22*B21 (1x3)
        C22 = A21*B12 + A22*B22 (1x4)
        
        Now, for C11 (2x3 product):
        Standard: 6 multiplications
        Can we do better? A 2x3 matrix product has rank 6 (it's 2*3=6 independent bilinear terms).
        So no improvement there.
        
        For C12 (2x4): rank 8
        For C21 (1x3): rank 3
        For C22 (1x4): rank 4
        
        Total: 6+8+3+4 = 21? That doesn't seem right for rank 99.
        
        Wait, I'm confusing myself. The "rank" in tensor decomposition is the number of 
        rank-1 tensors needed. For standard matrix multiplication of m x n by n x p, 
        the rank is m*n*p.
        
        The Strassen trick works when you have a 2x2x2 subproblem. Let me reconsider.
        
        Actually, let me just use a simpler approach: start with schoolbook and try to 
        merge/eliminate some terms using Strassen-like identities.
        
        For now, let me create a valid decomposition at rank 105 (schoolbook) and 
        then try local search to reduce it.
        """
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_strassen_99():
        """
        Construct a rank-99 decomposition using Strassen on a 2x2x2 sub-block.
        
        The idea: In the schoolbook algorithm for 3x5x7, we have 105 terms.
        If we can identify a 2x2x2 subproblem and replace its 8 terms with 
        Strassen's 7 terms, we get 105 - 8 + 7 = 104.
        
        To get to 99, we need to find more such opportunities or use a 
        more sophisticated block decomposition.
        
        Let me try a specific construction:
        
        Split the indices as follows:
        i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
        
        Consider the subproblem where i in {0,1}, j in {0,1}, k in {0,1}.
        This is a 2x2x2 subproblem with 8 schoolbook terms.
        Replace with Strassen's 7 terms.
        
        Strassen for 2x2:
        M1 = (A11+A22)*(B11+B22)
        M2 = (A21+A22)*B11
        M3 = A11*(B12-B22)
        M4 = A22*(B21-B11)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*(B11+B12)
        M7 = (A12-A22)*(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        In our tensor notation:
        For the subproblem i in {0,1}, j in {0,1}, k in {0,1}:
        
        Schoolbook terms (8):
        (i,j,k) for i=0,1; j=0,1; k=0,1
        
        A indices: a = i*5 + j
        B indices: b = j*7 + k
        C indices: c = k*3 + i
        
        Let me enumerate:
        i=0,j=0,k=0: a=0, b=0, c=0
        i=0,j=0,k=1: a=0, b=1, c=1
        i=0,j=1,k=0: a=1, b=7, c=0
        i=0,j=1,k=1: a=1, b=8, c=1
        i=1,j=0,k=0: a=5, b=0, c=3
        i=1,j=0,k=1: a=5, b=1, c=4
        i=1,j=1,k=0: a=6, b=7, c=3
        i=1,j=1,k=1: a=6, b=8, c=4
        
        Strassen replacement:
        M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
           = (a=0 + a=6) * (b=0 + b=8)
           Contributes to: C[0,0], C[0,1], C[1,0], C[1,1] (in the subblock)
           = c=0, c=1, c=3, c=4
        
        Let me be more careful. In the 2x2x2 subproblem:
        A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[a=0, a=1], [a=5, a=6]]
        B_sub = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]] = [[b=0, b=1], [b=7, b=8]]
        C_sub = [[C[0,0], C[0,1]], [C[1,0], C[1,1]]] = [[c=0, c=1], [c=3, c=4]]
        
        Strassen:
        M1 = (A00+A11)*(B00+B11) = (a0+a6)*(b0+b8)
        M2 = (A10+A11)*B00 = (a5+a6)*b0
        M3 = A00*(B01-B11) = a0*(b1-b8)
        M4 = A11*(B10-B00) = a6*(b7-b0)
        M5 = (A00+A01)*B11 = (a0+a1)*b8
        M6 = (A10-A00)*(B00+B01) = (a5-a0)*(b0+b1)
        M7 = (A01-A11)*(B10+B11) = (a1-a6)*(b7+b8)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        Now, each M is a product of a linear combination of A entries and a linear 
        combination of B entries. In the tensor decomposition, each M corresponds to 
        one rank-1 term where:
        - U has the coefficients of the A linear combination
        - V has the coefficients of the B linear combination
        - W has the coefficients indicating which C entries it contributes to
        
        Let me build the full 99-rank decomposition:
        - Start with all 105 schoolbook terms
        - Remove the 8 terms for the 2x2x2 subblock (i in {0,1}, j in {0,1}, k in {0,1})
        - Add 7 Strassen terms for that subblock
        - Net: 105 - 8 + 7 = 104
        
        To get to 99, I need to find 5 more such replacements or a larger block structure.
        
        Let me try another 2x2x2 subblock: i in {0,1}, j in {2,3}, k in {2,3}
        A entries: a=2,3,7,8
        B entries: b=14,15,21,22
        C entries: c=6,7,9,10
        
        And another: i in {0,1}, j in {3,4}, k in {4,5}
        A entries: a=3,4,8,9
        B entries: b=21,22,28,29
        C entries: c=12,13,15,16
        
        Each gives -1 reduction. So 104 - 3 = 101. Still not 99.
        
        Let me try: i in {1,2}, j in {0,1}, k in {0,1}
        A entries: a=5,6,10,11
        B entries: b=0,1,7,8
        C entries: c=3,4,6,7
        
        Another: i in {1,2}, j in {2,3}, k in {2,3}
        A entries: a=7,8,12,13
        B entries: b=14,15,21,22
        C entries: c=9,10,12,13
        
        That's 5 total Strassen replacements: 105 - 5 = 100.
        
        Hmm, still not 99. Let me check if there's a way to get 99.
        
        Actually, maybe the known result of 99 uses a different technique. Let me just 
        implement the 104-rank version and then use local search to try to reduce further.
        """
        # Start with schoolbook
        U = np.zeros((n * m, n * m * p), dtype=np.float64)
        V = np.zeros((m * p, n * m * p), dtype=np.float64)
        W = np.zeros((p * n, n * m * p), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        
        # Now apply Strassen to 2x2x2 subblocks
        # Subblock 1: i in {0,1}, j in {0,1}, k in {0,1}
        # Remove terms: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        # These are at columns 0,1,2,3,4,5,6,7 (since we iterate i,j,k in order)
        
        # Actually, let me rebuild more carefully.
        # I'll create a list of (i,j,k) tuples and their column indices.
        
        terms = []
        col_map = {}
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
                    col_map[(i, j, k)] = col
                    col += 1
        
        # Identify 2x2x2 subblocks to apply Strassen to
        # Subblock defined by (i_set, j_set, k_set) where each has size 2
        subblocks = [
            ({0, 1}, {0, 1}, {0, 1}),
            ({0, 1}, {2, 3}, {2, 3}),
            ({1, 2}, {0, 1}, {0, 1}),
            ({1, 2}, {2, 3}, {2, 3}),
            ({0, 1}, {3, 4}, {4, 5}),
        ]
        
        # For each subblock, replace 8 terms with 7 Strassen terms
        # Build new U, V, W matrices
        
        new_U_list = []
        new_V_list = []
        new_W_list = []
        
        # First, add all non-subblock terms
        subblock_terms = set()
        for i_set, j_set, k_set in subblocks:
            for i in i_set:
                for j in j_set:
                    for k in k_set:
                        subblock_terms.add((i, j, k))
        
        for (i, j, k) in terms:
            if (i, j, k) not in subblock_terms:
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                u_vec[a] = 1
                v_vec[b] = 1
                w_vec[c] = 1
                new_U_list.append(u_vec)
                new_V_list.append(v_vec)
                new_W_list.append(w_vec)
        
        # Now add Strassen terms for each subblock
        for i_set, j_set, k_set in subblocks:
            i0, i1 = sorted(i_set)
            j0, j1 = sorted(j_set)
            k0, k1 = sorted(k_set)
            
            # A indices
            a00 = i0 * m + j0
            a01 = i0 * m + j1
            a10 = i1 * m + j0
            a11 = i1 * m + j1
            
            # B indices
            b00 = j0 * p + k0
            b01 = j0 * p + k1
            b10 = j1 * p + k0
            b11 = j1 * p + k1
            
            # C indices
            c00 = k0 * n + i0
            c01 = k1 * n + i0
            c10 = k0 * n + i1
            c11 = k1 * n + i1
            
            # Strassen M1 = (A00+A11)*(B00+B11)
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[a00] = 1; u[a11] = 1
            v[b00] = 1; v[b11] = 1
            w[c00] = 1; w[c01] = 1; w[c10] = 1; w[c11] = 1
            new_U_list.append(u); new_V_list.append(v); new_W_list.append(w)
            
            # M2 = (A10+A11)*B00
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[a10] = 1; u[a11] = 1
            v[b00] = 1
            w[c10] = 1; w[c11] = 1
            new_U_list.append(u); new_V_list.append(v); new_W_list.append(w)
            
            # M3 = A00*(B01-B11)
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[a00] = 1
            v[b01] = 1; v[b11] = -1
            w[c01] = 1; w[c11] = 1
            new_U_list.append(u); new_V_list.append(v); new_W_list.append(w)
            
            # M4 = A11*(B10-B00)
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[a11] = 1
            v[b10] = 1; v[b00] = -1
            w[c00] = 1; w[c10] = 1
            new_U_list.append(u); new_V_list.append(v); new_W_list.append(w)
            
            # M5 = (A00+A01)*B11
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[a00] = 1; u[a01] = 1
            v[b11] = 1
            w[c00] = -1; w[c01] = 1
            new_U_list.append(u); new_V_list.append(v); new_W_list.append(w)
            
            # M6 = (A10-A00)*(B00+B01)
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[a10] = 1; u[a00] = -1
            v[b00] = 1; v[b01] = 1
            w[c11] = 1
            new_U_list.append(u); new_V_list.append(v); new_W_list.append(w)
            
            # M7 = (A01-A11)*(B10+B11)
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[a01] = 1; u[a11] = -1
            v[b10] = 1; v[b11] = 1
            w[c00] = 1; w[c11] = 1
            new_U_list.append(u); new_V_list.append(v); new_W_list.append(w)
        
        U_out = np.column_stack(new_U_list)
        V_out = np.column_stack(new_V_list)
        W_out = np.column_stack(new_W_list)
        
        return U_out, V_out, W_out

    # Get the Strassen-based decomposition
    U_str, V_str, W_str = get_strassen_99()
    current_rank = U_str.shape[1]
    
    best_rank = current_rank
    best_U, best_V, best_W = U_str, V_str, W_str
    
    # Verify it's valid
    if is_valid(U_str, V_str, W_str):
        print(f"Strassen decomposition valid with rank {current_rank}")
    else:
        print(f"Strassen decomposition NOT valid, using schoolbook")
        U_sch, V_sch, W_sch = get_strassen_decomposition()
        best_U, best_V, best_W = U_sch, V_sch, W_sch
        best_rank = n * m * p
    
    # Now try local search to reduce rank
    # Strategy: try to merge/eliminate columns by finding linear dependencies
    
    def try_reduce_rank(U, V, W, max_time=30):
        """Try to reduce the rank by finding and eliminating redundant columns."""
        start = time.time()
        U = U.copy()
        V = V.copy()
        W = W.copy()
        current_rank = U.shape[1]
        
        # Try removing columns one at a time and see if we can compensate
        # This is a heuristic approach
        
        for attempt in range(100):
            if time.time() - start > max_time:
                break
            if current_rank <= 40:
                break
            
            # Pick a random column to try to eliminate
            col_to_remove = random.randint(0, current_rank - 1)
            
            # Save the column
            u_col = U[:, col_to_remove].copy()
            v_col = V[:, col_to_remove].copy()
            w_col = W[:, col_to_remove].copy()
            
            # Remove it
            U_temp = np.delete(U, col_to_remove, axis=1)
            V_temp = np.delete(V, col_to_remove, axis=1)
            W_temp = np.delete(W, col_to_remove, axis=1)
            
            # Check if the decomposition still holds
            # The removed term contributed: u_col * v_col * w_col (outer product)
            # We need to see if this can be represented by the remaining terms
            # This is hard to check directly, so let's just verify the full decomposition
            
            X = np.einsum('ar,br,cr->abc', U_temp, V_temp, W_temp)
            if np.array_equal(X, T):
                # We can remove this column!
                U = U_temp
                V = V_temp
                W = W_temp
                current_rank -= 1
                continue
            
            # Try modifying nearby columns to compensate
            # Pick a random other column and adjust it
            other_col = random.randint(0, current_rank - 1)
            if other_col == col_to_remove:
                continue
            
            # Try adding the removed term's contribution to another column
            # This won't work directly since we need exact match
            
            # Alternative: try a local perturbation of the removed column's values
            # and see if we can make the decomposition work with fewer terms
            
            # For now, just skip this attempt
            pass
        
        return U, V, W, current_rank
    
    # Try to reduce the rank
    if time.time() < deadline - 20:
        U_red, V_red, W_red, red_rank = try_reduce_rank(best_U, best_V, best_W, max_time=30)
        if red_rank < best_rank and is_valid(U_red, V_red, W_red):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = red_rank
            print(f"Reduced rank to {best_rank}")
    
    # Try local search on the current best to see if we can find a lower rank
    # by modifying entries (keeping rank fixed or trying slightly lower ranks)
    
    def local_search_fixed_rank(U_init, V_init, W_init, target_rank, max_iters=1000):
        """Local search keeping rank fixed, trying to find exact decomposition."""
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        current_err = compute_error(U, V, W)
        if current_err == 0:
            return U, V, W, 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry to perturb
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
            
            i, j = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
            old_val = mat[i, j]
            
            # Propose a new value from a small set
            new_val = random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            if new_val == old_val:
                continue
                
            mat[i, j] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
            else:
                mat[i, j] = old_val
            
            if current_err == 0:
                return U, V, W, 0
        
        return U, V, W, current_err
    
    # Try to find a decomposition at lower ranks
    # Start from current best and try to go lower
    for target_rank in range(best_rank - 1, max(30, best_rank - 20), -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with a subset of the current best
        U_init = best_U[:, :target_rank]
        V_init = best_V[:, :target_rank]
        W_init = best_W[:, :target_rank]
        
        U_res, V_res, W_res, err = local_search_fixed_rank(U_init, V_init, W_init, target_rank, max_iters=200)
        
        if err == 0 and is_valid(U_res, V_res, W_res):
            best_U, best_V, best_W = U_res, V_res, W_res
            best_rank = target_rank
            print(f"Found valid decomposition at rank {best_rank}")
            break
    
    # Final validation
    if not is_valid(best_U, best_V, best_W):
        # Fall back to schoolbook
        U_sch, V_sch, W_sch = get_strassen_decomposition()
        best_U, best_V, best_W = U_sch, V_sch, W_sch
        best_rank = n * m * p
    
    # Ensure half-integer values
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    # Final check
    if not is_valid(best_U, best_V, best_W):
        # Use schoolbook as guaranteed fallback
        U_sch, V_sch, W_sch = get_strassen_decomposition()
        best_U, best_V, best_W = U_sch, V_sch, W_sch
        best_rank = n * m * p
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
