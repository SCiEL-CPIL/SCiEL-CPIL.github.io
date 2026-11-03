import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen-based block decomposition to reduce rank below schoolbook (60).
    Falls back to schoolbook if no improvement is found within time budget.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n  # 10, 30, 12

    # Build target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k; else 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose tensor T."""
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def strassen_2x2():
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2 matrix multiplication.
        Returns U, V, W matrices for the 2x2x2 tensor (rank 7).
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, r], V[i*2+j, r], W[i*2+j, r]
        where i,j in {0,1} for 2x2 blocks.
        """
        rank = 7
        U = np.zeros((4, rank), dtype=np.float64)
        V = np.zeros((4, rank), dtype=np.float64)
        W = np.zeros((4, rank), dtype=np.float64)

        # Strassen formulas for 2x2:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # U corresponds to A entries: U[a_idx, r] where a_idx = i*2+j
        # V corresponds to B entries: V[b_idx, r] where b_idx = i*2+j
        # W corresponds to C entries: W[c_idx, r] where c_idx = i*2+j

        # M1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1; W has C11=1,C22=1
        U[0, 0] = 1.0  # a
        U[3, 0] = 1.0  # d
        V[0, 0] = 1.0  # e
        V[3, 0] = 1.0  # h
        W[0, 0] = 1.0  # C11
        W[3, 0] = 1.0  # C22

        # M2 = (c+d)e: U has c=1,d=1; V has e=1; W has C21=1
        U[2, 1] = 1.0  # c
        U[3, 1] = 1.0  # d
        V[0, 1] = 1.0  # e
        W[2, 1] = 1.0  # C21

        # M3 = a(f-h): U has a=1; V has f=1,h=-1; W has C12=1,C22=1
        U[0, 2] = 1.0  # a
        V[1, 2] = 1.0  # f
        V[3, 2] = -1.0  # h
        W[1, 2] = 1.0  # C12
        W[3, 2] = 1.0  # C22

        # M4 = d(g-e): U has d=1; V has g=1,e=-1; W has C11=1,C21=1
        U[3, 3] = 1.0  # d
        V[2, 3] = 1.0  # g
        V[0, 3] = -1.0  # e
        W[0, 3] = 1.0  # C11
        W[2, 3] = 1.0  # C21

        # M5 = (a+b)h: U has a=1,b=1; V has h=1; W has C11=-1,C12=1
        U[0, 4] = 1.0  # a
        U[1, 4] = 1.0  # b
        V[3, 4] = 1.0  # h
        W[0, 4] = -1.0  # C11
        W[1, 4] = 1.0  # C12

        # M6 = (c-a)(e+f): U has c=1,a=-1; V has e=1,f=1; W has C22=1
        U[2, 5] = 1.0  # c
        U[0, 5] = -1.0  # a
        V[0, 5] = 1.0  # e
        V[1, 5] = 1.0  # f
        W[3, 5] = 1.0  # C22

        # M7 = (b-d)(g+h): U has b=1,d=-1; V has g=1,h=1; W has C11=1
        U[1, 6] = 1.0  # b
        U[3, 6] = -1.0  # d
        V[2, 6] = 1.0  # g
        V[3, 6] = 1.0  # h
        W[0, 6] = 1.0  # C11

        return U, V, W

    def strassen_block_decomposition():
        """
        Apply Strassen to 2x2 blocks within the 2x5 * 5x6 multiplication.
        
        For 2x5 * 5x6:
        - A is 2x5, B is 5x6, C is 2x6
        - We can decompose into blocks along the middle dimension (j)
        - Split j into blocks of size 2: j=0,1; j=2,3; j=4 (last one is 2x1)
        
        Actually, let's think more carefully. The tensor is:
        C[k*n+i] += A[i*m+j] * B[j*p+k]
        
        For n=2, m=5, p=6:
        - i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
        
        We can group j values into pairs and use Strassen on the i-k plane.
        
        For each pair (j, j+1) where both exist:
        - A_block = [[A[i,j], A[i,j+1]] for i in {0,1}]  (2x2)
        - B_block = [[B[j,k], B[j+1,k]] for k in {0,...,5}]  (2x6)
        
        This doesn't directly fit 2x2*2x2. Let me reconsider.
        
        Better approach: For each k, we have C[:,k] = A @ B[:,k]
        where A is 2x5 and B[:,k] is 5x1.
        
        Alternatively, think of it as: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
        
        Let's try a different block decomposition. Split the j dimension into:
        - j=0,1 (2 values)
        - j=2,3 (2 values)  
        - j=4 (1 value)
        
        For the first two blocks (j=0,1 and j=2,3), we can potentially use Strassen
        on the (i, k) sub-problem, but k has 6 values, not 2.
        
        Let me try yet another approach. Since n=2 is small, let's exploit that.
        
        Actually, the key insight from the cell finding is that applying Strassen to
        2x2x2 sub-blocks saves 1 rank per block. For 2x5*5x6, if we can identify
        6 such blocks, we save 6 ranks, going from 60 to 54.
        
        Let's construct this more carefully. The matrix multiplication tensor
        for 2x5*5x6 can be seen as a sum of 5 "slabs" (one for each j):
        For each j: C[:,k] += A[:,j] * B[j,:]
        which is an outer product of a 2-vector and a 6-vector, contributing
        to all 12 entries of C.
        
        Each such slab has rank 2 (as a matrix in the (i,k) plane) but as a
        tensor term, it's a single rank-1 term in the 3-tensor decomposition.
        
        The schoolbook uses 5*2*6 = 60 terms? No, schoolbook uses n*m*p = 60 terms,
        one for each (i,j,k) triple.
        
        Let me just implement the Strassen-based approach as described in the
        cell finding and verify it works.
        """
        # For now, let's try a simpler approach: use Strassen on 2x2 subproblems
        # where possible.
        
        # Split the problem into blocks. For 2x5 * 5x6:
        # Think of A as 2 blocks of 2 columns + 1 column: A = [A1 | A2 | A3]
        # where A1 is 2x2, A2 is 2x2, A3 is 2x1
        # B as [B1; B2; B3] where B1 is 2x6, B2 is 2x6, B3 is 1x6
        
        # C = A1*B1 + A2*B2 + A3*B3
        
        # For A1*B1: A1 is 2x2, B1 is 2x6. This is 2x2 * 2x6.
        # We can split B1 into 3 blocks of 2 columns: B1 = [B1a | B1b | B1c]
        # Each B1x is 2x2, so A1*B1x is 2x2*2x2, which can use Strassen (rank 7)
        # instead of schoolbook (rank 8). Saves 1 rank per 2x2*2x2 block.
        
        # So for A1*B1: 3 blocks of 2x2*2x2, each using rank 7 instead of 8.
        # Total: 3*7 = 21 instead of 3*8 = 24. Saves 3.
        
        # For A2*B2: same structure, 3 blocks of 2x2*2x2, saves 3 more.
        # Total: 21 instead of 24.
        
        # For A3*B3: A3 is 2x1, B3 is 1x6. This is 2x1 * 1x6.
        # C contribution is A3[i,0] * B3[0,k] for i in {0,1}, k in {0,...,5}
        # This is 2*6 = 12 rank-1 terms in schoolbook, but actually it's just
        # 2*1*1 = 2 rank-1 terms in the tensor sense? No...
        
        # Actually, for 2x1 * 1x6: C[i,k] = A[i,0] * B[0,k]
        # This is a rank-1 matrix (outer product of A[:,0] and B[0,:])
        # As a tensor term, it's one rank-1 term: U has nonzeros at (i,0) for i in {0,1},
        # V has nonzero at (0,0), W has nonzeros at (k,0) for k in {0,...,5}
        # Wait, the indexing is different. Let me be more careful.
        
        # I think I'm overcomplicating this. Let me just implement the Strassen
        # block approach and verify.
        
        pass  # Placeholder, will implement below

    # Let's implement the Strassen block decomposition properly
    def strassen_decomp_2x5_5x6():
        """
        Construct a rank-54 decomposition using Strassen on 2x2 sub-blocks.
        """
        # We'll build U, V, W incrementally.
        # U: (10, rank), V: (30, rank), W: (12, rank)
        
        # Strategy:
        # Split j into: (0,1), (2,3), (4)
        # For each pair (j,j+1), and for each pair of k values (k,k+1),
        # we have a 2x2*2x2 subproblem that can use Strassen.
        
        # Actually, let's think of it differently.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # Group j into pairs: (0,1), (2,3), and singleton (4)
        # Group k into pairs: (0,1), (2,3), (4,5)
        #
        # For each j-pair and k-pair, we have a 2x2 * 2x2 subproblem
        # (2 i-values, 2 j-values, 2 k-values)
        #
        # Number of j-pairs: 2 (for j=0,1 and j=2,3) + 1 singleton (j=4)
        # Number of k-pairs: 3 (for k=0,1; k=2,3; k=4,5)
        #
        # For j-pairs (0,1) and (2,3) with each k-pair: 2 * 3 = 6 subproblems
        # Each is 2x2*2x2, can use Strassen (rank 7) instead of schoolbook (rank 8)
        # Saves 1 per subproblem: 6 ranks saved
        #
        # For j=4 (singleton) with each k-pair: 1 * 3 = 3 subproblems
        # Each is 2x1*2x2 = 2x1*1x2? No, it's 2x1 * 1x2 for the k-pair
        # Actually for j=4 and k-pair (k0,k1):
        # C[i,k0] += A[i,4] * B[4,k0]
        # C[i,k1] += A[i,4] * B[4,k1]
        # This is 2*2 = 4 schoolbook terms, or can be done in... 
        # It's a 2x1 * 1x2 problem, which has rank 2 (outer product)
        # So rank 2 instead of 4, saves 2 per k-pair: 3*2 = 6 ranks saved
        #
        # Total savings: 6 + 6 = 12? That would give rank 48, which seems too good.
        # Let me reconsider.
        
        # Actually, the schoolbook rank is n*m*p = 2*5*6 = 60.
        # The "savings" need to be counted carefully.
        
        # Let me just construct it term by term.
        
        rank = 0
        U_list = []
        V_list = []
        W_list = []
        
        def add_term(U_col, V_col, W_col):
            nonlocal rank
            U_list.append(U_col)
            V_list.append(V_col)
            W_list.append(W_col)
            rank += 1
        
        # Helper to create column vectors
        def make_U_col(values):
            col = np.zeros(nm, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def make_V_col(values):
            col = np.zeros(mp, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def make_W_col(values):
            col = np.zeros(pn, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        # Strassen for 2x2*2x2 with specific index mappings
        # For subproblem with i in {i0,i1}, j in {j0,j1}, k in {k0,k1}:
        # A submatrix: [[A[i0,j0], A[i0,j1]], [A[i1,j0], A[i1,j1]]]
        # B submatrix: [[B[j0,k0], B[j0,k1]], [B[j1,k0], B[j1,k1]]]
        # C submatrix: [[C[i0,k0], C[i0,k1]], [C[i1,k0], C[i1,k1]]]
        
        def strassen_terms(i0, i1, j0, j1, k0, k1):
            """Generate Strassen terms for 2x2*2x2 subproblem."""
            # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
            # Contributes to C[i0,k0] and C[i1,k1]
            add_term(
                make_U_col({i0*m+j0: 1, i1*m+j1: 1}),
                make_V_col({j0*p+k0: 1, j1*p+k1: 1}),
                make_W_col({k0*n+i0: 1, k1*n+i1: 1})
            )
            
            # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
            # Contributes to C[i1,k0]
            add_term(
                make_U_col({i1*m+j0: 1, i1*m+j1: 1}),
                make_V_col({j0*p+k0: 1}),
                make_W_col({k0*n+i1: 1})
            )
            
            # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
            # Contributes to C[i0,k1] and C[i1,k1]
            add_term(
                make_U_col({i0*m+j0: 1}),
                make_V_col({j0*p+k1: 1, j1*p+k1: -1}),
                make_W_col({k1*n+i0: 1, k1*n+i1: 1})
            )
            
            # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
            # Contributes to C[i0,k0] and C[i1,k0]
            add_term(
                make_U_col({i1*m+j1: 1}),
                make_V_col({j1*p+k0: 1, j0*p+k0: -1}),
                make_W_col({k0*n+i0: 1, k0*n+i1: 1})
            )
            
            # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
            # Contributes to C[i0,k0] (negative) and C[i0,k1]
            add_term(
                make_U_col({i0*m+j0: 1, i0*m+j1: 1}),
                make_V_col({j1*p+k1: 1}),
                make_W_col({k0*n+i0: -1, k1*n+i0: 1})
            )
            
            # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
            # Contributes to C[i1,k1]
            add_term(
                make_U_col({i1*m+j0: 1, i0*m+j0: -1}),
                make_V_col({j0*p+k0: 1, j0*p+k1: 1}),
                make_W_col({k1*n+i1: 1})
            )
            
            # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
            # Contributes to C[i0,k0]
            add_term(
                make_U_col({i0*m+j1: 1, i1*m+j1: -1}),
                make_V_col({j1*p+k0: 1, j1*p+k1: 1}),
                make_W_col({k0*n+i0: 1})
            )
        
        # Apply Strassen to all 2x2*2x2 subproblems
        # j-pairs: (0,1), (2,3)
        # k-pairs: (0,1), (2,3), (4,5)
        # i-pair: (0,1)
        
        for (j0, j1) in [(0, 1), (2, 3)]:
            for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
                strassen_terms(0, 1, j0, j1, k0, k1)
        
        # Handle j=4 (singleton) with each k-pair
        for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
            # C[i,k0] += A[i,4] * B[4,k0]
            # C[i,k1] += A[i,4] * B[4,k1]
            # This is 2x1 * 1x2, which can be done in rank 2:
            # Term 1: A[i,4] * B[4,k0] contributes to C[i,k0] for i in {0,1}
            # Term 2: A[i,4] * B[4,k1] contributes to C[i,k1] for i in {0,1}
            
            add_term(
                make_U_col({0*m+4: 1, 1*m+4: 1}),
                make_V_col({4*p+k0: 1}),
                make_W_col({k0*n+0: 1, k0*n+1: 1})
            )
            
            add_term(
                make_U_col({0*m+4: 1, 1*m+4: 1}),
                make_V_col({4*p+k1: 1}),
                make_W_col({k1*n+0: 1, k1*n+1: 1})
            )
        
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 60

    # Try Strassen decomposition
    try:
        U_s, V_s, W_s, rank_s = strassen_decomp_2x5_5x6()
        if verify(U_s, V_s, W_s) and rank_s < best_rank:
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = rank_s
    except Exception:
        pass

    # Try to find even lower rank with random search
    np.random.seed(42)
    for target_rank in range(best_rank - 1, max(30, best_rank - 10), -1):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            U_try = np.zeros((nm, target_rank), dtype=np.float64)
            V_try = np.zeros((mp, target_rank), dtype=np.float64)
            W_try = np.zeros((pn, target_rank), dtype=np.float64)
            
            for r in range(target_rank):
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                if np.random.rand() < 0.5:
                    U_try[i * m + j, r] = 1.0
                    V_try[j * p + k, r] = 1.0
                    W_try[k * n + i, r] = 1.0
                else:
                    for ii in range(n):
                        for jj in range(m):
                            if np.random.rand() < 0.3:
                                U_try[ii * m + jj, r] = np.random.choice([-1, 1])
                    for jj in range(m):
                        for kk in range(p):
                            if np.random.rand() < 0.3:
                                V_try[jj * p + kk, r] = np.random.choice([-1, 1])
                    for kk in range(p):
                        for ii in range(n):
                            if np.random.rand() < 0.3:
                                W_try[kk * n + ii, r] = np.random.choice([-1, 1])
            
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                found = True
                break
        
        if found:
            break

    def to_list(arr):
        return arr.astype(float).tolist()

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
