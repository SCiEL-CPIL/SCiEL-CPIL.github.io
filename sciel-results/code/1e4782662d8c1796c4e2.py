"""Rewritten program: recursive block-splitting with Strassen for 2x2x2 sub-blocks.
Produces a rank-71 exact tensor decomposition for (3,5,5) matrix multiplication."""

def construct():
    """Return {'U','V','W'} with shapes (15,rank),(25,rank),(15,rank) for (3,5,5)."""
    # We will build the tensor decomposition recursively.
    # Each "problem" is described by (n, m, p) and a mapping of (i,j,k) to global indices.
    # We accumulate rank-1 terms (triples of vectors).
    
    # Instead of building sparse vectors, we'll build the full U, V, W matrices.
    # Total rank will be determined by the recursion.
    
    # Let's use a cleaner approach: generate a list of (u_row, v_row, w_row, coeff) tuples
    # where each represents a rank-1 term with coefficient 1 (we'll handle signs).
    # Actually, for tensor decomposition, each rank-1 term contributes U[row,c]*V[row,c]*W[row,c]
    # to T[i,j,k]. The coefficient is absorbed into the entries.
    
    # Let me think differently. I'll build U, V, W as lists of lists, growing columns as needed.
    
    # For a sub-problem with local dimensions (n, m, p), the local tensor indices are:
    # i in [0,n), j in [0,m), k in [0,p)
    # The global mapping is provided by offsets.
    
    # I'll use a recursive function that, given local (n,m,p) and offset functions,
    # adds rank-1 terms to global U, V, W.
    
    # Let me define the problem as: we need to compute C = A * B where A is n x m, B is m x p.
    # The tensor T[i,j,k] = 1 for all i,j,k means: C[i,k] = sum_j A[i,j] * B[j,k].
    
    # For a sub-block A_sub (n1 x m1) and B_sub (m1 x p1), we compute C_sub (n1 x p1).
    # The local tensor has indices i in [0,n1), j in [0,m1), k in [0,p1).
    
    # I'll build this with explicit index tracking.
    
    # Start with empty factor matrices.
    U = []  # will be list of lists, shape (15, rank)
    V = []  # shape (25, rank)
    W = []  # shape (15, rank)
    
    # We need to grow these. Let's pre-allocate with 0s and track rank.
    # But we don't know rank ahead. Let's use a list of columns.
    
    # Better: accumulate terms as (i_offset, j_offset, k_offset, u_val, v_val, w_val) 
    # where u_val, v_val, w_val are the coefficients for that rank-1 term.
    # Actually, for each rank-1 term c, we need:
    #   U[i*m+j, c] = u_{i,j}
    #   V[j*p+k, c] = v_{j,k}
    #   W[k*n+i, c] = w_{k,i}
    # such that for each (i,j,k): sum_c u_{i,j} * v_{j,k} * w_{k,i} = 1
    
    # For schoolbook on (n,m,p): one term per (i,j,k) with u=1, v=1, w=1.
    # For Strassen on (2,2,2): 7 terms with specific u,v,w patterns.
    
    # Let me define a function that takes local (n,m,p) and returns a list of rank-1 terms.
    # Each term is a tuple (U_dict, V_dict, W_dict) where:
    #   U_dict maps (i,j) -> value
    #   V_dict maps (j,k) -> value
    #   W_dict maps (k,i) -> value
    # For a valid decomposition, for each (i,j,k): sum over terms of U_dict[(i,j)]*V_dict[(j,k)]*W_dict[(k,i)] = 1
    
    # Schoolbook terms for (n,m,p):
    def schoolbook_terms(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_dict = {(i,j): 1}
                    V_dict = {(j,k): 1}
                    W_dict = {(k,i): 1}
                    terms.append((U_dict, V_dict, W_dict))
        return terms
    
    # Strassen terms for (2,2,2):
    def strassen_terms():
        # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
        # C = [[c00,c01],[c10,c11]]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Each M_t corresponds to a rank-1 term. For M_t = A_lin * B_lin,
        # where A_lin is a linear combination of A entries and B_lin is a linear combo of B entries.
        # The contribution to C[i,k] is coeff * M_t.
        # So for term t, U_dict[(i,j)] = coeff_A[i,j] (the coefficient of A[i,j] in A_lin),
        # V_dict[(j,k)] = coeff_B[j,k] (the coefficient of B[j,k] in B_lin),
        # W_dict[(k,i)] = coeff_C[k,i] (the coefficient in C[i,k] = sum coeff_C[k,i] * M_t)
        # Wait, I need to be more careful.
        #
        # The tensor decomposition requires: for each (i,j,k), sum_c U[(i,j),c] V[(j,k),c] W[(k,i),c] = 1
        # This means the "1" in T[i,j,k] is decomposed.
        #
        # For Strassen, we have: C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_{i',j'} alpha_t[i',j'] A[i',j']) (sum_{j',k'} beta_t[j',k'] B[j',k']) * gamma_t[i,k]
        # where gamma_t[i,k] is the coefficient of M_t in C[i,k].
        #
        # So U[(i,j), t] = alpha_t[i,j], V[(j,k), t] = beta_t[j,k], W[(k,i), t] = gamma_t[i,k]
        # (note: W index is (k,i) per the convention W[k*n+i, c])
        
        terms = []
        
        # M1 = (a00+a11)(b00+b11), contributes to:
        # C00: +1, C11: +1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        
        # M2 = (a10+a11)b00, contributes to:
        # C10: +1, C11: -1
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
        
        # M3 = a00(b01-b11), contributes to:
        # C01: +1, C11: +1
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
        
        # M4 = a11(b10-b00), contributes to:
        # C00: +1, C10: +1
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
        
        # M5 = (a00+a01)b11, contributes to:
        # C00: -1, C01: +1
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
        
        # M6 = (a10-a00)(b00+b01), contributes to:
        # C11: +1
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        
        # M7 = (a01-a11)(b10+b11), contributes to:
        # C00: +1
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
        
        return terms
    
    # Now, recursive decomposition.
    # decompose(n, m, p) returns a list of terms for the (n,m,p) problem.
    # Each term is (U_dict, V_dict, W_dict) with local indices.
    
    def decompose(n, m, p):
        # Base cases
        if n == 2 and m == 2 and p == 2:
            return strassen_terms()
        if n == 1:
            return schoolbook_terms(n, m, p)
        if m == 1:
            return schoolbook_terms(n, m, p)
        if p == 1:
            return schoolbook_terms(n, m, p)
        
        # Try splitting m into m1 + m2
        # C = A[:, :m1] @ B[:m1, :] + A[:, m1:] @ B[m1:, :]
        # The two sub-problems are (n, m1, p) and (n, m2, p)
        # They share the same i and k indices, different j ranges.
        
        # Choose split point. Let's try m1 = m // 2, m2 = m - m1
        m1 = m // 2
        m2 = m - m1
        
        # But we need to be careful: the j indices in the two sub-problems are offset.
        # Sub-problem 1: j in [0, m1), uses A[i, j] and B[j, k]
        # Sub-problem 2: j in [m1, m), uses A[i, j] and B[j, k]
        # In local terms for sub-problem 2, j_local = j_global - m1
        
        terms = []
        
        # Sub-problem 1: (n, m1, p)
        for U_d, V_d, W_d in decompose(n, m1, p):
            terms.append((U_d, V_d, W_d))
        
        # Sub-problem 2: (n, m2, p) with j offset by m1
        for U_d, V_d, W_d in decompose(n, m2, p):
            # Shift j indices in U and V
            new_U_d = {}
            for (i, j), val in U_d.items():
                new_U_d[(i, j + m1)] = val
            new_V_d = {}
            for (j, k), val in V_d.items():
                new_V_d[(j + m1, k)] = val
            terms.append((new_U_d, new_V_d, W_d))
        
        return terms
    
    # But wait, this splitting strategy might not always be optimal. Let me also try splitting n or p.
    # For (3,5,5), splitting m=5 into 2+3 gives (3,2,5) + (3,3,5).
    # For (3,2,5), we can split p=5 into 2+3: (3,2,2) + (3,2,3).
    # For (3,2,2), split n=3 into 2+1: (2,2,2) + (1,2,2).
    # etc.
    
    # Let me implement a more general recursive strategy that tries all possible splits.
    
    def decompose_general(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return strassen_terms()
        if n <= 1 or m <= 1 or p <= 1:
            return schoolbook_terms(n, m, p)
        
        # Try all possible splits
        best = None
        best_len = float('inf')
        
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            terms1 = decompose_general(n, m1, p)
            terms2 = decompose_general(n, m2, p)
            # Shift terms2
            shifted_terms2 = []
            for U_d, V_d, W_d in terms2:
                new_U_d = {(i, j + m1): val for (i, j), val in U_d.items()}
                new_V_d = {(j + m1, k): val for (j, k), val in V_d.items()}
                shifted_terms2.append((new_U_d, new_V_d, W_d))
            total = terms1 + shifted_terms2
            if len(total) < best_len:
                best_len = len(total)
                best = total
        
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            terms1 = decompose_general(n, m, p1)
            terms2 = decompose_general(n, m, p2)
            shifted_terms2 = []
            for U_d, V_d, W_d in terms2:
                new_V_d = {(j, k + p1): val for (j, k), val in V_d.items()}
                new_W_d = {(k + p1, i): val for (k, i), val in W_d.items()}
                shifted_terms2.append((U_d, new_V_d, new_W_d))
            total = terms1 + shifted_terms2
            if len(total) < best_len:
                best_len = len(total)
                best = total
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            terms1 = decompose_general(n1, m, p)
            terms2 = decompose_general(n2, m, p)
            shifted_terms2 = []
            for U_d, V_d, W_d in terms2:
                new_U_d = {(i + n1, j): val for (i, j), val in U_d.items()}
                new_W_d = {(k, i + n1): val for (k, i), val in W_d.items()}
                shifted_terms2.append((new_U_d, V_d, new_W_d))
            total = terms1 + shifted_terms2
            if len(total) < best_len:
                best_len = len(total)
                best = total
        
        if best is None:
            return schoolbook_terms(n, m, p)
        return best
    
    # Get the decomposition for (3,5,5)
    all_terms = decompose_general(3, 5, 5)
    rank = len(all_terms)
    
    # Build U, V, W
    U = [[0] * rank for _ in range(15)]  # n*m = 3*5 = 15
    V = [[0] * rank for _ in range(25)]  # m*p = 5*5 = 25
    W = [[0] * rank for _ in range(15)]  # p*n = 5*3 = 15
    
    n, m, p = 3, 5, 5
    for c, (U_d, V_d, W_d) in enumerate(all_terms):
        for (i, j), val in U_d.items():
            U[i * m + j][c] = val
        for (j, k), val in V_d.items():
            V[j * p + k][c] = val
        for (k, i), val in W_d.items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
