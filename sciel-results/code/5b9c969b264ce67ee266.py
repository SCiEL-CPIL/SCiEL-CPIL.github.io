<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Uses structured Strassen-like ansatz for rank 16.
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit

    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True

    # Generate schoolbook as baseline (rank 18)
    U_sb = np.zeros((6, 18), dtype=int)
    V_sb = np.zeros((9, 18), dtype=int)
    W_sb = np.zeros((6, 18), dtype=int)
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = 18

    # Structured rank-16 ansatz based on Strassen-like splitting
    # For 2x3 * 3x3, we can use the following construction:
    # Split the 3x3 B matrix into two overlapping 2x3 submatrices
    # and use bilinear combinations to reduce products.
    
    # The key idea: use 16 terms with the following structure
    # Terms 0-7: handle the first two columns of A with all rows of B
    # Terms 8-15: handle the third column of A with corrections
    
    # Let's try a specific structured pattern:
    # U rows: [A00, A01, A02, A10, A11, A12]
    # V rows: [B00, B01, B02, B10, B11, B12, B20, B21, B22]
    # W rows: [C00, C01, C02, C10, C11, C12]
    
    # Try this rank-16 construction with integer coefficients:
    U_16 = np.array([
        # Row 0: A[0,0]
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: A[0,1]
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Row 2: A[0,2]
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # Row 3: A[1,0]
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: A[1,1]
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Row 5: A[1,2]
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    V_16 = np.array([
        # Row 0: B[0,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: B[0,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 2: B[0,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 3: B[1,0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: B[1,1]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 5: B[1,2]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 6: B[2,0]
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 7: B[2,1]
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 8: B[2,2]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    W_16 = np.array([
        # Row 0: C[0,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: C[0,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 2: C[0,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 3: C[1,0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: C[1,1]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 5: C[1,2]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    # Verify the structured ansatz
    if verify_decomp(U_16, V_16, W_16):
        best_rank = 16
        best_decomp = (U_16, V_16, W_16)
    else:
        # If structured ansatz doesn't work, try random search for rank 16
        import random
        random.seed(42)
        
        def try_random_rank(r, max_iters=3000, coeff_range=(-2, 2)):
            """Try random integer coefficient search for given rank."""
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                U = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
                V = np.random.randint(coeff_range[0], coeff_range[1]+1, (9, r))
                W = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
                if verify_decomp(U, V, W):
                    return (U, V, W)
            return None
        
        result = try_random_rank(16, max_iters=2000)
        if result is not None:
            best_rank = 16
            best_decomp = result
        elif time.time() < deadline - 10:
            result = try_random_rank(17, max_iters=2000)
            if result is not None:
                best_rank = 17
                best_decomp = result

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Uses structured Strassen-like ansatz with parameterized integer coefficients.
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit

    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True

    # Generate schoolbook as baseline (rank 18)
    U_sb = np.zeros((6, 18), dtype=int)
    V_sb = np.zeros((9, 18), dtype=int)
    W_sb = np.zeros((6, 18), dtype=int)
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = 18

    # Structured algebraic ansatz: parameterize rank-18 schoolbook with
    # small integer coefficients to exploit Strassen-like symmetries.
    # 
    # Key insight: For 2x3 * 3x3, we can split A into two 1x3 row blocks
    # and B into overlapping column blocks, using bilinear combinations.
    #
    # Ansatz: Use 16 terms where each term c has:
    #   U[:, c] = linear combo of A rows with coeffs in {-1, 0, 1}
    #   V[:, c] = linear combo of B rows with coeffs in {-1, 0, 1}
    #   W[:, c] = linear combo of C columns with coeffs in {-1, 0, 1}
    #
    # The tensor condition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    #
    # We search over a structured family of 16-term decompositions.

    import itertools
    import random
    random.seed(42)

    # Structured search: try specific algebraic patterns
    # Pattern 1: Split into 2x2 Strassen-like blocks with correction terms
    
    # For rank 16, we need 16 independent bilinear forms.
    # Try a systematic search over small coefficient sets.
    
    def try_structured_rank(r, max_patterns=50000):
        """
        Search for rank-r decomposition using structured patterns.
        Each term uses coefficients from {-1, 0, 1} for U, V, W.
        """
        # Precompute all possible single-term contributions
        # A term (u, v, w) contributes to tensor entry (i,j,k) by:
        #   u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # We'll use a smarter approach: build terms that cover specific (i,j,k) triples
        # while canceling unwanted contributions.
        
        # Start with a greedy construction
        U = np.zeros((n*m, r), dtype=int)
        V = np.zeros((m*p, r), dtype=int)
        W = np.zeros((p*n, r), dtype=int)
        
        # For each term, try to assign coefficients that satisfy the tensor equation
        # This is a constraint satisfaction problem.
        
        # Alternative: use known algebraic identities
        # For 2x3 * 3x3, consider the following structured decomposition:
        
        # Let's try a specific family parameterized by a few integers
        # U[i,j,c] = a_c * delta(i, i_c) + b_c * delta(i, i_c') 
        # V[j,k,c] = c_c * delta(j, j_c) + d_c * delta(j, j_c')
        # W[k,i,c] = e_c * delta(k, k_c) + f_c * delta(k, k_c')
        
        # This is getting complex. Let's try a direct but efficient search.
        
        # Use numpy for fast verification
        def fast_verify(U, V, W):
            """Fast verification using numpy broadcasting."""
            # U: (6, r), V: (9, r), W: (6, r)
            # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # This is a 3D tensor contraction.
            # Reshape: U as (n, m, r), V as (m, p, r), W as (p, n, r)
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            # Compute tensor T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
            # Use einsum: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # This requires careful indexing.
            # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
            
            # Let's compute it manually for small dimensions
            T = np.zeros((n, m, p), dtype=int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
                        T[i,j,k] = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                        # Wait, that's not right. It should be element-wise product then sum.
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        T[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            return np.all(T == 1)
        
        # Search over structured patterns
        # For rank r, we have r columns in U, V, W.
        # Each column is a vector of length n*m, m*p, p*n respectively.
        # We restrict entries to {-1, 0, 1}.
        
        # For r=16, this is a large search space. Use a heuristic approach.
        
        # Heuristic: build terms one at a time, ensuring each new term
        # helps satisfy at least one unsatisfied (i,j,k) equation.
        
        # Actually, let's try a different approach: use the fact that
        # the matrix multiplication tensor has a specific structure.
        
        # For 2x3 * 3x3, the tensor T[i,j,k] = 1 for all i,j,k.
        # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1.
        
        # Try a rank-16 construction based on the following idea:
        # Split the 3 columns of B into groups and use Strassen-like products.
        
        # Let's try a specific known construction:
        # Terms 0-3: handle (i=0, j=0, k=0..2) and (i=1, j=0, k=0..2)
        # Terms 4-7: handle (i=0, j=1, k=0..2) and (i=1, j=1, k=0..2)
        # Terms 8-11: handle (i=0, j=2, k=0..2) and (i=1, j=2, k=0..2)
        # Terms 12-15: correction terms to cancel unwanted contributions
        
        # This is getting too complex for a quick search. Let's try a 
        # more systematic but bounded search.
        
        # For small r, we can try random search with structured initialization.
        
        # Initialize with schoolbook pattern and perturb
        if r >= 18:
            # Just return schoolbook
            U_r = np.zeros((n*m, r), dtype=int)
            V_r = np.zeros((m*p, r), dtype=int)
            W_r = np.zeros((p*n, r), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U_r[i*m+j, col] = 1
                            V_r[j*p+k, col] = 1
                            W_r[k*n+i, col] = 1
                            col += 1
            return (U_r, V_r, W_r)
        
        # For r < 18, we need a smarter construction
        # Try random search with small coefficient range
        for attempt in range(max_patterns):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small coefficients
            U_rand = np.random.randint(-1, 2, (n*m, r))
            V_rand = np.random.randint(-1, 2, (m*p, r))
            W_rand = np.random.randint(-1, 2, (p*n, r))
            
            if fast_verify(U_rand, V_rand, W_rand):
                return (U_rand, V_rand, W_rand)
        
        return None

    # Try rank 16 with structured search
    if time.time() < deadline - 20:
        result_16 = try_structured_rank(16, max_patterns=20000)
        if result_16 is not None:
            best_rank = 16
            best_decomp = result_16
    elif time.time() < deadline - 10:
        # Try rank 17
        result_17 = try_structured_rank(17, max_patterns=20000)
        if result_17 is not None:
            best_rank = 17
            best_decomp = result_17

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
