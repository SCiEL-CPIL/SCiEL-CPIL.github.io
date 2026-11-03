"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Block decomposition 5 = 2 + 3.
    - Use Strassen (rank 7) for the 2x2x2 diagonal block.
    - Use schoolbook for all other sub-block multiplications.
    Total rank: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    """
    import time
    import numpy as np
    
    start_time = time.time()
    DEADLINE = 135
    
    n, m, p = DIMENSIONS  # all 5
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() - start_time > DEADLINE - 10:
        return schoolbook()
    
    # Block sizes
    a1, a2 = 2, 3  # rows of A split
    b1, b2 = 2, 3  # cols of A / rows of B split
    c1, c2 = 2, 3  # cols of B / cols of C split
    
    # Build full 5x5x5 decomposition via blocks
    # A is 5x5: rows 0-1 are block 1, rows 2-4 are block 2
    # B is 5x5: cols 0-1 are block 1, cols 2-4 are block 2
    # C is 5x5: rows 0-1 are block 1, cols 0-1 are block 1
    
    # We'll accumulate U, V, W columns
    U_cols = []  # list of column vectors (length 25)
    V_cols = []  # list of column vectors (length 25)
    W_cols = []  # list of column vectors (length 25)
    
    def add_term(u_vec, v_vec, w_vec):
        """Add one rank-1 term. u_vec, v_vec, w_vec are length-25 lists."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Helper: create zero vector of length 25
    def zero_vec():
        return [0] * 25
    
    # Strassen 2x2x2 (rank 7) for C11 = A11 * B11
    # A11 is rows 0-1, cols 0-1 of A -> indices i*5+j for i in [0,1], j in [0,1]
    # B11 is rows 0-1, cols 0-1 of B -> indices j*5+k for j in [0,1], k in [0,1]
    # C11 is rows 0-1, cols 0-1 of C -> W index k*5+i for k in [0,1], i in [0,1]
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs) where each is dict mapping (local_i,local_j) etc to value
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B01-B10)
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B10)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        # M7 = (A01-A11)(B01+B11)
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1}),
    ]
    
    for u_coeff, v_coeff, w_coeff in strassen_terms:
        u_vec = zero_vec()
        v_vec = zero_vec()
        w_vec = zero_vec()
        for (li, lj), val in u_coeff.items():
            u_vec[li * 5 + lj] = val  # global row = li (0 or 1), col = lj
        for (lj, lk), val in v_coeff.items():
            v_vec[lj * 5 + lk] = val  # global row = lj, col = lk
        for (lk, li), val in w_coeff.items():
            w_vec[lk * 5 + li] = val  # W[k*n+i] = W[lk*5+li]
        add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C11 cross term: A12(2x3) * B21(3x2)
    # A12: rows 0-1, cols 2-4 of A
    # B21: rows 2-4, cols 0-1 of B
    # C11: rows 0-1, cols 0-1 of C
    # C11[i,k] += sum_{j=2}^{4} A[i,j] * B[j,k]
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C12 = A11(2x2) * B12(2x3)
    # A11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-4
    # C12: rows 0-1, cols 2-4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C12 cross term: A12(2x3) * B22(3x3)
    # A12: rows 0-1, cols 2-4
    # B22: rows 2-4, cols 2-4
    # C12: rows 0-1, cols 2-4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C21 = A21(3x2) * B11(2x2)
    # A21: rows 2-4, cols 0-1
    # B11: rows 0-1, cols 0-1
    # C21: rows 2-4, cols 0-1
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C21 cross term: A22(3x3) * B21(3x2)
    # A22: rows 2-4, cols 2-4
    # B21: rows 2-4, cols 0-1
    # C21: rows 2-4, cols 0-1
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C22 = A21(3x2) * B12(2x3)
    # A21: rows 2-4, cols 0-1
    # B12: rows 0-1, cols 2-4
    # C22: rows 2-4, cols 2-4
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C22 cross term: A22(3x3) * B22(3x3)
    # A22: rows 2-4, cols 2-4
    # B22: rows 2-4, cols 2-4
    # C22: rows 2-4, cols 2-4
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Assemble final matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    for r in range(rank):
        for idx in range(25):
            U[idx][r] = U_cols[r][idx]
            V[idx][r] = V_cols[r][idx]
            W[idx][r] = W_cols[r][idx]
    
    # Verify: check that the decomposition is correct
    # C[i,k] = sum_r U[i*5+j_r] * V[j_r*5+k_r] * W[k_r*5+i_r] ... actually verify via tensor
    # Let's do a quick numerical check with random matrices
    try:
        rng = np.random.RandomState(42)
        A_test = rng.rand(5, 5)
        B_test = rng.rand(5, 5)
        C_expected = A_test @ B_test
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # C[i,k] = sum_{r} sum_{j} U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r]
        # Actually the tensor decomposition means:
        # T[(i,j),(j',k),(k',i')] = sum_r U[i*5+j, r] * V[j'*5+k, r] * W[k'*5+i', r]
        # And T should equal delta(j,j') * delta(k,k') * delta(i,i')
        
        # Simpler check: compute C from the decomposition
        C_computed = np.zeros((5, 5))
        for r in range(rank):
            # U[:,r] gives coefficients for A, V[:,r] for B, W[:,r] for C
            # The contribution to C[i,k] from term r is:
            # sum_j U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r]
            # But this isn't quite right. Let me think again.
            #
            # In the tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
            # where T[(i,j),(j',k),(k',i')] = sum_r u_r[i*5+j] * v_r[j'*5+k] * w_r[k'*5+i']
            # For matrix multiplication, T[(i,j),(j',k),(k',i')] = 1 iff j=j', k=k', i=i'
            #
            # So C[i,k] = A @ B means C[i,k] = sum_j A[i,j] B[j,k]
            # The decomposition says: sum_j A[i,j] B[j,k] = sum_r (sum_j u_r[i*5+j] A[i,j]) * (sum_k' v_r[j'*5+k'] B[j',k']) * ... 
            # No, that's not right either.
            #
            # Actually: C[i,k] = sum_r w_r[k*5+i] * (sum_j u_r[i*5+j] * A[i,j]) * ... 
            # Hmm, let me just verify by checking the tensor identity directly.
            pass
        
        # Direct verification: for random A, B, check A@B matches
        # C[i,k] should equal sum_r W[k*5+i, r] * (U[:,r]^T * A_flat) * (V[:,r]^T * B_flat)
        # No! That's not how tensor decomposition works for matrix multiplication.
        #
        # The correct interpretation: The multiplication tensor M has M[i,j,k] = 1 (it's the identity in the middle).
        # M = sum_r u_r ⊗ v_r ⊗ w_r where:
        # u_r is in R^{25} (A space), v_r in R^{25} (B space), w_r in R^{25} (C space)
        # M[i,j,k] = sum_r u_r[i*5+j] * v_r[j*5+k] * w_r[k*5+i]
        # And M[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        #
        # Wait no. The multiplication tensor is: T[(i,j),(j',k),(k',i')] = delta(j,j')*delta(k,k')*delta(i,i')
        # And T = sum_r u_r ⊗ v_r ⊗ w_r
        # So: delta(j,j')*delta(k,k')*delta(i,i') = sum_r u_r[i*5+j] * v_r[j'*5+k] * w_r[k'*5+i']
        #
        # For the schoolbook, each term has u_r[i*5+j]=1, v_r[j*5+k]=1, w_r[k*5+i]=1 for specific (i,j,k).
        #
        # To verify: pick random A, B, compute C = A@B, and check that
        # C[i,k] = sum_j A[i,j]*B[j,k] matches what the decomposition gives.
        #
        # From the tensor identity: sum_j A[i,j]*B[j,k] = sum_r (sum_j u_r[i*5+j]*A[i,j]) * (sum_{j',k} v_r[j'*5+k]*B[j',k]) * ... 
        # No, this is getting confusing. Let me just verify numerically by expanding.
        
        # The decomposition guarantees: for any A, B:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_r w_r[k*5+i] * (sum_j u_r[i*5+j] A[i,j]) * ... 
        # Actually NO. The tensor decomposition means:
        # The bilinear map (A,B) -> C satisfies C[i,k] = sum_j A[i,j] B[j,k]
        # And the decomposition gives: C[i,k] = sum_r (linear form on A)_r * (linear form on B)_r * w_r[k*5+i]
        # where (linear form on A)_r = sum_{i',j'} u_r[i'*5+j'] A[i',j']
        # and (linear form on B)_r = sum_{j',k'} v_r[j'*5+k'] B[j',k']
        #
        # So: C[i,k] = sum_r w_r[k*5+i] * (sum_{i',j'} u_r[i'*5+j'] A[i',j']) * (sum_{j',k'} v_r[j'*5+k'] B[j',k'])
        #
        # Let's verify this numerically:
        C_from_decomp = np.zeros((5, 5))
        for r in range(rank):
            la = np.sum(U_np[:, r] * A_test.flatten())
            lb = np.sum(V_np[:, r] * B_test.flatten())
            for i in range(5):
                for k in range(5):
                    C_from_decomp[i, k] += W_np[k * 5 + i, r] * la * lb
        
        if not np.allclose(C_from_decomp, C_expected, atol=1e-9):
            # Decomposition is wrong, fall back to schoolbook
            return schoolbook()
    except Exception:
        return schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
