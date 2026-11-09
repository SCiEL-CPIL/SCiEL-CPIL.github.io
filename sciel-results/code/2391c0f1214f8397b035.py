import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (20, rank)
    
    # We will build the decomposition term by term.
    # Each term is a rank-1 tensor: u * v * w
    # We store them in lists and then convert to matrices.
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_term(u, v, w):
        terms_U.append(u)
        terms_V.append(v)
        terms_W.append(w)

    # Helper to get index in U (A matrix)
    # A is n x m. Index i*m + j
    def idx_U(i, j):
        return i * m + j

    # Helper to get index in V (B matrix)
    # B is m x p. Index j*p + k
    def idx_V(j, k):
        return j * p + k

    # Helper to get index in W (C matrix)
    # C is n x p. Trace convention W[k*n + i, r]
    # So index is k*n + i
    def idx_W(k, i):
        return k * n + i

    # 1. Decompose the core 4x4x4 part (j in 0..3, k in 0..3)
    # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # This is handled by 8 Strassen decompositions.
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # Each coeff is a dict mapping (i,j) to value for U, (j,k) for V, (k,i) for W
        # But it's easier to just add individual rank-1 terms for each M.
        # We will generate the 7 terms for a specific 2x2 block.
    ]
    
    def add_strassen_block(i0, j0, k0):
        # i0, j0, k0 are the top-left indices of the 2x2 blocks in A, B, C
        # A block: rows i0, i0+1; cols j0, j0+1
        # B block: rows j0, j0+1; cols k0, k0+1
        # C block: rows i0, i0+1; cols k0, k0+1
        
        # M1
        u1 = { (i0, j0): 1, (i0+1, j0+1): 1 }
        v1 = { (j0, k0): 1, (j0+1, k0+1): 1 }
        w1 = { (k0, i0): 1, (k0+1, i0+1): 1 } # C00, C11 contributions
        # Wait, W is sparse. For a rank-1 term, W is a vector.
        # The term is sum u_i v_j w_k.
        # For M1, it contributes to C00 and C11.
        # C00 index: k0, i0. C11 index: k0+1, i0+1.
        # So w1 has non-zero at idx_W(k0, i0) and idx_W(k0+1, i0+1).
        
        # We need to add 7 terms.
        # Term 1: M1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i0, j0)] = 1; u[idx_U(i0+1, j0+1)] = 1
        v[idx_V(j0, k0)] = 1; v[idx_V(j0+1, k0+1)] = 1
        w[idx_W(k0, i0)] = 1; w[idx_W(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        # Term 2: M2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i0+1, j0)] = 1; u[idx_U(i0+1, j0+1)] = 1
        v[idx_V(j0, k0)] = 1
        w[idx_W(k0, i0+1)] = 1; w[idx_W(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        # Term 3: M3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i0, j0)] = 1
        v[idx_V(j0, k0+1)] = 1; v[idx_V(j0+1, k0+1)] = -1
        w[idx_W(k0+1, i0)] = 1
        add_term(u, v, w)
        
        # Term 4: M4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i0+1, j0+1)] = 1
        v[idx_V(j0+1, k0)] = 1; v[idx_V(j0, k0)] = -1
        w[idx_W(k0, i0+1)] = 1
        add_term(u, v, w)
        
        # Term 5: M5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i0, j0)] = 1; u[idx_U(i0, j0+1)] = 1
        v[idx_V(j0+1, k0+1)] = 1
        w[idx_W(k0+1, i0)] = -1 # C01 gets -M5? No.
        # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # M5 contributes to C01 (pos) and C11 (neg)?
        # Formula: C01 = M3 + M5. C11 = ... - M5?
        # Let's re-verify Strassen.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M5 is +1 in C01, -1 in C00? No, C00 has -M5.
        # So w for M5: idx_W(k0+1, i0) is C01?
        # C indices: C[i, k].
        # C01 is i=0, k=1. Relative to block i0, k0: i=i0, k=k0+1.
        # C00 is i=0, k=0. Relative: i=i0, k=k0.
        # So w[idx_W(k0+1, i0)] = 1 (for C01)
        # w[idx_W(k0, i0)] = -1 (for C00)
        w[idx_W(k0+1, i0)] = 1
        w[idx_W(k0, i0)] = -1
        add_term(u, v, w)
        
        # Term 6: M6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i0+1, j0)] = 1; u[idx_U(i0, j0)] = -1
        v[idx_V(j0, k0)] = 1; v[idx_V(j0, k0+1)] = 1
        w[idx_W(k0+1, i0+1)] = 1 # C11 gets +M6
        add_term(u, v, w)
        
        # Term 7: M7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i0, j0+1)] = 1; u[idx_U(i0+1, j0+1)] = -1
        v[idx_V(j0+1, k0)] = 1; v[idx_V(j0+1, k0+1)] = 1
        w[idx_W(k0, i0)] = 1 # C00 gets +M7
        add_term(u, v, w)

    # Add the 8 Strassen blocks for the 4x4x4 core
    # Block (0,0) -> i0=0, j0=0, k0=0
    add_strassen_block(0, 0, 0)
    # Block (0,1) -> i0=0, j0=0, k0=2 (C top-right)
    # Wait, C block (0,1) is i=0,1 and k=2,3.
    # A block is i=0,1 and j=0,1? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For C[0:2, 2:4] (top-right 2x2 of C):
    # k is 2,3.
    # j sums over 0..3.
    # This involves A[0:2, 0:4] and B[0:4, 2:4].
    # This is not a simple 2x2x2 block.
    
    # My previous decomposition logic was:
    # C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2:4]B[2:4, 0:2]
    # So we need Strassen for (i0=0, j0=0, k0=0) and (i0=0, j0=2, k0=0).
    
    add_strassen_block(0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    
    add_strassen_block(0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    add_strassen_block(2, 0, 0) # A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    add_strassen_block(2, 2, 0) # A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    
    add_strassen_block(2, 0, 2) # A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    add_strassen_block(2, 2, 2) # A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    
    # 2. Decompose the remaining parts (j=4 or k=4) using Schoolbook
    # Part A: j=4, k=0..3
    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx_U(i, 4)] = 1
            v[idx_V(4, k)] = 1
            w[idx_W(k, i)] = 1
            add_term(u, v, w)
            
    # Part B: j=0..3, k=4
    # A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for j in range(4):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx_U(i, j)] = 1
            v[idx_V(j, 4)] = 1
            w[idx_W(4, i)] = 1
            add_term(u, v, w)

    # Part C: j=4, k=4
    # A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx_U(i, 4)] = 1
        v[idx_V(4, 4)] = 1
        w[idx_W(4, i)] = 1
        add_term(u, v, w)

    rank = len(terms_U)
    U = [term for term in terms_U]
    V = [term for term in terms_V]
    W = [term for term in terms_W]
    
    return {"U": U, "V": V, "W": W}
