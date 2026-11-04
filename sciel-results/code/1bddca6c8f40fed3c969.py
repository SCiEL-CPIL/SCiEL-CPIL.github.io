"""Rewritten program for matrix multiplication tensor decomposition 3x5x5."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    total_rank = 49 + 25  # 74
    
    # U: (n*m, rank) = (15, 74)
    # V: (m*p, rank) = (25, 74)
    # W: (p*n, rank) = (15, 74)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_schoolbook(offset, i_base, j_base, k_base, ni, nj, np_):
        """Add schoolbook decomposition for M_{ni, nj, np_} with index offsets."""
        local_rank = ni * nj * np_
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(np_):
                    col = offset + (ii * nj + jj) * np_ + kk
                    # U index: (i_base + ii) * m + (j_base + jj)
                    u_idx = (i_base + ii) * m + (j_base + jj)
                    # V index: (j_base + jj) * p + (k_base + kk)
                    v_idx = (j_base + jj) * p + (k_base + kk)
                    # W index: (k_base + kk) * n + (i_base + ii)
                    w_idx = (k_base + kk) * n + (i_base + ii)
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
        return offset + local_rank
    
    def add_strassen(offset, i_base, j_base, k_base):
        """Add Strassen decomposition for M_{2,2,2} with index offsets.
        
        Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
        Global: i = i_base + i', j = j_base + j', k = k_base + k'
        
        A[i',j'] * B[j',k'] = C[i',k']
        
        Strassen products:
        m1 = (a+d)(e+h)    -> a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
                              e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-f)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        
        C[0,0] = m1 + m4 - m5 + m7
        C[0,1] = m3 + m5
        C[1,0] = m2 + m4
        C[1,1] = m1 - m2 + m3 + m6
        """
        # 7 rank-1 terms
        # Each term c has: U[i', j', c], V[j', k', c], W[k', i', c]
        # For Strassen, each product m_r = (linear combo of A) * (linear combo of B)
        # This gives a rank-1 tensor where U encodes the A-coefficients,
        # V encodes the B-coefficients, and W encodes the C-coefficients.
        
        # Product m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # U: A[0,0]=1, A[1,1]=1 => U[0*m+0][c]=1, U[1*m+1][c]=1
        # V: B[0,0]=1, B[1,1]=1 => V[0*p+0][c]=1, V[1*p+1][c]=1
        # W: contributes to C[0,0]+1, C[1,1]+1
        c = offset
        U[(i_base + 0) * m + (j_base + 0)][c] = 1
        U[(i_base + 1) * m + (j_base + 1)][c] = 1
        V[(j_base + 0) * p + (k_base + 0)][c] = 1
        V[(j_base + 1) * p + (k_base + 1)][c] = 1
        W[(k_base + 0) * n + (i_base + 0)][c] = 1  # C[0,0]
        W[(k_base + 1) * n + (i_base + 1)][c] = 1  # C[1,1]
        
        # Product m2 = (A[1,0] + A[1,1]) * B[0,0]
        c = offset + 1
        U[(i_base + 1) * m + (j_base + 0)][c] = 1
        U[(i_base + 1) * m + (j_base + 1)][c] = 1
        V[(j_base + 0) * p + (k_base + 0)][c] = 1
        W[(k_base + 0) * n + (i_base + 1)][c] = 1  # C[1,0]
        
        # Product m3 = A[0,0] * (B[0,1] - B[1,1])
        c = offset + 2
        U[(i_base + 0) * m + (j_base + 0)][c] = 1
        V[(j_base + 0) * p + (k_base + 1)][c] = 1
        V[(j_base + 1) * p + (k_base + 1)][c] = -1
        W[(k_base + 1) * n + (i_base + 0)][c] = 1  # C[0,1]
        
        # Product m4 = A[1,1] * (B[1,0] - B[0,0])
        c = offset + 3
        U[(i_base + 1) * m + (j_base + 1)][c] = 1
        V[(j_base + 1) * p + (k_base + 0)][c] = 1
        V[(j_base + 0) * p + (k_base + 0)][c] = -1
        W[(k_base + 0) * n + (i_base + 1)][c] = 1  # C[1,0]
        
        # Product m5 = (A[0,0] + A[0,1]) * B[1,1]
        c = offset + 4
        U[(i_base + 0) * m + (j_base + 0)][c] = 1
        U[(i_base + 0) * m + (j_base + 1)][c] = 1
        V[(j_base + 1) * p + (k_base + 1)][c] = 1
        W[(k_base + 0) * n + (i_base + 0)][c] = -1  # C[0,0]
        W[(k_base + 1) * n + (i_base + 0)][c] = 1   # C[0,1]
        
        # Product m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        c = offset + 5
        U[(i_base + 1) * m + (j_base + 0)][c] = 1
        U[(i_base + 0) * m + (j_base + 0)][c] = -1
        V[(j_base + 0) * p + (k_base + 0)][c] = 1
        V[(j_base + 0) * p + (k_base + 1)][c] = 1
        W[(k_base + 1) * n + (i_base + 1)][c] = 1  # C[1,1]
        
        # Product m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        c = offset + 6
        U[(i_base + 0) * m + (j_base + 1)][c] = 1
        U[(i_base + 1) * m + (j_base + 1)][c] = -1
        V[(j_base + 1) * p + (k_base + 0)][c] = 1
        V[(j_base + 1) * p + (k_base + 1)][c] = 1
        W[(k_base + 0) * n + (i_base + 0)][c] = 1  # C[0,0]
        
        return offset + 7
    
    # Partition:
    # i in {0,1,2} -> split into i in {0,1} and i in {2}
    # j in {0,1,2,3,4} -> split into j in {0,1,2} and j in {3,4}
    # k in {0,1,2,3,4} -> split into k in {0,1,2} and k in {3,4}
    
    # Sub-problem 1: i in {0,1}, j in {0,1,2}, k in {0,1,2} -> schoolbook M_{2,3,3} = 18
    offset = add_schoolbook(0, 0, 0, 0, 2, 3, 3)
    
    # Sub-problem 2: i in {0,1}, j in {3,4}, k in {3,4} -> Strassen M_{2,2,2} = 7
    offset = add_strassen(offset, 0, 3, 3)
    
    # Sub-problem 3: i in {0,1}, j in {0,1,2}, k in {3,4} -> schoolbook M_{2,3,2} = 12
    offset = add_schoolbook(offset, 0, 0, 3, 2, 3, 2)
    
    # Sub-problem 4: i in {0,1}, j in {3,4}, k in {0,1,2} -> schoolbook M_{2,2,3} = 12
    offset = add_schoolbook(offset, 0, 3, 0, 2, 2, 3)
    
    # Sub-problem 5: i in {2}, j in {0,1,2,3,4}, k in {0,1,2,3,4} -> schoolbook M_{1,5,5} = 25
    offset = add_schoolbook(offset, 2, 0, 0, 1, 5, 5)
    
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Should not happen, but guard just in case
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
