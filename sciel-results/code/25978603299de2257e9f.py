<<<<<<< SEARCH
DIMENSIONS = [3, 3, 4]

def _make_3x3x3_rank26():
    """Construct rank-26 decomposition for 3x3x3 matrix multiplication.
    
    Returns U (9x26), V (9x26), W (9x26) for the 3x3x3 tensor.
    Indexing: U[i*3+j], V[j*3+k], W[k*3+i] for i,j,k in {0,1,2}.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen for 2x2x2: 7 terms
    # A11=A[0:2,0:2], A12=A[0:2,2], A21=A[2,0:2], A22=A[2,2]
    # B11=B[0:2,0:2], B12=B[0:2,2], B21=B[2,0:2], B22=B[2,2]
    # Strassen 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen M1: (A11+A22)*(B11+B22)
    # A11 = [[A[0,0],A[0,1]],[A[1,0],A[1,1]]], A22 = [[A[2,2]]]
    # So (A11+A22) has entries: A[0,0],A[0,1];A[1,0],A[1,1];A[2,2] (at position (2,2))
    # Actually in our block: A is 3x3, A11 is 2x2 top-left, A22 is 1x1 bottom-right.
    # (A11+A22) means we add A[2,2] to the (1,1) position of A11? No.
    # Strassen: M1 = (A11+A22)*(B11+B22) where A11,A22 are 2x2 blocks.
    # But here A22 is 1x1, not 2x2. So this doesn't directly apply.
    
    # Let me redo: use the full schoolbook for the 2x2*2x2 part = rank 4,
    # plus the cross terms.
    
    # Actually, let me just use schoolbook for everything to get rank 36,
    # then try to find a better decomposition.
    
    # For now, let me implement the rank-36 schoolbook and return it.
    # This is the baseline.
    
    # Reconstruct schoolbook:
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    return U, V, W, col
=======
DIMENSIONS = [3, 3, 4]

def _add_term(U, V, W, col, u_entry, v_entry, w_entry, u_val, v_val, w_val):
    """Add a single rank-1 term to the decomposition."""
    U[u_entry][col] += u_val
    V[v_entry][col] += v_val
    W[w_entry][col] += w_val
>>>>>>> REPLACE

