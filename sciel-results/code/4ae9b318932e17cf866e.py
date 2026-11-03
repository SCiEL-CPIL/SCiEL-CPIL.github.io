import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _elapsed(start):
    return time.time() - start

def _time_left(start, deadline=145.0):
    return deadline - _elapsed(start)

def _verify(U, V, W, n=3, m=3, p=8, seed=42):
    """Verify if U, V, W decompose the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with multiple seeds for robustness
    for test_seed in [seed, seed + 1000, seed + 2000]:
        np.random.seed(test_seed)
        A = np.random.randint(-10, 10, (n, m)).astype(float)
        B = np.random.randint(-10, 10, (m, p)).astype(float)
        C = A @ B
        
        C_decomp = np.zeros((n, p))
        for c in range(U.shape[1]):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            XY = (u_c * A) @ (v_c * B)
            C_decomp += XY * w_c.T
        
        if not np.allclose(C, C_decomp, atol=1e-6):
            return False
    
    return True

def _schoolbook(n=3, m=3, p=8):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _strassen_decomposition(n=3, m=3, p=8):
    """
    Construct a rank-52 decomposition using lacing.
    Split p=8 into four blocks of 2 columns.
    For each block of 2 columns:
      - Use Strassen 2x2x2 for rows 0,1. (7 terms)
      - Use schoolbook for row 2. (6 terms)
    Total: 4 * (7 + 6) = 52 terms.
    """
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    def strassen_2x2x2(i0, j0, k0):
        """
        Adds 7 Strassen terms for a 2x2x2 block starting at (i0, j0, k0).
        i0: start row index in A
        j0: start col index in A (and row in B)
        k0: start col index in B
        """
        # M1 = (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
        u1 = np.zeros(n*m); u1[i0*m+j0] = 1; u1[(i0+1)*m+(j0+1)] = 1
        v1 = np.zeros(m*p); v1[j0*p+k0] = 1; v1[(j0+1)*p+(k0+1)] = 1
        w1 = np.zeros(p*n); w1[k0*n+i0] = 1; w1[(k0+1)*n+i0] = -1; w1[k0*n+(i0+1)] = -1; w1[(k0+1)*n+(i0+1)] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1])B[j0,k0]
        u2 = np.zeros(n*m); u2[(i0+1)*m+j0] = 1; u2[(i0+1)*m+(j0+1)] = 1
        v2 = np.zeros(m*p); v2[j0*p+k0] = 1
        w2 = np.zeros(p*n); w2[k0*n+(i0+1)] = 1
        add_term(u2, v2, w2)
        
        # M3 = A[i0,j0](B[j0,k0+1]-B[j0+1,k0+1])
        u3 = np.zeros(n*m); u3[i0*m+j0] = 1
        v3 = np.zeros(m*p); v3[j0*p+(k0+1)] = 1; v3[(j0+1)*p+(k0+1)] = -1
        w3 = np.zeros(p*n); w3[(k0+1)*n+i0] = 1
        add_term(u3, v3, w3)
        
        # M4 = A[i0+1,j0+1](B[j0+1,k0]-B[j0,k0])
        u4 = np.zeros(n*m); u4[(i0+1)*m+(j0+1)] = 1
        v4 = np.zeros(m*p); v4[(j0+1)*p+k0] = 1; v4[j0*p+k0] = -1
        w4 = np.zeros(p*n); w4[k0*n+(i0+1)] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A[i0,j0]+A[i0,j0+1])B[j0+1,k0+1]
        u5 = np.zeros(n*m); u5[i0*m+j0] = 1; u5[i0*m+(j0+1)] = 1
        v5 = np.zeros(m*p); v5[(j0+1)*p+(k0+1)] = 1
        w5 = np.zeros(p*n); w5[(k0+1)*n+i0] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A[i0+1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k0+1])
        u6 = np.zeros(n*m); u6[(i0+1)*m+j0] = 1; u6[i0*m+j0] = -1
        v6 = np.zeros(m*p); v6[j0*p+k0] = 1; v6[j0*p+(k0+1)] = 1
        w6 = np.zeros(p*n); w6[k0*n+i0] = -1; w6[k0*n+(i0+1)] = 1
        add_term(u6, v6, w6)
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1])(B[j0+1,k0]+B[j0+1,k0+1])
        u7 = np.zeros(n*m); u7[i0*m+(j0+1)] = 1; u7[(i0+1)*m+(j0+1)] = -1
        v7 = np.zeros(m*p); v7[(j0+1)*p+k0] = 1; v7[(j0+1)*p+(k0+1)] = 1
        w7 = np.zeros(p*n); w7[(k0+1)*n+i0] = -1; w7[(k0+1)*n+(i0+1)] = 1
        add_term(u7, v7, w7)

    def schoolbook_row(i, k_start, k_end):
        """
        Adds schoolbook terms for row i of A and columns k_start to k_end-1 of B.
        Number of terms: (k_end - k_start) * m
        """
        for k in range(k_start, k_end):
            for j in range(m):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                add_term(u, v, w)

    # Split columns into 4 blocks of 2
    # Block 0: cols 0,1
    # Block 1: cols 2,3
    # Block 2: cols 4,5
    # Block 3: cols 6,7
    
    for block in range(4):
        k0 = block * 2
        # Strassen for rows 0,1; cols k0, k0+1
        strassen_2x2x2(0, 0, k0)
        
        # Schoolbook for row 2; cols k0, k0+1
        schoolbook_row(2, k0, k0+2)
        
    # Convert lists to arrays
    U_arr = np.column_stack(U)
    V_arr = np.column_stack(V)
    W_arr = np.column_stack(W)
    
    return U_arr, V_arr, W_arr

def _strassen_3x3x3_decomposition(n=3, m=3, p=8):
    """
    Try to use a 3x3x3 decomposition for the first 3 columns,
    then handle the remaining 5 columns with schoolbook.
    This is exploratory - we need a known 3x3x3 decomposition with rank < 27.
    """
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # For now, use schoolbook for all 8 columns as baseline
    # This will be replaced if we find a better 3x3x3 decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                add_term(u, v, w)
    
    U_arr = np.column_stack(U)
    V_arr = np.column_stack(V)
    W_arr = np.column_stack(W)
    
    return U_arr, V_arr, W_arr

def _mixed_block_decomposition(n=3, m=3, p=8):
    """
    Try different block sizes: 3+3+2 or 4+4.
    For 3-column blocks, use 2x2 Strassen + schoolbook for 3rd row.
    """
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    def strassen_2x2x2(i0, j0, k0):
        u1 = np.zeros(n*m); u1[i0*m+j0] = 1; u1[(i0+1)*m+(j0+1)] = 1
        v1 = np.zeros(m*p); v1[j0*p+k0] = 1; v1[(j0+1)*p+(k0+1)] = 1
        w1 = np.zeros(p*n); w1[k0*n+i0] = 1; w1[(k0+1)*n+i0] = -1; w1[k0*n+(i0+1)] = -1; w1[(k0+1)*n+(i0+1)] = 1
        add_term(u1, v1, w1)
        
        u2 = np.zeros(n*m); u2[(i0+1)*m+j0] = 1; u2[(i0+1)*m+(j0+1)] = 1
        v2 = np.zeros(m*p); v2[j0*p+k0] = 1
        w2 = np.zeros(p*n); w2[k0*n+(i0+1)] = 1
        add_term(u2, v2, w2)
        
        u3 = np.zeros(n*m); u3[i0*m+j0] = 1
        v3 = np.zeros(m*p); v3[j0*p+(k0+1)] = 1; v3[(j0+1)*p+(k0+1)] = -1
        w3 = np.zeros(p*n); w3[(k0+1)*n+i0] = 1
        add_term(u3, v3, w3)
        
        u4 = np.zeros(n*m); u4[(i0+1)*m+(j0+1)] = 1
        v4 = np.zeros(m*p); v4[(j0+1)*p+k0] = 1; v4[j0*p+k0] = -1
        w4 = np.zeros(p*n); w4[k0*n+(i0+1)] = 1
        add_term(u4, v4, w4)
        
        u5 = np.zeros(n*m); u5[i0*m+j0] = 1; u5[i0*m+(j0+1)] = 1
        v5 = np.zeros(m*p); v5[(j0+1)*p+(k0+1)] = 1
        w5 = np.zeros(p*n); w5[(k0+1)*n+i0] = 1
        add_term(u5, v5, w5)
        
        u6 = np.zeros(n*m); u6[(i0+1)*m+j0] = 1; u6[i0*m+j0] = -1
        v6 = np.zeros(m*p); v6[j0*p+k0] = 1; v6[j0*p+(k0+1)] = 1
        w6 = np.zeros(p*n); w6[k0*n+i0] = -1; w6[k0*n+(i0+1)] = 1
        add_term(u6, v6, w6)
        
        u7 = np.zeros(n*m); u7[i0*m+(j0+1)] = 1; u7[(i0+1)*m+(j0+1)] = -1
        v7 = np.zeros(m*p); v7[(j0+1)*p+k0] = 1; v7[(j0+1)*p+(k0+1)] = 1
        w7 = np.zeros(p*n); w7[(k0+1)*n+i0] = -1; w7[(k0+1)*n+(i0+1)] = 1
        add_term(u7, v7, w7)

    def schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = np.zeros(n*m); u[i*m+j] = 1
                    v = np.zeros(m*p); v[j*p+k] = 1
                    w = np.zeros(p*n); w[k*n+i] = 1
                    add_term(u, v, w)
    
    # Strategy: 4 blocks of 2 columns each (same as current)
    # But let's also try: first 3 columns with 2x2 Strassen + schoolbook row 2,
    # then next 3 columns same, then last 2 columns with 2x2 Strassen + schoolbook row 2
    
    # Block 0: cols 0,1,2
    strassen_2x2x2(0, 0, 0)
    schoolbook_block(2, 3, 0, 3, 0, 3)  # Row 2, all 3 cols
    
    # Block 1: cols 3,4,5
    strassen_2x2x2(0, 0, 3)
    schoolbook_block(2, 3, 0, 3, 3, 6)  # Row 2, cols 3-5
    
    # Block 2: cols 6,7
    strassen_2x2x2(0, 0, 6)
    schoolbook_block(2, 3, 0, 3, 6, 8)  # Row 2, cols 6-7
    
    U_arr = np.column_stack(U)
    V_arr = np.column_stack(V)
    W_arr = np.column_stack(W)
    
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try the Strassen-laced decomposition (4 blocks of 2)
    if _time_left(start, deadline) > 10:
        U_laced, V_laced, W_laced = _strassen_decomposition(n, m, p)
        if U_laced.shape[1] < best_rank and _verify(U_laced, V_laced, W_laced, n, m, p, seed=42) and _verify(U_laced, V_laced, W_laced, n, m, p, seed=123):
            best_U, best_V, best_W = U_laced, V_laced, W_laced
            best_rank = U_laced.shape[1]
    
    # Try mixed block decomposition (3+3+2)
    if _time_left(start, deadline) > 10:
        U_mixed, V_mixed, W_mixed = _mixed_block_decomposition(n, m, p)
        if U_mixed.shape[1] < best_rank and _verify(U_mixed, V_mixed, W_mixed, n, m, p, seed=42) and _verify(U_mixed, V_mixed, W_mixed, n, m, p, seed=123):
            best_U, best_V, best_W = U_mixed, V_mixed, W_mixed
            best_rank = U_mixed.shape[1]
    
    # Convert to lists for the return interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
