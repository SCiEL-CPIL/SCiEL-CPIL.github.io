import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Recursive Strassen for 4x4: split into 2x2 blocks of 2x2 submatrices
    # Each 2x2 block multiplication uses Strassen (rank 7)
    # Total: 7 products of 2x2 matrices, each with rank 7 -> 49 scalar products
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # Strassen 2x2 scheme:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For 4x4: A is split into 2x2 blocks where each block is 2x2
    # A[2i+a][2j+b] for block (i,j) element (a,b)
    # B[2k+c][2l+d] for block (k,l) element (c,d)
    # C[2p+e][2q+f] for block (p,q) element (e,f)
    
    # Strassen coefficients for 2x2:
    # X matrices (for A side), Y matrices (for B side)
    # Each is 2x2x2x2: X[a][b][I][J] means contribution to element (a,b) from block (I,J)
    
    strassen_products = [
        # (X_coeffs, Y_coeffs, C_weights)
        # X[a][b][I][J], Y[c][d][K][L], C_weights[(P,Q)]
        #
        # M0: (A00+A11)(B00+B11)
        (
            # X: A00 + A11
            [[[[1,0],[0,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[0,0]]]],
            # Y: B00 + B11
            [[[[1,0],[0,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[0,0]]]],
            # C: M0 contributes to C00 and C11
            {(0,0): 1, (1,1): 1}
        ),
        # M1: (A10+A11)B00
        (
            # X: A10 + A11
            [[[[0,0],[0,0]],[[0,0],[0,0]]],
             [[[1,0],[0,0]],[[0,0],[0,0]]]],
            # Y: B00
            [[[[1,0],[0,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[0,0]]]],
            # C: M1 contributes to C10 (+1) and C11 (-1)
            {(1,0): 1, (1,1): -1}
        ),
        # M2: A00(B01-B11)
        (
            # X: A00
            [[[[1,0],[0,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[0,0]]]],
            # Y: B01 - B11
            [[[[0,0],[1,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[-1,0]]]],
            # C: M2 contributes to C01 (+1) and C11 (+1)
            {(0,1): 1, (1,1): 1}
        ),
        # M3: A11(B10-B00)
        (
            # X: A11
            [[[[0,0],[0,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[1,0],[0,0]]]],
            # Y: B10 - B00
            [[[[0,0],[0,0]],[[0,0],[0,0]]],
             [[[1,0],[0,0]],[[0,0],[0,0]]]],
            # C: M3 contributes to C00 (+1) and C10 (+1)
            {(0,0): 1, (1,0): 1}
        ),
        # M4: (A00+A01)B11
        (
            # X: A00 + A01
            [[[[1,0],[1,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[0,0]]]],
            # Y: B11
            [[[[0,0],[0,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[1,0],[0,0]]]],
            # C: M4 contributes to C00 (-1) and C01 (+1)
            {(0,0): -1, (0,1): 1}
        ),
        # M5: (A10-A00)(B00+B01)
        (
            # X: A10 - A00
            [[[[0,0],[0,0]],[[0,0],[0,0]]],
             [[[1,0],[0,0]],[[0,0],[0,0]]]],
            # Y: B00 + B01
            [[[[1,0],[1,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[0,0]]]],
            # C: M5 contributes to C11 (+1)
            {(1,1): 1}
        ),
        # M6: (A01-A11)(B10+B11)
        (
            # X: A01 - A11
            [[[[0,0],[1,0]],[[0,0],[0,0]]],
             [[[0,0],[0,0]],[[0,0],[-1,0]]]],
            # Y: B10 + B11
            [[[[0,0],[0,0]],[[0,0],[0,0]]],
             [[[1,0],[1,0]],[[0,0],[0,0]]]],
            # C: M6 contributes to C00 (-1)
            {(0,0): -1}
        ),
    ]
    
    # For each of the 7 Strassen products, we have a 2x2 matrix multiplication
    # which itself decomposes into 7 scalar products (inner Strassen)
    # So total rank = 7 * 7 = 49
    
    # The 2x2 matrix product Z = X*Y where X, Y are 2x2:
    # Z[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
    # Z[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
    # Z[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
    # Z[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
    
    # Apply inner Strassen to each 2x2 product:
    # Inner Strassen for 2x2:
    # m0 = (X00+X11)(Y00+Y11)
    # m1 = (X10+X11)Y00
    # m2 = X00(Y01-Y11)
    # m3 = X11(Y10-Y00)
    # m4 = (X00+X01)Y11
    # m5 = (X10-X00)(Y00+Y01)
    # m6 = (X01-X11)(Y10+Y11)
    # Z00 = m0 + m3 - m4 - m6
    # Z01 = m2 + m4
    # Z10 = m1 + m3
    # Z11 = m0 - m1 + m2 + m5
    
    # Inner Strassen coefficients:
    # For X side: inner_X[t][a][b] where t is product index, (a,b) is element
    # For Y side: inner_Y[t][c][d]
    
    inner_strassen = [
        # (X_coeffs[2][2], Y_coeffs[2][2], Z_weights[(e,f)])
        # m0: (X00+X11)(Y00+Y11)
        (
            [[1,0],[1,0]],  # X00+X11: element (0,0) gets 1, (1,1) gets 1... wait
            # Actually: X00+X11 means the matrix has X[0][0] in position (0,0) and X[1][1] in position (1,1)
            # So coeff matrix: [[1,0],[0,1]]
            [[1,0],[0,1]],
            [[1,0],[0,1]],
            {(0,0): 1, (1,1): 1}
        ),
        # m1: (X10+X11)Y00
        (
            [[0,0],[1,1]],
            [[1,0],[0,0]],
            {(1,0): 1, (1,1): -1}
        ),
        # m2: X00(Y01-Y11)
        (
            [[1,0],[0,0]],
            [[0,1],[0,-1]],
            {(0,1): 1, (1,1): 1}
        ),
        # m3: X11(Y10-Y00)
        (
            [[0,0],[0,1]],
            [[0,-1],[1,0]],
            {(0,0): 1, (1,0): 1}
        ),
        # m4: (X00+X01)Y11
        (
            [[1,1],[0,0]],
            [[0,0],[0,1]],
            {(0,0): -1, (0,1): 1}
        ),
        # m5: (X10-X00)(Y00+Y01)
        (
            [[-1,0],[1,0]],
            [[1,1],[0,0]],
            {(1,1): 1}
        ),
        # m6: (X01-X11)(Y10+Y11)
        (
            [[0,1],[0,-1]],
            [[0,0],[1,1]],
            {(0,0): -1}
        ),
    ]
    
    # Now build the full decomposition:
    # Outer product s (0-6), inner product t (0-6) -> rank index r = s*7 + t
    
    for s, (X_outer, Y_outer, C_outer) in enumerate(strassen_products):
        for t, (X_inner, Y_inner, Z_inner) in enumerate(inner_strassen):
            r = s * 7 + t
            
            # Build U contributions:
            # The outer X gives a 2x2 matrix X_out[a][b] for a,b in {0,1}
            # The inner X gives coefficients X_in[a][b] 
            # Combined X coefficient for element (a,b) of the 2x2 block:
            # For block (I,J) of the 4x4 matrix, element (a,b):
            # coefficient = X_outer[a][b][I][J] * X_inner[a][b]
            # Wait, that's not right. Let me think more carefully.
            
            # The outer Strassen computes: for each element (a,b) of the 2x2 result,
            # X_out[a][b] = sum_{I,J} X_outer[a][b][I][J] * A_block[I][J][a][b]
            # where A_block[I][J][a][b] = A[2I+a][2J+b]
            
            # Then the inner Strassen multiplies this 2x2 matrix X_out by Y_out.
            # The inner product t computes a scalar:
            # scalar_t = sum_{a,b,c,d} X_inner[a][b] * Y_inner[c][d] * (delta_{b,c}) * X_out[a][b] * Y_out[c][d]
            # Wait no. The inner Strassen product t is:
            # scalar_t = (sum_{a,b} X_inner[a][b] * X_out[a][b]) * (sum_{c,d} Y_inner[c][d] * Y_out[c][d])
            
            # So the total scalar for rank r is:
            # sum_{I,J,a,b} X_outer[a][b][I][J] * X_inner[a][b] * A[2I+a][2J+b]
            # times
            # sum_{K,L,c,d} Y_outer[c][d][K][L] * Y_inner[c][d] * B[2K+c][2L+d]
            
            # U contribution: for A[2I+a][2J+b], coefficient = X_outer[a][b][I][J] * X_inner[a][b]
            for I in range(2):
                for J in range(2):
                    for a in range(2):
                        for b in range(2):
                            coeff = X_outer[a][b][I][J] * X_inner[a][b]
                            if coeff != 0:
                                u_idx = (2*I + a) * m + (2*J + b)
                                U[u_idx][r] += coeff
            
            # V contribution: for B[2K+c][2L+d], coefficient = Y_outer[c][d][K][L] * Y_inner[c][d]
            for K in range(2):
                for L in range(2):
                    for c in range(2):
                        for d in range(2):
                            coeff = Y_outer[c][d][K][L] * Y_inner[c][d]
                            if coeff != 0:
                                v_idx = (2*K + c) * p + (2*L + d)
                                V[v_idx][r] += coeff
            
            # W contribution: the inner product t contributes to Z elements (e,f) with weight Z_inner[(e,f)]
            # The outer product s contributes to C blocks (P,Q) with weight C_outer[(P,Q)]
            # C[2P+e][2Q+f] gets contribution: C_outer[(P,Q)] * Z_inner[(e,f)]
            # W index: k*n + i where k=col=2Q+f, i=row=2P+e
            for (P, Q), w_outer in C_outer.items():
                for (e, f), w_inner in Z_inner.items():
                    weight = w_outer * w_inner
                    if weight != 0:
                        row = 2*P + e
                        col = 2*Q + f
                        w_idx = col * n + row
                        W[w_idx][r] += weight
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        # C[i,k] = sum_r U[i*m+k, r] * V[j*p+l, r] * W[k*n+i, r] ... 
        # Actually: C = sum_r (U[:,r] outer V[:,r]) reshaped and weighted by W
        # More precisely: C[i][k] = sum_{j} A[i][j] * B[j][k]
        # In tensor form: C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
        # Where T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_val = 0.0
                    for r in range(rank):
                        t_val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    C_computed[i][k] += t_val * A[i][j] * B[j][k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            return _construct_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
