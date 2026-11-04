import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    start_time = time.time()
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if i corresponds to (row_a, col_a) 
    # and k corresponds to (row_a, col_b) with j being (col_a, col_b)
    # U index: (a, c) where a in [0,2), c in [0,4) -> index a*4+c
    # V index: (c, d) where c in [0,4), d in [0,4) -> index c*4+d  
    # W index: (a, d) where a in [0,2), d in [0,4) -> index a*4+d
    
    # The tensor T[a*4+c, c*4+d, a*4+d] = 1 for all valid a,c,d
    # Otherwise 0
    
    # For rank decomposition, we need U[i,r]*V[j,r]*W[k,r] summed over r = T[i,j,k]
    
    # Let's try a rank-14 construction using block Strassen
    # Split c into c1 in [0,2) and c2 in [2,4), similarly d into d1 in [0,2), d2 in [2,4)
    
    # For each a (0 or 1):
    #   A_a = [A_a1, A_a2] (1x2 x 2x2 blocks, but actually A is 2x4 so each row is 1x4)
    #   Wait, A is 2x4, so row a has entries A[a, c] for c in 0..3
    #   B is 4x4, so B[c, d] for c,d in 0..3
    #   Result C[a, d] = sum_c A[a,c] * B[c,d]
    
    # Split c into {0,1} and {2,3}, d into {0,1} and {2,3}
    # For row a:
    #   C[a, 0:2] = A[a,0:2] @ B[0:2, 0:2] + A[a,2:4] @ B[2:4, 0:2]
    #   C[a, 2:4] = A[a,0:2] @ B[0:2, 2:4] + A[a,2:4] @ B[2:4, 2:4]
    
    # Using Strassen on the 2x2 blocks for each row:
    # Let A1 = A[a,0:2] (1x2), A2 = A[a,2:4] (1x2)
    # Let B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    
    # Strassen for 1x2 * 2x2 -> 1x2:
    # This is like multiplying a 1x2 by 2x2, which can be done with... 
    # Actually 1x2 * 2x2 = 1x2, needs 4 scalar mults schoolbook, or can we do better?
    # For 1xn * nxn, the rank is at least n (lower bound from slice rank or something)
    # For n=2, rank is 4? No, let's think...
    # (x1, x2) * [[b11,b12],[b21,b22]] = (x1*b11+x2*b21, x1*b12+x2*b22)
    # This needs 4 mults: x1*b11, x2*b21, x1*b12, x2*b22. Can we do it in 3?
    # The tensor is in R^{2} x R^{4} x R^{2}, entries T[i,j,k] where i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # T[0,0,0]=1, T[0,1,0]=0, T[0,2,0]=1, T[0,3,0]=0, T[1,0,1]=0, T[1,1,1]=1, T[1,2,1]=0, T[1,3,1]=1
    # This is two independent rank-1 terms: (e1, e0+e2, e1) and (e2, e1+e3, e2)? No...
    # Term 1: i=0, j=0, k=0 and i=0, j=2, k=0 -> not rank 1
    # Actually the tensor factors as: for output 0: sum of x1*b11 + x2*b21, for output 1: x1*b12 + x2*b22
    # These share x1 and x2 but have different B entries.
    # Rank is 4 (schoolbook) and I believe it's minimal for 1x2*2x2.
    
    # So for each row a, we need rank 4 for left half + rank 4 for right half = rank 8 per row
    # Total rank 16 for both rows.
    
    # But wait, can we share between rows? The B matrix is shared.
    # Row 0 uses: A00*B00 + A02*B20, A01*B01 + A03*B21 (using 0-indexed c)
    # Row 1 uses: A10*B00 + A12*B20, A11*B01 + A13*B21
    # The B factors are the same! So we can combine:
    # (A00+A10)*B00, (A02+A12)*B20, etc. But that changes the A factors.
    # Actually no - we need separate outputs for each row.
    
    # Let me just build rank-16 and see if we can reduce.
    
    rank = 16
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    # For each row a (0,1), for each half of c (0:2, 2:4), for each half of d (0:2, 2:4):
    # We have a 1x2 * 2x2 multiplication which needs 4 rank-1 terms
    
    # Let's index: for row a, c-half ch (0 or 1), d-half dh (0 or 1):
    # The 4 scalar multiplications are:
    #   A[a, ch*2+0] * B[ch*2+0, dh*2+0] -> contributes to C[a, dh*2+0]
    #   A[a, ch*2+0] * B[ch*2+0, dh*2+1] -> contributes to C[a, dh*2+1]
    #   A[a, ch*2+1] * B[ch*2+1, dh*2+0] -> contributes to C[a, dh*2+0]
    #   A[a, ch*2+1] * B[ch*2+1, dh*2+1] -> contributes to C[a, dh*2+1]
    
    # This is just schoolbook with rank = 2*2*2*2 = 16
    
    col = 0
    for a in range(2):
        for ch in range(2):
            for dh in range(2):
                for cc in range(2):
                    for dd in range(2):
                        # U index: a*4 + ch*2 + cc
                        # V index: (ch*2+cc)*4 + dh*2+dd
                        # W index: a*4 + dh*2+dd
                        U[a*4 + ch*2 + cc][col] = 1.0
                        V[(ch*2+cc)*4 + dh*2+dd][col] = 1.0
                        W[a*4 + dh*2+dd][col] = 1.0
                        col += 1
    
    # Verify the decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check: sum_r U[i,r]*V[j,r]*W[k,r] should equal T[i,j,k]
    # T[i,j,k] = 1 if i = a*4+c, j = c*4+d, k = a*4+d for some a,c,d
    
    # Build expected tensor
    T_expected = np.zeros((8, 16, 8))
    for a in range(2):
        for c in range(4):
            for d in range(4):
                T_expected[a*4+c, c*4+d, a*4+d] = 1.0
    
    # Compute actual
    T_actual = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    if not np.allclose(T_actual, T_expected, atol=1e-10):
        # Fallback to rank 28 Strassen
        rank28 = 28
        U28 = [[0.0] * rank28 for _ in range(nm)]
        V28 = [[0.0] * rank28 for _ in range(mp)]
        W28 = [[0.0] * rank28 for _ in range(pn)]
        
        def add_strassen_block(col_off, X, Y, Z):
            # M1
            U28[X[0]][col_off+0] += 1; U28[X[3]][col_off+0] += 1
            V28[Y[0]][col_off+0] += 1; V28[Y[3]][col_off+0] += 1
            W28[Z[0]][col_off+0] += 1; W28[Z[3]][col_off+0] += 1
            # M2
            U28[X[2]][col_off+1] += 1; U28[X[3]][col_off+1] += 1
            V28[Y[0]][col_off+1] += 1
            W28[Z[2]][col_off+1] += 1; W28[Z[3]][col_off+1] -= 1
            # M3
            U28[X[0]][col_off+2] += 1
            V28[Y[1]][col_off+2] += 1; V28[Y[3]][col_off+2] -= 1
            W28[Z[1]][col_off+2] += 1; W28[Z[3]][col_off+2] += 1
            # M4
            U28[X[3]][col_off+3] += 1
            V28[Y[2]][col_off+3] += 1; V28[Y[0]][col_off+3] -= 1
            W28[Z[0]][col_off+3] += 1; W28[Z[2]][col_off+3] += 1
            # M5
            U28[X[0]][col_off+4] += 1; U28[X[1]][col_off+4] += 1
            V28[Y[3]][col_off+4] += 1
            W28[Z[0]][col_off+4] -= 1; W28[Z[1]][col_off+4] += 1
            # M6
            U28[X[2]][col_off+5] += 1; U28[X[0]][col_off+5] -= 1
            V28[Y[0]][col_off+5] += 1; V28[Y[1]][col_off+5] += 1
            W28[Z[3]][col_off+5] += 1
            # M7
            U28[X[1]][col_off+6] += 1; U28[X[3]][col_off+6] -= 1
            V28[Y[2]][col_off+6] += 1; V28[Y[3]][col_off+6] += 1
            W28[Z[0]][col_off+6] += 1
        
        X1=[0,1,4,5]; Y1=[0,1,4,5]; Z1=[0,2,1,3]
        add_strassen_block(0, X1, Y1, Z1)
        X2=[2,3,6,7]; Y2=[8,9,12,13]; Z2=[0,2,1,3]
        add_strassen_block(7, X2, Y2, Z2)
        X3=[0,1,4,5]; Y3=[2,3,6,7]; Z3=[4,6,5,7]
        add_strassen_block(14, X3, Y3, Z3)
        X4=[2,3,6,7]; Y4=[10,11,14,15]; Z4=[4,6,5,7]
        add_strassen_block(21, X4, Y4, Z4)
        
        return {"U": U28, "V": V28, "W": W28}
    
    return {"U": U, "V": V, "W": W}
