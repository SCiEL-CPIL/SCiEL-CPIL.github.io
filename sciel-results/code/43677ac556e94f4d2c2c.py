"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # 150s wall-clock, 5s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition: rank = n*m*p = 36
    # We will try to improve upon this using block decomposition with Strassen.
    
    # Strategy: Split the 3x3x4 matmul into blocks and apply Strassen where possible.
    # 
    # Split: i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}
    # 
    # We can group:
    #   Block A: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 (Strassen: rank 7)
    #   Block B: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 (Strassen: rank 7)
    #   Block C: i in {0,1}, j = 2, k in {0,1} -> 2x1x2 (schoolbook: rank 4)
    #   Block D: i in {0,1}, j = 2, k in {2,3} -> 2x1x2 (schoolbook: rank 4)
    #   Block E: i = 2, j in {0,1}, k in {0,1} -> 1x2x2 (schoolbook: rank 4)
    #   Block F: i = 2, j in {0,1}, k in {2,3} -> 1x2x2 (schoolbook: rank 4)
    #   Block G: i = 2, j = 2, k in {0,1} -> 1x1x2 (schoolbook: rank 2)
    #   Block H: i = 2, j = 2, k in {2,3} -> 1x1x2 (schoolbook: rank 2)
    #
    # Total: 7+7+4+4+4+4+2+2 = 34
    
    # Strassen 2x2x2 tensor decomposition:
    # For C = A*B where A, B, C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 + M3 - M2 + M6
    #
    # In tensor form for each multiplication r, we need:
    # U[i,j] * V[j,k] * W[k,i] summed over r = delta (the matmul tensor)
    #
    # For Strassen, the 7 terms have the following (U, V, W) coefficients:
    # Term 1: U = [[1,0],[1,0]] wait no...
    
    # Let me be very careful. The matmul tensor T[i,j,k] = 1 for all i,j,k (in the 2x2x2 case).
    # We want sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    # Strassen's 7 multiplications:
    # Each multiplication r computes (linear combo of A entries) * (linear combo of B entries)
    # and contributes to C entries.
    
    # U_r[i,j] = coefficient of A[i,j] in the first factor of multiplication r
    # V_r[j,k] = coefficient of B[j,k] in the second factor of multiplication r  
    # W_r[k,i] = coefficient of C[i,k] from multiplication r
    
    # M1 = (A00+A11)(B00+B11)
    #   U1: A00=1, A11=1 -> U1[0,0]=1, U1[1,1]=1, others 0
    #   V1: B00=1, B11=1 -> V1[0,0]=1, V1[1,1]=1, others 0
    #   W1: contributes to C00, C01, C10, C11
    #   C00 gets +1 from M1, C01 gets +1, C10 gets +1, C11 gets +1
    #   W1[0,0]=1 (C[0,0]), W1[1,0]=1 (C[0,1]), W1[0,1]=1 (C[1,0]), W1[1,1]=1 (C[1,1])
    
    # Wait, W is indexed W[k*n+i] where n=2 for the 2x2 case.
    # W[k,i] is the coefficient for C[i,k].
    
    # Let me just verify by direct computation that Strassen works:
    
    def strassen_rank():
        """Return the 7 (U,V,W) triplets for 2x2x2 Strassen."""
        # U[r][i,j], V[r][j,k], W[r][k,i] for r in 0..6
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        U1 = [[1,0],[0,1]]  # A00 + A11
        V1 = [[1,0],[0,1]]  # B00 + B11
        W1 = [[1,1],[1,1]]  # C00+=1, C01+=1, C10+=1, C11+=1
        # W1[k,i]: W1[0,0]=C[0,0], W1[1,0]=C[0,1], W1[0,1]=C[1,0], W1[1,1]=C[1,1]
        terms.append((U1, V1, W1))
        
        # M2 = (A10+A11)(B00)
        U2 = [[0,0],[1,1]]  # A10 + A11
        V2 = [[1,0],[0,0]]  # B00
        W2 = [[0,0],[1,0]]  # C10 += 1
        # W2[0,0]=C[0,0]=0, W2[1,0]=C[0,1]=0, W2[0,1]=C[1,0]=1, W2[1,1]=C[1,1]=0
        terms.append((U2, V2, W2))
        
        # M3 = A00(B10-B11)
        U3 = [[1,0],[0,0]]  # A00
        V3 = [[0,0],[1,-1]]  # B10 - B11
        W3 = [[0,1],[0,0]]  # C01 += 1
        # W3[0,0]=C[0,0]=0, W3[1,0]=C[0,1]=1, W3[0,1]=C[1,0]=0, W3[1,1]=C[1,1]=0
        terms.append((U3, V3, W3))
        
        # M4 = A11(B01-B00)
        U4 = [[0,0],[0,1]]  # A11
        V4 = [[-1,1],[0,0]]  # B01 - B00
        W4 = [[1,-1],[0,0]]  # C00 -= 1, C01 += 1
        # W4[0,0]=C[0,0]=-1, W4[1,0]=C[0,1]=1, W4[0,1]=C[1,0]=0, W4[1,1]=C[1,1]=0
        terms.append((U4, V4, W4))
        
        # M5 = (A00+A01)(B11)
        U5 = [[1,1],[0,0]]  # A00 + A01
        V5 = [[0,0],[0,1]]  # B11
        W5 = [[1,0],[0,0]]  # C00 += 1
        terms.append((U5, V5, W5))
        
        # M6 = (A10-A00)(B00+B10)
        U6 = [[-1,0],[1,0]]  # A10 - A00
        V6 = [[1,0],[1,0]]  # B00 + B10
        W6 = [[0,0],[0,1]]  # C11 += 1
        terms.append((U6, V6, W6))
        
        # M7 = (A01-A11)(B01+B11)
        U7 = [[0,1],[0,-1]]  # A01 - A11
        V7 = [[0,1],[0,1]]  # B01 + B11
        W7 = [[1,0],[1,0]]  # C00 += 1, C10 += 1
        # W7[0,0]=C[0,0]=1, W7[1,0]=C[0,1]=0, W7[0,1]=C[1,0]=1, W7[1,1]=C[1,1]=0
        terms.append((U7, V7, W7))
        
        return terms
    
    # Verify Strassen 2x2x2
    strassen_terms = strassen_rank()
    valid_strassen = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for (Ur, Vr, Wr) in strassen_terms:
                    s += Ur[i][j] * Vr[j][k] * Wr[k][i]
                if abs(s - 1.0) > 1e-10:
                    valid_strassen = False
                    break
            if not valid_strassen:
                break
        if not valid_strassen:
            break
    
    if not valid_strassen:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the 3x3x4 decomposition using blocks
    # Total rank: 7+7+4+4+4+4+2+2 = 34
    total_rank = 34
    U = [[0]*total_rank for _ in range(n*m)]  # 9 x 34
    V = [[0]*total_rank for _ in range(m*p)]  # 12 x 34
    W = [[0]*total_rank for _ in range(p*n)]  # 12 x 34
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        """Add Strassen 2x2x2 block at offsets (i_off, j_off, k_off)."""
        nonlocal col
        for (Ur, Vr, Wr) in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if Ur[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj)][col] += Ur[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if Vr[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk)][col] += Vr[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if Wr[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii)][col] += Wr[kk][ii]
            col += 1
    
    def add_schoolbook_block(i_indices, j_indices, k_indices):
        """Add schoolbook terms for given index sets."""
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    # Block A: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
    add_strassen_block(0, 0, 0)
    
    # Block B: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen
    add_strassen_block(0, 0, 2)
    
    # Block C: i in {0,1}, j = 2, k in {0,1} -> schoolbook
    add_schoolbook_block([0,1], [2], [0,1])
    
    # Block D: i in {0,1}, j = 2, k in {2,3} -> schoolbook
    add_schoolbook_block([0,1], [2], [2,3])
    
    # Block E: i = 2, j in {0,1}, k in {0,1} -> schoolbook
    add_schoolbook_block([2], [0,1], [0,1])
    
    # Block F: i = 2, j in {0,1}, k in {2,3} -> schoolbook
    add_schoolbook_block([2], [0,1], [2,3])
    
    # Block G: i = 2, j = 2, k in {0,1} -> schoolbook
    add_schoolbook_block([2], [2], [0,1])
    
    # Block H: i = 2, j = 2, k in {2,3} -> schoolbook
    add_schoolbook_block([2], [2], [2,3])
    
    # Validate
    if time.time() - t0 < DEADLINE:
        if not _validate(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
