"""Rewritten program for 3x4x5 matrix multiplication tensor decomposition."""
import time
import random

DIMENSIONS = [3, 4, 5]

def _strassen_222():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, t], V[i*2+j, t], W[i*2+j, t]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen 7 terms:
    # M0 = A00*B00
    # M1 = A01*B11
    # M2 = (A00+A11)*(B00+B11)
    # M3 = (A00+A01)*B11
    # M4 = (A10+A11)*B00
    # M5 = A10*(B11-B00)
    # M6 = A01*(B00-B10)
    #
    # C00 = M0 + M2 - M4 - M6
    # C01 = M1 + M3 - M2 - M5
    # C10 = M0 + M4 - M1 - M3
    # C11 = M1 + M2 - M0 - M4
    
    # U coefficients (for A indices i*2+j):
    # A00 = row 0, A01 = row 1, A10 = row 2, A11 = row 3
    U[0][0] = 1  # A00 in M0
    U[1][1] = 1  # A01 in M1
    U[0][2] = 1  # A00 in M2
    U[3][2] = 1  # A11 in M2
    U[0][3] = 1  # A00 in M3
    U[1][3] = 1  # A01 in M3
    U[2][4] = 1  # A10 in M4
    U[3][4] = 1  # A11 in M4
    U[2][5] = 1  # A10 in M5
    U[1][6] = 1  # A01 in M6
    
    # V coefficients (for B indices i*2+j):
    # B00 = row 0, B01 = row 1, B10 = row 2, B11 = row 3
    V[0][0] = 1  # B00 in M0
    V[3][1] = 1  # B11 in M1
    V[0][2] = 1  # B00 in M2
    V[3][2] = 1  # B11 in M2
    V[3][3] = 1  # B11 in M3
    V[0][4] = 1  # B00 in M4
    V[3][5] = 1  # B11 in M5
    V[0][5] = -1 # B00 in M5 (negative)
    V[0][6] = 1  # B00 in M6
    V[2][6] = -1 # B10 in M6 (negative)
    
    # W coefficients (for C indices i*2+j):
    # C00 = row 0, C01 = row 1, C10 = row 2, C11 = row 3
    # C00 = M0 + M2 - M4 - M6
    W[0][0] = 1; W[0][2] = 1; W[0][4] = -1; W[0][6] = -1
    # C01 = M1 + M3 - M2 - M5
    W[1][1] = 1; W[1][3] = 1; W[1][2] = -1; W[1][5] = -1
    # C10 = M0 + M4 - M1 - M3
    W[2][0] = 1; W[2][4] = 1; W[2][1] = -1; W[2][3] = -1
    # C11 = M1 + M2 - M0 - M4
    W[3][1] = 1; W[3][2] = 1; W[3][0] = -1; W[3][4] = -1
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 5
    
    # Partition indices:
    # i (rows of A, cols of C): 0,1 | 2  -> sizes 2, 1
    # j (cols of A, rows of B): 0,1 | 2,3 -> sizes 2, 2
    # k (cols of B, rows of C): 0,1 | 2,3 | 4 -> sizes 2, 2, 1
    
    # Block structure:
    # For each block (i_block, j_block, k_block):
    #   i_sizes: [2, 1]
    #   j_sizes: [2, 2]
    #   k_sizes: [2, 2, 1]
    
    i_blocks = [(0, 2), (2, 1)]
    j_blocks = [(0, 2), (2, 2)]
    k_blocks = [(0, 2), (2, 2), (4, 1)]
    
    all_U = []
    all_V = []
    all_W = []
    
    for (i0, isz) in i_blocks:
        for (j0, jsz) in j_blocks:
            for (k0, ksz) in k_blocks:
                # Block dimensions: isz x jsz x ksz
                if isz == 2 and jsz == 2 and ksz == 2:
                    # Use Strassen 2x2x2
                    Ub, Vb, Wb = _strassen_222()
                    for c in range(7):
                        u_col = [0]*(n*m)
                        v_col = [0]*(m*p)
                        w_col = [0]*(p*n)
                        for ii in range(4):
                            u_col[(i0 + ii//2)*m + (j0 + ii%2)] = Ub[ii][c]
                            v_col[(j0 + ii//2)*p + (k0 + ii%2)] = Vb[ii][c]
                            w_col[(k0 + ii//2)*n + (i0 + ii%2)] = Wb[ii][c]
                        all_U.append(u_col)
                        all_V.append(v_col)
                        all_W.append(w_col)
                else:
                    # Schoolbook for small blocks
                    for ii in range(isz):
                        for jj in range(jsz):
                            for kk in range(ksz):
                                u_col = [0]*(n*m)
                                v_col = [0]*(m*p)
                                w_col = [0]*(p*n)
                                u_col[(i0+ii)*m + (j0+jj)] = 1
                                v_col[(j0+jj)*p + (k0+kk)] = 1
                                w_col[(k0+kk)*n + (i0+ii)] = 1
                                all_U.append(u_col)
                                all_V.append(v_col)
                                all_W.append(w_col)
    
    rank = len(all_U)
    
    # Convert to matrices
    U = [list(col) for col in all_U]
    V = [list(col) for col in all_V]
    W = [list(col) for col in all_W]
    
    # Transpose to correct shapes: U is (n*m, rank)
    U_final = [[U[i][c] for i in range(n*m)] for c in range(rank)]
    V_final = [[V[i][c] for i in range(m*p)] for c in range(rank)]
    W_final = [[W[i][c] for i in range(p*n)] for c in range(rank)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
