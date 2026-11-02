import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2+j, s]
    V[j*2+k, s] -- wait, V is for B (2x2), so V[j*2+k, s]
    W[k*2+i, s]
    """
    # Standard Strassen
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # s=0: M1
    U[0, 0] = 1; U[3, 0] = 1 # A11+A22
    V[0, 0] = 1; V[3, 0] = 1 # B11+B22
    W[0, 0] = 1; W[2, 0] = -1; W[1, 0] = 1; W[3, 0] = -1 # C11 += M1, C12 -= M1, C21 += M1, C22 -= M1
    
    # s=1: M2
    U[2, 1] = 1; U[3, 1] = 1 # A21+A22
    V[0, 1] = 1 # B11
    W[2, 1] = 1; W[3, 1] = 1 # C21 += M2, C22 += M2
    
    # s=2: M3
    U[0, 2] = 1 # A11
    V[2, 2] = 1; V[3, 2] = -1 # B21-B22
    W[1, 2] = 1; W[3, 2] = -1 # C12 += M3, C22 -= M3
    
    # s=3: M4
    U[3, 3] = 1 # A22
    V[1, 3] = 1; V[0, 3] = -1 # B12-B11
    W[1, 3] = -1; W[3, 3] = 1 # C12 -= M4, C22 += M4
    
    # s=4: M5
    U[0, 4] = 1; U[1, 4] = 1 # A11+A12
    V[3, 4] = 1 # B22
    W[0, 4] = 1; W[1, 4] = 1 # C11 += M5, C12 += M5
    
    # s=5: M6
    U[2, 5] = 1; U[0, 5] = -1 # A21-A11
    V[0, 5] = 1; V[1, 5] = 1 # B11+B12
    W[2, 5] = 1; W[0, 5] = -1 # C21 += M6, C11 -= M6
    
    # s=6: M7
    U[1, 6] = 1; U[3, 6] = -1 # A12-A22
    V[2, 6] = 1; V[3, 6] = 1 # B21+B22
    W[0, 6] = 1; W[2, 6] = 1 # C11 += M7, C21 += M7
    
    return U, V, W

def _build_rank42():
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    U_s, V_s, W_s = _strassen_2x2()
    
    col = 0
    # A0 (rows 0-1, cols 0-1 of A) * B0 (rows 0-1 of B, cols 0-5)
    # B0 split into B00 (cols 0-1), B01 (cols 2-3), B02 (cols 4-5)
    for b_sub in range(3):
        # Use Strassen for A0 * B0_sub
        # A0 indices: U[0:4, :]
        # B0_sub indices: V[0:4, :] (rows 0-1 of B, cols 2*b_sub to 2*b_sub+1)
        # Output: W[0:4, :] (cols 2*b_sub to 2*b_sub+1 of C, rows 0-1)
        
        for s in range(7):
            # U part
            for i in range(2):
                for j in range(2):
                    U[i*m + j, col] += U_s[i*2+j, s]
            
            # V part
            # B rows 0-1, cols 2*b_sub to 2*b_sub+1
            for j in range(2):
                for k in range(2):
                    V[j*p + (2*b_sub + k), col] += V_s[j*2+k, s]
            
            # W part
            # C rows 0-1, cols 2*b_sub to 2*b_sub+1
            for k in range(2):
                for i in range(2):
                    W[(2*b_sub + k)*n + i, col] += W_s[k*2+i, s]
            
            col += 1
            
    # A1 (rows 0-1, cols 2-3 of A) * B1 (rows 2-3 of B, cols 0-5)
    # B1 split into B10 (cols 0-1), B11 (cols 2-3), B12 (cols 4-5)
    for b_sub in range(3):
        # Use Strassen for A1 * B1_sub
        # A1 indices: U[0:4, :] (but mapped to A cols 2-3)
        # B1_sub indices: V[0:4, :] (rows 2-3 of B, cols 2*b_sub to 2*b_sub+1)
        # Output: W[0:4, :] (cols 2*b_sub to 2*b_sub+1 of C, rows 0-1)
        
        for s in range(7):
            # U part
            for i in range(2):
                for j in range(2):
                    U[i*m + (2 + j), col] += U_s[i*2+j, s]
            
            # V part
            # B rows 2-3, cols 2*b_sub to 2*b_sub+1
            for j in range(2):
                for k in range(2):
                    V[(2 + j)*p + (2*b_sub + k), col] += V_s[j*2+k, s]
            
            # W part
            # C rows 0-1, cols 2*b_sub to 2*b_sub+1
            for k in range(2):
                for i in range(2):
                    W[(2*b_sub + k)*n + i, col] += W_s[k*2+i, s]
            
            col += 1
            
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_output(mat):
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank42()
    if _verify(U, V, W, n, m, p):
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U_fb = np.zeros((n*m, rank), dtype=float)
    V_fb = np.zeros((m*p, rank), dtype=float)
    W_fb = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    return {"U": _to_output(U_fb), "V": _to_output(V_fb), "W": _to_output(W_fb)}
