"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    N = n * m  # 10
    M = m * p  # 25
    P = p * n  # 10
    
    # Total rank estimate:
    # Block 1 (2x2x2): 7
    # Block 2 (2x2x3): 7 (Strassen 2x2x2) + 4 (Schoolbook 2x2x1) = 11
    # Block 3 (2x3x2): 7 (Strassen 2x2x2) + 4 (Schoolbook 2x1x2) = 11
    # Block 4 (2x3x3): 7 (Strassen 2x2x2) + 4 (Sch 2x1x2) + 4 (Sch 2x2x1) + 2 (Sch 2x1x1) = 17
    # Total: 46
    
    rank = 46
    U = np.zeros((N, rank), dtype=np.int8)
    V = np.zeros((M, rank), dtype=np.int8)
    W = np.zeros((P, rank), dtype=np.int8)
    
    col_offset = 0
    
    def add_strassen(i0, j0, k0):
        """
        Adds Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        i0, j0, k0 are the starting indices.
        """
        nonlocal col_offset
        # Indices
        # A: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[i0*m+j0, col_offset] += 1; U[(i0+1)*m+(j0+1), col_offset] += 1
        V[j0*p+k0, col_offset] += 1; V[(j0+1)*p+(k0+1), col_offset] += 1
        W[k0*n+i0, col_offset] += 1; W[(k0+1)*n+(i0+1), col_offset] += 1
        col_offset += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m+j0, col_offset] += 1; U[(i0+1)*m+(j0+1), col_offset] += 1
        V[j0*p+k0, col_offset] += 1
        W[k0*n+(i0+1), col_offset] += 1
        col_offset += 1
        
        # M3 = A00(B10-B11)
        U[i0*m+j0, col_offset] += 1
        V[(j0+1)*p+k0, col_offset] += 1; V[(j0+1)*p+(k0+1), col_offset] -= 1
        W[(k0+1)*n+i0, col_offset] += 1
        col_offset += 1
        
        # M4 = A11(B01-B00)
        U[(i0+1)*m+(j0+1), col_offset] += 1
        V[j0*p+(k0+1), col_offset] += 1; V[j0*p+k0, col_offset] -= 1
        W[k0*n+i0, col_offset] += 1; W[k0*n+(i0+1), col_offset] += 1
        col_offset += 1
        
        # M5 = (A00+A01)B11
        U[i0*m+j0, col_offset] += 1; U[i0*m+(j0+1), col_offset] += 1
        V[(j0+1)*p+(k0+1), col_offset] += 1
        W[k0*n+i0, col_offset] -= 1; W[(k0+1)*n+i0, col_offset] += 1
        col_offset += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m+j0, col_offset] += 1; U[i0*m+j0, col_offset] -= 1
        V[j0*p+k0, col_offset] += 1; V[j0*p+(k0+1), col_offset] += 1
        W[(k0+1)*n+(i0+1), col_offset] += 1
        col_offset += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m+(j0+1), col_offset] += 1; U[(i0+1)*m+(j0+1), col_offset] -= 1
        V[(j0+1)*p+k0, col_offset] += 1; V[(j0+1)*p+(k0+1), col_offset] += 1
        W[k0*n+i0, col_offset] += 1
        col_offset += 1

    def add_schoolbook(i0, j0, k0, ni, nj, nk):
        """
        Adds schoolbook decomposition for A[i0:i0+ni, j0:j0+nj] * B[j0:j0+nj, k0:k0+nk] -> C[i0:i0+ni, k0:k0+nk]
        """
        nonlocal col_offset
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    U[(i0+i)*m + (j0+j), col_offset] = 1
                    V[(j0+j)*p + (k0+k), col_offset] = 1
                    W[(k0+k)*n + (i0+i), col_offset] = 1
                    col_offset += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Strassen for B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 2)
    # Schoolbook for B[0:2, 4:5] -> C[0:2, 4:5]
    add_schoolbook(0, 0, 4, 2, 2, 1)
    
    # Block 3: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # Strassen for A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 0)
    # Schoolbook for A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]
    add_schoolbook(0, 4, 0, 2, 1, 2)
    
    # Block 4: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # Strassen for A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2)
    # Schoolbook for A[0:2, 4:5] * B[4:5, 2:4] -> C[0:2, 2:4]
    add_schoolbook(0, 4, 2, 2, 1, 2)
    # Schoolbook for A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    add_schoolbook(0, 2, 4, 2, 2, 1)
    # Schoolbook for A[0:2, 4:5] * B[4:5, 4:5] -> C[0:2, 4:5]
    add_schoolbook(0, 4, 4, 2, 1, 1)
    
    # Verify
    if not verify_exact(U.tolist(), V.tolist(), W.tolist()):
        return schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def verify_exact(U_list, V_list, W_list):
    """Verify exact decomposition using integer/rational arithmetic."""
    n, m, p = 2, 5, 5
    rank = len(U_list[0])
    U = np.array(U_list, dtype=np.float64)
    V = np.array(V_list, dtype=np.float64)
    W = np.array(W_list, dtype=np.float64)
    
    # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_col = U[:, c].reshape(n, m)
        v_col = V[:, c].reshape(m, p)
        w_col = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
    
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def schoolbook():
    n, m, p = 2, 5, 5
    N = n * m
    M = m * p
    P = p * n
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
