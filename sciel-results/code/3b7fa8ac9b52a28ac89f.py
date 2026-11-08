"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import itertools
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build tensor from factors
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for r in range(rank):
        T += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    # Compare with true tensor
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(mat):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    for row in mat:
        for val in row:
            if isinstance(val, (int, np.integer)):
                continue
            v = float(val)
            if abs(2 * v - round(2 * v)) > 1e-9:
                return False
    return True

def _to_rational_list(mat):
    """Convert a matrix of floats (that are half-integers) to list of Fractions."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = float(val)
            # Convert to fraction: multiply by 2, round, divide by 2
            r.append(Fraction(round(2 * v), 2))
        result.append(r)
    return result

def _get_strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2 by 2x2 matrix multiplication using Strassen's algorithm (Rank 7).
    U: 4x7, V: 4x7, W: 4x7
    Indices for A, B, C: 0=(1,1), 1=(1,2), 2=(2,1), 3=(2,2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_schoolbook_2x2_2x3():
    """
    Returns U, V, W for 2x2 by 2x3 matrix multiplication using Schoolbook (Rank 12).
    U: 4x12, V: 6x12, W: 6x12
    A indices: 0=(1,1), 1=(1,2), 2=(2,1), 3=(2,2)
    B indices: 0=(1,1), 1=(1,2), 2=(1,3), 3=(2,1), 4=(2,2), 5=(2,3)
    C indices: 0=(1,1), 1=(1,2), 2=(1,3), 3=(2,1), 4=(2,2), 5=(2,3)
    """
    U = np.zeros((4, 12), dtype=int)
    V = np.zeros((6, 12), dtype=int)
    W = np.zeros((6, 12), dtype=int)
    
    # C11 = A11*B11 + A12*B21
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    
    # C12 = A11*B12 + A12*B22
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    U[1, 3] = 1; V[4, 3] = 1; W[1, 3] = 1
    
    # C13 = A11*B13 + A12*B23
    U[0, 4] = 1; V[2, 4] = 1; W[2, 4] = 1
    U[1, 5] = 1; V[5, 5] = 1; W[2, 5] = 1
    
    # C21 = A21*B11 + A22*B21
    U[2, 6] = 1; V[0, 6] = 1; W[3, 6] = 1
    U[3, 7] = 1; V[3, 7] = 1; W[3, 7] = 1
    
    # C22 = A21*B12 + A22*B22
    U[2, 8] = 1; V[1, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[4, 9] = 1; W[4, 9] = 1
    
    # C23 = A21*B13 + A22*B23
    U[2, 10] = 1; V[2, 10] = 1; W[5, 10] = 1
    U[3, 11] = 1; V[5, 11] = 1; W[5, 11] = 1
    
    return U, V, W

def _construct_rank_52():
    """
    Construct a rank-52 decomposition for 2x4 by 4x7.
    Splits A into two 2x2 blocks (A1, A2).
    Splits B into two 2x7 blocks (B1, B2).
    Splits B1 into B1a(2x2), B1b(2x2), B1c(2x3).
    Splits B2 into B2a(2x2), B2b(2x2), B2c(2x3).
    
    Uses Strassen (rank 7) for 2x2 by 2x2.
    Uses Schoolbook (rank 12) for 2x2 by 2x3.
    
    Total rank = 2 * (7 + 7 + 12) = 52.
    """
    n, m, p = 2, 4, 7
    rank = 52
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U_s, V_s, W_s = _get_strassen_2x2_2x2()
    U_b, V_b, W_b = _get_schoolbook_2x2_2x3()
    
    # Helper to add a local decomposition to global
    def add_block(U_local, V_local, W_local, a_offset, b_offset, c_offset, col_offset):
        U[a_offset:a_offset+4, col_offset:col_offset+U_local.shape[1]] = U_local
        V[b_offset:b_offset+V_local.shape[0], col_offset:col_offset+V_local.shape[1]] = V_local
        W[c_offset:c_offset+W_local.shape[0], col_offset:col_offset+W_local.shape[1]] = W_local

    # Block 1: A1 * B1
    # A1 indices: 0-3
    # B1 indices: 0-13
    #   B1a: 0-3
    #   B1b: 4-7
    #   B1c: 8-13
    # C1 indices: 0-13
    #   C1a: 0-3
    #   C1b: 4-7
    #   C1c: 8-13
    
    # A1 * B1a (Strassen)
    add_block(U_s, V_s, W_s, 
              a_offset=0, 
              b_offset=0, 
              c_offset=0, 
              col_offset=0)
              
    # A1 * B1b (Strassen)
    add_block(U_s, V_s, W_s, 
              a_offset=0, 
              b_offset=4, 
              c_offset=4, 
              col_offset=7)
              
    # A1 * B1c (Schoolbook)
    add_block(U_b, V_b, W_b, 
              a_offset=0, 
              b_offset=8, 
              c_offset=8, 
              col_offset=14)
              
    # Block 2: A2 * B2
    # A2 indices: 4-7
    # B2 indices: 14-27
    #   B2a: 14-17
    #   B2b: 18-21
    #   B2c: 22-27
    # C2 indices: 0-13 (same as C1, they sum up)
    
    # A2 * B2a (Strassen)
    add_block(U_s, V_s, W_s, 
              a_offset=4, 
              b_offset=14, 
              c_offset=0, 
              col_offset=26)
              
    # A2 * B2b (Strassen)
    add_block(U_s, V_s, W_s, 
              a_offset=4, 
              b_offset=18, 
              c_offset=4, 
              col_offset=33)
              
    # A2 * B2c (Schoolbook)
    add_block(U_b, V_b, W_b, 
              a_offset=4, 
              b_offset=22, 
              c_offset=8, 
              col_offset=40)
              
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return U_list, V_list, W_list

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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
    return U, V, W

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    
    # Try the rank-52 construction
    try:
        U, V, W = _construct_rank_52()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
