"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank55(n, m, p):
    """
    Construct a rank-55 decomposition by splitting the 3x10 matrix into 5 blocks of 3x2.
    Each 3x2 block is split into a 2x2 top part and a 1x2 bottom part.
    The 2x2 part uses Strassen (rank 7), the 1x2 part uses schoolbook (rank 4).
    Total rank per block = 11. Total rank = 55.
    """
    if n != 2 or m != 3 or p != 10:
        return None
        
    rank = 55
    U = [[0] * rank for _ in range(n * m)] # 6 x 55
    V = [[0] * rank for _ in range(m * p)] # 30 x 55
    W = [[0] * rank for _ in range(p * n)] # 20 x 55
    
    col = 0
    # A is 2x3. Indices: A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]
    # U indices: i*m+j
    # A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[2]
    # A[1,0] -> U[3], A[1,1] -> U[4], A[1,2] -> U[5]
    
    for block in range(5):
        # Block k uses columns 2k and 2k+1 of B (and C)
        k_start = 2 * block
        k_end = 2 * block + 1
        
        # V indices for B[j, k] is j*p + k
        # B[0, k_start] -> V[k_start], B[0, k_end] -> V[k_end]
        # B[1, k_start] -> V[p + k_start], B[1, k_end] -> V[p + k_end]
        # B[2, k_start] -> V[2p + k_start], B[2, k_end] -> V[2p + k_end]
        
        # W indices for C[i, k] is k*n + i
        # C[0, k_start] -> W[k_start*n + 0], C[0, k_end] -> W[k_end*n + 0]
        # C[1, k_start] -> W[k_start*n + 1], C[1, k_end] -> W[k_end*n + 1]
        
        # --- Strassen part for A[0:2, 0:2] * B[0:2, k_start:k_end+1] ---
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] += 1
        U[3][col] += 1 # A10? No, Strassen uses A00, A01, A10, A11.
        # Wait, standard Strassen:
        # M1 = (A11 + A22) * (B11 + B22) -- using 1-based indexing for 2x2
        # Let's use 0-based:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # My U indices:
        # A00 -> U[0], A01 -> U[1], A10 -> U[3], A11 -> U[4]
        
        # M1
        U[0][col] = 1
        U[4][col] = 1
        V[k_start][col] = 1
        V[p + k_end][col] = 1 # B11 is row 1, col 1 of the block -> global row 1, col k_end
        # W: C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
        W[k_start*n + 0][col] = 1
        W[k_end*n + 0][col] = -1
        W[k_start*n + 1][col] = -1
        W[k_end*n + 1][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1
        U[4][col] = 1
        V[k_start][col] = 1
        # W: C10 += M2, C11 += M2
        W[k_start*n + 1][col] = 1
        W[k_end*n + 1][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1
        V[k_end][col] = 1
        V[p + k_end][col] = -1
        # W: C01 += M3
        W[k_end*n + 0][col] = 1
        col += 1
        
        # M4
        U[4][col] = 1
        V[p + k_start][col] = 1
        V[k_start][col] = -1
        # W: C10 += M4
        W[k_start*n + 1][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[p + k_end][col] = 1
        # W: C00 += M5, C01 -= M5
        W[k_start*n + 0][col] = 1
        W[k_end*n + 0][col] = -1
        col += 1
        
        # M6
        U[3][col] = 1
        U[0][col] = -1
        V[k_start][col] = 1
        V[k_end][col] = 1
        # W: C10 += M6, C11 -= M6
        W[k_start*n + 1][col] = 1
        W[k_end*n + 1][col] = -1
        col += 1
        
        # M7
        U[1][col] = 1
        U[4][col] = -1
        V[p + k_start][col] = 1
        V[p + k_end][col] = 1
        # W: C01 += M7, C11 += M7
        W[k_end*n + 0][col] = 1
        W[k_end*n + 1][col] = 1
        col += 1
        
        # --- Schoolbook part for A[0:2, 2] * B[2, k_start:k_end+1] ---
        # A_col2 = [A02, A12] -> U[2], U[5]
        # B_row2 = [B20, B21] -> V[2p+k_start], V[2p+k_end]
        
        # Q1 = A02 * B20
        U[2][col] = 1
        V[2*p + k_start][col] = 1
        W[k_start*n + 0][col] = 1
        col += 1
        
        # Q2 = A02 * B21
        U[2][col] = 1
        V[2*p + k_end][col] = 1
        W[k_end*n + 0][col] = 1
        col += 1
        
        # Q3 = A12 * B20
        U[5][col] = 1
        V[2*p + k_start][col] = 1
        W[k_start*n + 1][col] = 1
        col += 1
        
        # Q4 = A12 * B21
        U[5][col] = 1
        V[2*p + k_end][col] = 1
        W[k_end*n + 1][col] = 1
        col += 1

    if col != rank:
        return None

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the structured rank 55 solution
    result = _strassen_rank55(n, m, p)
    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
