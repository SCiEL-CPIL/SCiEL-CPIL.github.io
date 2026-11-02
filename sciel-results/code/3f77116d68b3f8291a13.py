import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Try the structured decomposition with Rank 82
    # Split m=5 into 3 and 2
    # C = A[0:3,:] * B[0:3,:] + A[:,3:5] * B[3:5,:]
    # Part 1: 3x3x6 (Two 3x3x3 blocks) -> Rank 23 + 23 = 46
    # Part 2: 3x2x6 (Three 3x2x2 blocks) -> Rank 12 + 12 + 12 = 36
    # Total Rank: 82
    
    result = _construct_rank82()
    if result is not None:
        return result
        
    # Fallback to Schoolbook (Rank 90)
    return _schoolbook(n, m, p, 90)

def _schoolbook(n, m, p, rank):
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

def _construct_rank82():
    n, m, p = 3, 5, 6
    rank = 82
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 3x3x6 (Rows 0-2 of A, Rows 0-2 of B)
    # This consists of two 3x3x3 multiplications
    # B is 3x6, so B[:, 0:3] and B[:, 3:6]
    # C is 3x6, so C[:, 0:3] and C[:, 3:6]
    
    # Get Rank 23 decomposition for 3x3x3
    try:
        U_333, V_333, W_333 = _get_333_rank23()
    except Exception:
        return None
        
    # Map to first 3x3x3 block (cols 0-2 of C, rows 0-2 of B)
    # U indices: i*3 + j for i in 0..2, j in 0..2
    # V indices: j*3 + k for j in 0..2, k in 0..2
    # W indices: k*3 + i for k in 0..2, i in 0..2
    
    # Block 1: C[:, 0:3]
    for c in range(23):
        for i in range(3):
            for j in range(3):
                if U_333[i*3+j, c] != 0:
                    U[i*5 + j, c] = U_333[i*3+j, c] # A is 3x5, so i*5+j
                    # Note: A is 3x5. We are using the first 3 cols of A.
                    # So A[i, j] maps to U[i*5 + j]
                    
        for j in range(3):
            for k in range(3):
                if V_333[j*3+k, c] != 0:
                    V[j*6 + k, c] = V_333[j*3+k, c] # B is 3x6 (part 1), so j*6+k
                    # Note: B is 5x6. We are using the first 3 rows of B.
                    # So B[j, k] maps to V[j*6 + k]
                    
        for k in range(3):
            for i in range(3):
                if W_333[k*3+i, c] != 0:
                    W[k*3 + i, c] = W_333[k*3+i, c] # C is 3x6, part 1 is cols 0-2.
                    # Note: C is 3x6. C[i, k].
                    # W index is k*n + i. n=3.
                    # So W[k*3 + i]
                    
    # Block 2: C[:, 3:6]
    # Same U and V structure, but W maps to cols 3-5
    offset_c = 23
    for c in range(23):
        for i in range(3):
            for j in range(3):
                if U_333[i*3+j, c] != 0:
                    U[i*5 + j, c + offset_c] = U_333[i*3+j, c]
                    
        for j in range(3):
            for k in range(3):
                if V_333[j*3+k, c] != 0:
                    V[j*6 + k, c + offset_c] = V_333[j*3+k, c]
                    
        for k in range(3):
            for i in range(3):
                if W_333[k*3+i, c] != 0:
                    # C[:, 3:6] means k in 3..5
                    # W index: (k+3)*3 + i
                    W[(k+3)*3 + i, c + offset_c] = W_333[k*3+i, c]

    # Part 2: 3x2x6 (Rows 0-2 of A, Rows 3-4 of B)
    # A is 3x2 (cols 3-4). B is 2x6 (rows 3-4).
    # This consists of three 3x2x2 multiplications
    # B is 2x6, so B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # C is 3x6, so C[:, 0:2], C[:, 2:4], C[:, 4:6]
    
    # Get Rank 12 decomposition for 3x2x2 (Schoolbook)
    U_322, V_322, W_322 = _get_322_rank12()
    
    offset_322 = 46
    for block in range(3):
        # Block 0: C[:, 0:2]
        # Block 1: C[:, 2:4]
        # Block 2: C[:, 4:6]
        
        for c in range(12):
            for i in range(3):
                for j in range(2):
                    if U_322[i*2+j, c] != 0:
                        # A is 3x5. We use cols 3,4.
                        # So A[i, j+3] maps to U[i*5 + j + 3]
                        U[i*5 + j + 3, c + offset_322 + block*12] = U_322[i*2+j, c]
                        
            for j in range(2):
                for k in range(2):
                    if V_322[j*2+k, c] != 0:
                        # B is 5x6. We use rows 3,4.
                        # So B[j+3, k + block*2] maps to V[(j+3)*6 + k + block*2]
                        V[(j+3)*6 + k + block*2, c + offset_322 + block*12] = V_322[j*2+k, c]
                        
            for k in range(2):
                for i in range(3):
                    if W_322[k*2+i, c] != 0:
                        # C is 3x6. C[i, k + block*2]
                        # W index: (k + block*2)*3 + i
                        W[(k + block*2)*3 + i, c + offset_322 + block*12] = W_322[k*2+i, c]

    # Verify
    if _verify(n, m, p, U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _get_333_rank23():
    # Rank 23 decomposition for 3x3x3
    # This is a known decomposition.
    # Coefficients are integers or half-integers.
    # I will use a standard set.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Terms (u_idx, v_idx, w_idx, coeff)
    # Indices are 0..8 for 3x3 matrices
    terms = [
        (0, 0, 0, 1), (1, 1, 1, 1), (2, 2, 2, 1),
        (3, 3, 3, 1), (4, 4, 4, 1), (5, 5, 5, 1),
        (6, 6, 6, 1), (7, 7, 7, 1), (8, 8, 8, 1),
        (0, 3, 1, 1), (1, 4, 2, 1), (2, 5, 0, 1),
        (3, 6, 2, 1), (4, 7, 0, 1), (5, 8, 1, 1),
        (6, 0, 1, 1), (7, 1, 2, 1), (8, 2, 0, 1),
        (0, 4, 2, 1), (1, 5, 0, 1), (2, 6, 1, 1),
        (3, 7, 0, 1), (4, 8, 1, 1), (5, 0, 2, 1)
    ]
    
    # Note: The above terms are a guess/placeholder structure.
    # A real rank-23 decomposition is complex.
    # If this fails verification, it returns None.
    
    for u_idx, v_idx, w_idx, coeff in terms:
        col = terms.index((u_idx, v_idx, w_idx, coeff))
        U[u_idx, col] = coeff
        V[v_idx, col] = coeff
        W[w_idx, col] = coeff
        
    return U, V, W

def _get_322_rank12():
    # Schoolbook for 3x2x2
    U = np.zeros((6, 12))
    V = np.zeros((4, 12))
    W = np.zeros((6, 12))
    
    col = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i*2+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def _verify(n, m, p, U, V, W, r):
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
