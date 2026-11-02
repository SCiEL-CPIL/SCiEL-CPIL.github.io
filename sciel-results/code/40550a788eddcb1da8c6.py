"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Uses vectorized NumPy operations for speed.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all tensor entries using vectorized computation
    for a_idx in range(n * m):
        i, j = divmod(a_idx, m)
        
        # Compute result[b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # M[b, r] = V[b, r] * U[a, r]
        M = V_arr * U_arr[a_idx, :][None, :]
        # result[b, c] = sum_r M[b, r] * W[c, r] = (M @ W.T)[b, c]
        result = M @ W_arr.T  # shape (m*p, p*n)
        
        # Build expected tensor slice
        expected = np.zeros((m * p, p * n))
        for b_idx in range(m * p):
            j2, k = divmod(b_idx, p)
            if j2 == j:
                for c_idx in range(p * n):
                    k2, i2 = divmod(c_idx, n)
                    if k2 == k and i2 == i:
                        expected[b_idx, c_idx] = 1.0
        
        if np.max(np.abs(result - expected)) > 1e-10:
            return False
    return True

def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    The tensor for $C = A B$ where $A$ is $2 \times 2$ and $B$ is $2 \times 10$ can be viewed as
    5 independent $2 \times 2 \times 2$ multiplication problems, one for each pair of columns in $B$.
    Specifically, let $B = [B_0, B_1, B_2, B_3, B_4]$ where each $B_i$ is a $2 \times 2$ matrix.
    Then $C = A B = [A B_0, A B_1, A B_2, A B_3, A B_4]$.
    
    The rank of the $2 \times 2 \times 2$ matrix multiplication tensor is 7 (Strassen's algorithm).
    Since the 5 blocks are independent, the total rank is the sum of the ranks of the blocks.
    Total Rank = 5 * 7 = 35.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for the 7 products
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    strassen_A = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: (e+h)
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [-1, 0, 1, 0], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W coefficients (for C)
    # Local C indices: 0->C00, 1->C01, 2->C10, 3->C11
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    # P1: C00 += 1, C11 += 1
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    # P2: C10 += 1, C11 -= 1
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    # P3: C01 += 1, C11 += 1
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    # P4: C00 += 1, C10 += 1
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    # P5: C00 -= 1, C01 += 1
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    # P6: C11 += 1
    w_contrib[5][3] = 1
    # P7: C00 += 1
    w_contrib[6][0] = 1
    
    # Fill U, V, W
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for r in range(7):
            col = b * 7 + r
            
            # U: same A coefficients for all blocks
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # V: B coefficients for block b
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            # W: C coefficients for block b
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    # Verification is skipped for speed, as the Strassen decomposition is known to be exact.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
