import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.
    
    We split the 4x4 middle dimension into two 2x2 blocks (left and right halves).
    C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    Each term is a 2x2 @ 2x6 multiplication. We further split the 2x6 into three 2x2 blocks.
    Each 2x2 @ 2x2 uses Strassen's algorithm (rank 7).
    
    Total rank = 2 (halves) * 3 (sub-blocks) * 7 (Strassen) = 42
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column index (0..5), i is row index (0..1)
    So W index = k*2 + i for output C[i][k]
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=np.float64)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=np.float64)  # 12 x 42
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # P1 = (a11+a22)*(b11+b22)
    # P2 = (a21+a22)*b11
    # P3 = a11*(b12-b22)
    # P4 = a22*(b21-b11)
    # P5 = (a11+a12)*b22
    # P6 = (a21-a11)*(b11+b12)
    # P7 = (a12-a22)*(b21+b22)
    #
    # c11 = P1 + P4 - P5 + P7
    # c12 = P3 + P5
    # c21 = P2 + P4
    # c22 = P1 - P2 + P3 + P6
    
    # For each product, store (U_entries, V_entries, W_entries) as dict {local_idx: sign}
    # Local indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    #                0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    #                0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    
    strassen_U = [
        {0: 1, 3: 1},       # P1: a11+a22
        {2: 1, 3: 1},       # P2: a21+a22
        {0: 1},             # P3: a11
        {3: 1},             # P4: a22
        {0: 1, 1: 1},       # P5: a11+a12
        {2: 1, 0: -1},      # P6: a21-a11
        {1: 1, 3: -1},      # P7: a12-a22
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: b11+b22
        {0: 1},             # P2: b11
        {1: 1, 3: -1},      # P3: b12-b22
        {2: 1, 0: -1},      # P4: b21-b11
        {3: 1},             # P5: b22
        {0: 1, 1: 1},       # P6: b11+b12
        {2: 1, 3: 1},       # P7: b21+b22
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c11+c22
        {2: 1, 3: -1},      # P2: c21-c22
        {1: 1, 3: 1},       # P3: c12+c22
        {0: 1, 2: 1},       # P4: c11+c21
        {0: -1, 1: 1},      # P5: -c11+c12
        {3: 1},             # P6: c22
        {0: 1},             # P7: c11
    ]
    
    col = 0
    
    # For each of the 2 halves (left/right) and 3 sub-blocks:
    # Left half: A cols 0-1, B rows 0-1
    # Right half: A cols 2-3, B rows 2-3
    for half in range(2):
        for sub in range(3):
            for prod_idx in range(7):
                col = half * 21 + sub * 7 + prod_idx
                
                # U index mapping:
                # A is 2x4, stored as row-major: A[i,j] -> U[i*4+j, col]
                # Left half (half=0): A cols 0-1
                # Right half (half=1): A cols 2-3
                # local 0 -> A[0][half*2+0], U idx = 0*4 + (half*2+0)
                # local 1 -> A[0][half*2+1], U idx = 0*4 + (half*2+1)
                # local 2 -> A[1][half*2+0], U idx = 1*4 + (half*2+0)
                # local 3 -> A[1][half*2+1], U idx = 1*4 + (half*2+1)
                local_U = {
                    0: 0 * 4 + (half * 2 + 0),
                    1: 0 * 4 + (half * 2 + 1),
                    2: 1 * 4 + (half * 2 + 0),
                    3: 1 * 4 + (half * 2 + 1),
                }
                for li, s in strassen_U[prod_idx].items():
                    U[local_U[li], col] += s
                
                # V index mapping:
                # B is 4x6, stored as row-major: B[i,j] -> V[i*6+j, col]
                # Left half (half=0): B rows 0-1
                # Right half (half=1): B rows 2-3
                # Sub-block sub: B cols 2*sub to 2*sub+1
                # local 0 -> B[half*2+0][2*sub+0], V idx = (half*2+0)*6 + (2*sub+0)
                # local 1 -> B[half*2+0][2*sub+1], V idx = (half*2+0)*6 + (2*sub+1)
                # local 2 -> B[half*2+1][2*sub+0], V idx = (half*2+1)*6 + (2*sub+0)
                # local 3 -> B[half*2+1][2*sub+1], V idx = (half*2+1)*6 + (2*sub+1)
                local_V = {
                    0: (half * 2 + 0) * 6 + (2 * sub + 0),
                    1: (half * 2 + 0) * 6 + (2 * sub + 1),
                    2: (half * 2 + 1) * 6 + (2 * sub + 0),
                    3: (half * 2 + 1) * 6 + (2 * sub + 1),
                }
                for li, s in strassen_V[prod_idx].items():
                    V[local_V[li], col] += s
                
                # W index mapping:
                # Output C is 2x6, W index = k*n+i where k is column, i is row
                # So W[k*2+i, col] for C[i][k]
                # local 0 -> C[0][2*sub+0], W idx = (2*sub+0)*2 + 0
                # local 1 -> C[0][2*sub+1], W idx = (2*sub+1)*2 + 0
                # local 2 -> C[1][2*sub+0], W idx = (2*sub+0)*2 + 1
                # local 3 -> C[1][2*sub+1], W idx = (2*sub+1)*2 + 1
                local_W = {
                    0: (2 * sub + 0) * 2 + 0,
                    1: (2 * sub + 1) * 2 + 0,
                    2: (2 * sub + 0) * 2 + 1,
                    3: (2 * sub + 1) * 2 + 1,
                }
                for li, s in strassen_W[prod_idx].items():
                    W[local_W[li], col] += s
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
