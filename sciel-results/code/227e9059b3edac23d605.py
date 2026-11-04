import time
import sys

def construct():
    """
    Constructs a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    Split A (2x3) into A1 (2x2) and A2 (2x1).
    Split B (3x8) into B1 (2x8) and B2 (1x8).
    C = A1*B1 + A2*B2.
    
    A1*B1: Treat B1 as four 2x2 blocks. Use Strassen (rank 7) for each. Total 28.
    A2*B2: Outer product of 2x1 and 1x8. Use schoolbook (rank 16). Total 16.
    Total rank: 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        for r in range(7):
            # Fill U
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val
                        
            # Fill V
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val
                        
            # Fill W
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val
                        
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is column 2 of A. B2 is row 2 of B.
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
