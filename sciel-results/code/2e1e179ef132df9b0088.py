import time
import numpy as np

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication using Strassen's algorithm.
    U: 4 x 7 (indices i*2+j for i,j in 0..1)
    V: 4 x 7 (indices k*2+l for k,l in 0..1)
    W: 4 x 7 (indices k*2+i for k,i in 0..1)  # Output C[k,i] = sum A[i,j] B[j,k]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen components for C = A * B where A is 2x2, B is 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b01 - b10)
    # M5 = (a00 + a10) * b11
    # M6 = (a11 - a00) * (b00 + b10)
    # M7 = (a10 - a01) * (b11 + b01)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V indices: 0=b00, 1=b01, 2=b10, 3=b11
    # W indices: 0=c00 (k=0,i=0), 1=c01 (k=0,i=1), 2=c10 (k=1,i=0), 3=c11 (k=1,i=1)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[0, 5] = -1; U[3, 5] = 1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = -1; U[2, 6] = 1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    """
    Returns a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    Split A (2x3) into A1 (2x2) and A2 (2x1).
    Split B (3x8) into B1 (2x8) and B2 (1x8).
    C = A1*B1 + A2*B2.
    
    A1*B1:
    Split B1 (2x8) into four 2x2 blocks.
    Use Strassen's algorithm (Rank 7) for each 2x2 * 2x2 multiplication.
    Total rank for this part: 4 * 7 = 28.
    
    A2*B2:
    A2 is 2x1, B2 is 1x8. This is an outer product.
    Rank is 2 * 8 = 16.
    
    Total Rank: 28 + 16 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))  # 6 x 44
    V = np.zeros((m * p, rank))  # 24 x 44
    W = np.zeros((p * n, rank))  # 16 x 44
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0, 1 of B. Split into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U for A1
            # Global A indices: i in 0..1, a in 0..1
            # Local index la in 0..1, li in 0..1
            for li in range(2):
                for la in range(2):
                    val = S_U[li * 2 + la, r]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx, col] += val
            
            # Fill V for B1 block
            # Global B indices: a in 0..1, k in k_start..k_start+1
            for la in range(2):
                for lk in range(2):
                    val = S_V[la * 2 + lk, r]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx, col] += val
            
            # Fill W for C block
            # Global C indices: k in k_start..k_start+1, i in 0..1
            for lk in range(2):
                for li in range(2):
                    val = S_W[lk * 2 + li, r]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx, col] += val
            
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is col 2 of A. B2 is row 2 of B.
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
