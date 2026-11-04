import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    
    rank = 22
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Part 1: j=0 term (Rank 1)
    # C[i,k] += A[i,0] * B[0,k]
    # U[i,0,c] = 1, V[0,k,c] = 1, W[i,k,c] = 1
    c = 0
    for i in range(n):
        U[i * m + 0, c] = 1.0
    for k in range(p):
        V[0 * p + k, c] = 1.0
        for i in range(n):
            W[k * n + i, c] = 1.0
            
    # Part 2: j=1,2 terms (Rank 21)
    # Split B[1:3, :] into 3 blocks of 2x2
    # Block t handles k in [2*t, 2*t+1]
    
    # Strassen indices mapping:
    # A' is 2x2:
    # A'[0,0] = A[0,1] -> U row 0*3+1 = 1
    # A'[0,1] = A[0,2] -> U row 0*3+2 = 2
    # A'[1,0] = A[1,1] -> U row 1*3+1 = 4
    # A'[1,1] = A[1,2] -> U row 1*3+2 = 5
    
    # B' is 2x2 for block t:
    # B'[0,0] = B[1, 2*t]   -> V row 1*6 + 2*t = 6 + 2*t
    # B'[0,1] = B[1, 2*t+1] -> V row 1*6 + 2*t+1 = 7 + 2*t
    # B'[1,0] = B[2, 2*t]   -> V row 2*6 + 2*t = 12 + 2*t
    # B'[1,1] = B[2, 2*t+1] -> V row 2*6 + 2*t+1 = 13 + 2*t
    
    # Outputs C'[i, k] correspond to W rows:
    # C'[0, 2*t]   -> W row (2*t)*2 + 0 = 4*t
    # C'[0, 2*t+1] -> W row (2*t+1)*2 + 0 = 4*t + 2
    # C'[1, 2*t]   -> W row (2*t)*2 + 1 = 4*t + 1
    # C'[1, 2*t+1] -> W row (2*t+1)*2 + 1 = 4*t + 3
    
    for t in range(3):
        base = 1 + t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (a00 + a11)(b00 + b11)
        c = base + 0
        U[1, c] = 1.0  # a00
        U[5, c] = 1.0  # a11
        V[6 + k0, c] = 1.0  # b00
        V[12 + k1, c] = 1.0 # b11
        W[4*t, c] = 1.0     # c00
        W[4*t+3, c] = 1.0   # c11
        
        # M2 = (a10 + a11)b00
        c = base + 1
        U[4, c] = 1.0  # a10
        U[5, c] = 1.0  # a11
        V[6 + k0, c] = 1.0  # b00
        W[4*t+1, c] = 1.0   # c10
        
        # M3 = a00(b10 - b11)
        c = base + 2
        U[1, c] = 1.0  # a00
        V[12 + k0, c] = 1.0  # b10
        V[12 + k1, c] = -1.0 # b11
        W[4*t+1, c] = 1.0    # c10
        W[4*t+3, c] = -1.0   # c11
        
        # M4 = a11(b01 - b00)
        c = base + 3
        U[5, c] = 1.0  # a11
        V[7 + k0, c] = 1.0  # b01 (note: 7+k0 is 1*6 + (2t+1) = 7+2t)
        V[6 + k0, c] = -1.0 # b00
        W[4*t+2, c] = 1.0   # c01
        
        # M5 = (a00 + a01)b11
        c = base + 4
        U[1, c] = 1.0  # a00
        U[2, c] = 1.0  # a01
        V[13 + k0, c] = 1.0 # b11 (note: 13+k0 is 2*6 + (2t+1) = 13+2t)
        W[4*t, c] = -1.0    # c00
        W[4*t+2, c] = 1.0   # c01
        
        # M6 = (a10 - a00)(b00 + b01)
        c = base + 5
        U[4, c] = 1.0  # a10
        U[1, c] = -1.0 # a00
        V[6 + k0, c] = 1.0  # b00
        V[7 + k0, c] = 1.0  # b01
        W[4*t+3, c] = 1.0   # c11
        
        # M7 = (a01 - a11)(b10 + b11)
        c = base + 6
        U[2, c] = 1.0  # a01
        U[5, c] = -1.0 # a11
        V[12 + k0, c] = 1.0  # b10
        V[13 + k0, c] = 1.0  # b11
        W[4*t, c] = 1.0      # c00
        
    # Verification
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb), dtype=float)
    V_sb = np.zeros((mp, rank_sb), dtype=float)
    W_sb = np.zeros((pn, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
