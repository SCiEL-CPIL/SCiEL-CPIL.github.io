"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_decomposition():
    """
    Build the full 4x4 x 4x6 -> 4x6 matrix multiplication decomposition.
    
    Strategy: 2x2 block structure with Kronecker product.
    A is 4x4: 2x2 blocks of size 2x2.
    B is 4x6: 2x3 blocks of size 2x2.
    C is 4x6: 2x3 blocks of size 2x2.
    
    Outer structure: 2x2 * 2x3 block multiplication.
    - For columns 0,1: Strassen (7 terms) for each 2x2*2x2 block product
    - For column 2: schoolbook (4 terms) for each 2x2*2x2 block product
    
    Total rank: (7 + 4) * 7 = 77
    """
    n, m, p = 4, 4, 6
    sU, sV, sW = _strassen_2x2()
    
    total_rank = 11 * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Process each outer term
    for ot_idx in range(11):
        for t in range(7):
            col = ot_idx * 7 + t
            
            if ot_idx < 7:
                # Strassen terms for columns 0,1
                # U: A[ai][bj] for all ai,bj in {0,1}
                for ai in range(2):
                    for bj in range(2):
                        for ai2 in range(2):
                            for bj2 in range(2):
                                coeff = sU[ai2*2 + bj2, t]
                                if coeff != 0:
                                    u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                                    U[u_row, col] += coeff
                
                # V: B[bj][ck] for bj in {0,1}, ck in {0,1}
                for bj in range(2):
                    for ck in range(2):
                        for bj2 in range(2):
                            for ck2 in range(2):
                                coeff = sV[bj2*2 + ck2, t]
                                if coeff != 0:
                                    v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                                    V[v_row, col] += coeff
                
                # W: C[ai][ck] for ai in {0,1}, ck in {0,1}
                for ai in range(2):
                    for ck in range(2):
                        for ai2 in range(2):
                            for ck2 in range(2):
                                coeff = sW[ck2*2 + ai2, t]
                                if coeff != 0:
                                    w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                                    W[w_row, col] += coeff
            else:
                # Schoolbook terms for column 2
                sb_idx = ot_idx - 7  # 0, 1, 2, 3
                ai = sb_idx // 2
                bj = sb_idx % 2
                
                # U: A[ai][bj]
                for ai2 in range(2):
                    for bj2 in range(2):
                        u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                        U[u_row, col] += 1
                
                # V: B[bj][2]
                for bj2 in range(2):
                    for ck2 in range(2):
                        v_row = (2*bj + bj2) * 6 + (2*2 + ck2)
                        V[v_row, col] += 1
                
                # W: C[ai][2]
                for ai2 in range(2):
                    for ck2 in range(2):
                        w_row = (2*2 + ck2) * 4 + (2*ai + ai2)
                        W[w_row, col] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    return _build_decomposition()

# EVOLVE-BLOCK-END
