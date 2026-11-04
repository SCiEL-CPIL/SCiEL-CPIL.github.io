"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback: rank = n*m*p = 120
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Winograd FMA: 4x4x4 core (rank 16) + 4x1x6 strip (rank 24) = rank 40
    # Partition: n=4, m=5=4+1, p=6
    # Core: 4x4x6 -> use 4x4x4 blocks for first 4 cols of p, then 4x4x2 for last 2 cols
    # Actually, let's think more carefully.
    # m=5, so we split into m=4 core and m=1 strip
    # Core: 4x4x6 matrix multiply
    # Strip: 4x1x6 matrix multiply (outer product)
    
    # For the 4x4x6 core, we can use Winograd FMA on 4x4x4 sub-blocks
    # 4x4x4 Winograd FMA has rank 16 (4-point FFT)
    # We need 4x4x6, which is 4x4x4 + 4x4x2
    # 4x4x4: rank 16
    # 4x4x2: rank 8 (schoolbook)
    # Total core rank: 24
    # Strip 4x1x6: rank 24 (schoolbook)
    # Total: 48
    
    # Let me try a different approach: use the 4x4x4 Winograd for the 4x4x4 part
    # and handle the rest with schoolbook
    
    # Winograd FMA 4x4x4: uses 4-point DFT
    # The 4-point DFT matrix (with appropriate scaling)
    # For 4-point FFT: x[0], x[1], x[2], x[3]
    # X[0] = x0+x1+x2+x3
    # X[1] = x0-x1-x2+x3
    # X[2] = x0-x1+x2-x3
    # X[3] = x0-x1+x2-x3 ... wait, let me use the standard form
    
    # Actually, for Winograd FMA, we use:
    # Input transform: F (4x4)
    # Pointwise multiply
    # Output transform: G (4x4)
    # F * diag * G = identity * (cyclic convolution kernel)
    
    # For 4-point cyclic convolution, the rank is 4 (not 16!)
    # Because 4-point FFT has 4 frequency bins
    # So 4x4x4 cyclic convolution has rank 4
    # But matrix multiplication is NOT cyclic convolution!
    
    # I was wrong. Let me reconsider.
    # Matrix multiplication C = A * B where A is 4x4, B is 4x4
    # C[i][j] = sum_k A[i][k] * B[k][j]
    # This is NOT a cyclic convolution. It's a standard matrix multiply.
    
    # Winograd FMA works for convolutions, not directly for matrix multiplication.
    # However, there are tensor decomposition methods for matrix multiplication.
    
    # Let me fall back to a better block decomposition strategy.
    # Use Strassen for 2x2x2 blocks.
    
    def strassen2_decomposition():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        return U, V, W

    # Try multiple block decompositions and pick the best valid one
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def try_decomposition(n_bounds, m_bounds, p_bounds, use_strassen2=True):
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline:
            return
        
        # Calculate total rank
        total_rank = 0
        for bi in range(len(n_bounds)-1):
            for bj in range(len(m_bounds)-1):
                for bk in range(len(p_bounds)-1):
                    ns = n_bounds[bi+1] - n_bounds[bi]
                    ms = m_bounds[bj+1] - m_bounds[bj]
                    ps = p_bounds[bk+1] - p_bounds[bk]
                    if use_strassen2 and ns == 2 and ms == 2 and ps == 2:
                        total_rank += 7
                    else:
                        total_rank += ns * ms * ps
        
        if total_rank >= best_rank:
            return
        
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col_offset = 0
        U2, V2, W2 = strassen2_decomposition()
        
        for bi in range(len(n_bounds)-1):
            for bj in range(len(m_bounds)-1):
                for bk in range(len(p_bounds)-1):
                    ni = n_bounds[bi]
                    nj = n_bounds[bi+1]
                    mi = m_bounds[bj]
                    mj = m_bounds[bj+1]
                    pi = p_bounds[bk]
                    pj = p_bounds[bk+1]
                    
                    n_size = nj - ni
                    m_size = mj - mi
                    p_size = pj - pi
                    
                    if use_strassen2 and n_size == 2 and m_size == 2 and p_size == 2:
                        for r in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U_full[(ni+ii)*m + (mi+jj)][col_offset + r] = U2[ii*2+jj][r]
                            for jj in range(2):
                                for kk in range(2):
                                    V_full[(mi+jj)*p + (pi+kk)][col_offset + r] = V2[jj*2+kk][r]
                            for kk in range(2):
                                for ii in range(2):
                                    W_full[(pi+kk)*n + (ni+ii)][col_offset + r] = W2[kk*2+ii][r]
                        col_offset += 7
                    else:
                        for ii in range(n_size):
                            for jj in range(m_size):
                                for kk in range(p_size):
                                    U_full[(ni+ii)*m + (mi+jj)][col_offset] = 1
                                    V_full[(mi+jj)*p + (pi+kk)][col_offset] = 1
                                    W_full[(pi+kk)*n + (ni+ii)][col_offset] = 1
                                    col_offset += 1
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(total_rank):
                        s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
            return
    
    # Try various block decompositions
    # n=4, m=5, p=6
    
    # Option 1: 2+2, 2+2+1, 2+2+2
    try_decomposition([0,2,4], [0,2,4,5], [0,2,4,6])
    
    # Option 2: 2+2, 1+2+2, 2+2+2  
    try_decomposition([0,2,4], [0,1,3,5], [0,2,4,6])
    
    # Option 3: 2+2, 2+1+2, 2+2+2
    try_decomposition([0,2,4], [0,2,3,5], [0,2,4,6])
    
    # Option 4: 2+2, 1+1+3, 2+2+2
    try_decomposition([0,2,4], [0,1,2,5], [0,2,4,6])
    
    # Option 5: 1+3, 2+2+1, 2+2+2
    try_decomposition([0,1,4], [0,2,4,5], [0,2,4,6])
    
    # Option 6: 1+3, 1+2+2, 2+2+2
    try_decomposition([0,1,4], [0,1,3,5], [0,2,4,6])
    
    # Option 7: 2+2, 2+2+1, 1+2+3
    try_decomposition([0,2,4], [0,2,4,5], [0,1,3,6])
    
    # Option 8: 2+2, 2+2+1, 3+3
    try_decomposition([0,2,4], [0,2,4,5], [0,3,6])
    
    # Option 9: 2+2, 1+1+1+2, 2+2+2
    try_decomposition([0,2,4], [0,1,2,3,5], [0,2,4,6])
    
    # Option 10: 2+2, 2+1+1+1, 2+2+2
    try_decomposition([0,2,4], [0,2,3,4,5], [0,2,4,6])
    
    if best_rank < schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return best_result
# EVOLVE-BLOCK-END
