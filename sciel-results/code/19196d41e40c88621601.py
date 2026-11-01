"""Matrix multiplication 2,2,14 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix.
    
    Output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # The tensor for matrix multiplication A(n x m) * B(m x p) -> C(n x p)
    # has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We seek a decomposition: T = sum_r U_r * V_r * W_r (outer products)
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    
    # Key insight: For 2x2x14, we can use a block-diagonal structure.
    # The B matrix is 2x14, so we can view it as 7 blocks of 2x2.
    # For each block, we can use Strassen's algorithm (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # Let's construct the Strassen decomposition for 2x2x2 first.
    # The correct Strassen decomposition uses 7 rank-1 terms.
    
    # For 2x2x2: A is 2x2, B is 2x2, C = A*B is 2x2
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # U[i,j,c] for A[i,j], V[j,k,c] for B[j,k], W[k,i,c] for C[k,i]
    # Condition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # Verified Strassen decomposition (7 terms):
    # I'll construct this by solving the system directly.
    
    # For 2x2x2, we need 7 vectors U_c, V_c, W_c (each in R^4)
    # such that for all (i,j,k) in {0,1}^3:
    #   sum_c U_c[i*2+j] * V_c[j*2+k] * W_c[k*2+i] = 1
    
    # This is a system of 8 equations. Let's find a solution.
    # The known Strassen decomposition has integer coefficients.
    
    # After careful derivation, here is a verified Strassen decomposition:
    # Using the standard formulation:
    
    # Strassen linear forms for A (2x2)
    # A indices: A00=0, A01=1, A10=2, A11=3
    # U index: i*m + j
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # s=0: M1 = (A00+A11)
    # s=1: M2 = (A10+A11)
    # s=2: M3 = A00
    # s=3: M4 = A11
    # s=4: M5 = (A00+A01)
    # s=5: M6 = (A10-A00)
    # s=6: M7 = (A01-A11)
    
    U_forms = [
        [1, 0, 0, 1],   # s=0: A00+A11
        [0, 0, 1, 1],   # s=1: A10+A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00+A01
        [-1, 0, 1, 0],  # s=5: A10-A00
        [0, 1, 0, -1]   # s=6: A01-A11
    ]
    
    # V forms for B block (2x2)
    # B indices in block: B00, B01, B10, B11
    # V index: j*p + k
    # For block r, columns are 2r, 2r+1
    # B00 -> j=0, k=2r -> index 2r
    # B01 -> j=0, k=2r+1 -> index 2r+1
    # B10 -> j=1, k=2r -> index p + 2r
    # B11 -> j=1, k=2r+1 -> index p + 2r + 1
    
    # s=0: M1 = (B00+B11)
    # s=1: M2 = B00
    # s=2: M3 = (B01-B11)
    # s=3: M4 = (B10-B00)
    # s=4: M5 = B11
    # s=5: M6 = (B00+B01)
    # s=6: M7 = (B10+B11)
    
    # W forms for C block (2x2)
    # C indices in block: C00, C01, C10, C11
    # W index: k*n + i
    # For block r, columns are 2r, 2r+1
    # C00 -> k=2r, i=0 -> index 2r*2 + 0 = 4r
    # C01 -> k=2r+1, i=0 -> index (2r+1)*2 + 0 = 4r+2
    # C10 -> k=2r, i=1 -> index 2r*2 + 1 = 4r+1
    # C11 -> k=2r+1, i=1 -> index (2r+1)*2 + 1 = 4r+3
    
    # Strassen Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(num_blocks):
        # Block r uses B columns 2r and 2r+1
        idx_e = 2 * r       # B00
        idx_f = 2 * r + 1   # B01
        idx_g = p + 2 * r   # B10
        idx_h = p + 2 * r + 1 # B11
        
        # W indices for C in block r
        w_idx_c00 = 4 * r
        w_idx_c10 = 4 * r + 1
        w_idx_c01 = 4 * r + 2
        w_idx_c11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[s][i]
                
            # Fill V
            # s=0: e+h
            # s=1: e
            # s=2: f-h
            # s=3: g-e
            # s=4: h
            # s=5: e+f
            # s=6: g+h
            if s == 0:
                V[idx_e][c] = 1
                V[idx_h][c] = 1
            elif s == 1:
                V[idx_e][c] = 1
            elif s == 2:
                V[idx_f][c] = 1
                V[idx_h][c] = -1
            elif s == 3:
                V[idx_e][c] = -1
                V[idx_g][c] = 1
            elif s == 4:
                V[idx_h][c] = 1
            elif s == 5:
                V[idx_e][c] = 1
                V[idx_f][c] = 1
            elif s == 6:
                V[idx_g][c] = 1
                V[idx_h][c] = 1
                
            # Fill W
            # s=0 (M1): C00+1, C11+1
            # s=1 (M2): C10+1, C11-1
            # s=2 (M3): C01+1, C11+1
            # s=3 (M4): C00+1, C10+1
            # s=4 (M5): C00-1, C01+1
            # s=5 (M6): C11+1
            # s=6 (M7): C00+1
            
            if s == 0:
                W[w_idx_c00][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 1:
                W[w_idx_c10][c] = 1
                W[w_idx_c11][c] = -1
            elif s == 2:
                W[w_idx_c01][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 3:
                W[w_idx_c00][c] = 1
                W[w_idx_c10][c] = 1
            elif s == 4:
                W[w_idx_c00][c] = -1
                W[w_idx_c01][c] = 1
            elif s == 5:
                W[w_idx_c11][c] = 1
            elif s == 6:
                W[w_idx_c00][c] = 1
                
    # Verify the decomposition
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if verified:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook (rank 56)
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n*m)]
        V_sb = [[0] * rank_sb for _ in range(m*p)]
        W_sb = [[0] * rank_sb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
