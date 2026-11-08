import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Uses the Kronecker product structure: 8 independent 2x2x2 Strassen blocks.
    Each 2x2x2 block has rank 7, giving total rank 8*7 = 56.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)  # (8, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 56)
    
    # Define Strassen 2x2x2 factors.
    # For each rank element r (0..6), we specify which (i,j) entries of A,
    # which (j,k) entries of B, and which (k,i) entries of C are involved.
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Standard Strassen decomposition:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B01
    # M2 = A00*(B10-B11)
    # M3 = A11*(B00-B10)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    #
    # C00 = M0 + M3 - M1 + M4
    # C01 = M2 + M6
    # C10 = M5 + M1
    # C11 = M0 - M2 + M3 + M5
    #
    # Let me verify this is correct by checking:
    # C00 = (A00+A11)(B00+B11) + A11(B00-B10) - (A10+A11)B01 + (A00+A01)B11
    #     = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B00 - A11*B10 - A10*B01 - A11*B01 + A00*B11 + A01*B11
    #     = A00*B00 + 2*A00*B11 + 2*A11*B00 + A11*B11 - A11*B10 - A10*B01 - A11*B01 + A01*B11
    # That doesn't look right. Let me use a verified Strassen decomposition.
    
    # Actually, let me use the well-known rank-7 decomposition. I'll use the one from
    # the current program which was validated to work.
    
    strassen_U = [
        [(0, 1), (3, 1)],   # r=0: A[0,0] + A[1,1]
        [(2, 1), (3, 1)],   # r=1: A[1,0] + A[1,1]
        [(0, 1)],           # r=2: A[0,0]
        [(3, 1)],           # r=3: A[1,1]
        [(0, 1), (1, 1)],   # r=4: A[0,0] + A[0,1]
        [(2, 1), (0, -1)],  # r=5: A[1,0] - A[0,0]
        [(1, 1), (3, -1)],  # r=6: A[0,1] - A[1,1]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],   # r=0: B[0,0] + B[1,1]
        [(0, 1)],           # r=1: B[0,0]
        [(1, 1), (3, -1)],  # r=2: B[0,1] - B[1,1]
        [(2, 1), (0, -1)],  # r=3: B[1,0] - B[0,0]
        [(3, 1)],           # r=4: B[1,1]
        [(0, 1), (1, 1)],   # r=5: B[0,0] + B[0,1]
        [(2, 1), (3, 1)],   # r=6: B[1,0] + B[1,1]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],   # r=0: C[0,0] + C[1,1]
        [(1, 1), (3, -1)],  # r=1: C[1,0] - C[1,1]  (W[k*2+i], so C[1,0]=W[2], C[1,1]=W[3])
        [(2, 1), (3, 1)],   # r=2: C[1,0] + C[1,1]
        [(0, 1), (1, 1)],   # r=3: C[0,0] + C[0,1]
        [(0, -1), (2, 1)],  # r=4: -C[0,0] + C[1,0]
        [(3, 1)],           # r=5: C[1,1]
        [(0, 1)],           # r=6: C[0,0]
    ]
    
    # Apply 8 blocks:
    # Block b (0..7):
    #   i_offset = 0 (always, since n=2)
    #   j_offset = 2 * (b // 4)  (0 or 2)
    #   k_offset = 2 * (b % 4)   (0, 2, 4, 6)
    #   base_rank = 7 * b
    
    for b in range(8):
        if time.time() > deadline:
            break
            
        i_off = 0
        j_off = 2 * (b // 4)
        k_off = 2 * (b % 4)
        base = 7 * b
        
        for r in range(7):
            cr = base + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                gi = local_i + i_off
                gj = local_j + j_off
                U[gi * 4 + gj, cr] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                gj = local_j + j_off
                gk = local_k + k_off
                V[gj * 8 + gk, cr] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                gk = local_k + k_off
                gi = local_i + i_off
                W[gk * 2 + gi, cr] += val
    
    # Return as lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
