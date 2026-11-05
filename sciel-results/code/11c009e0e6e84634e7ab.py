import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 43
    R = 43
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset, rank_start):
        # i_offset: 0 or 1 (since n=2, only 0)
        # j_offset: start index of j in 0..4
        # k_offset: start index of k in 0..5
        # rank_start: index in R
        
        # Local indices: li in 0..1, lj in 0..1, lk in 0..1
        # Global: i = i_offset + li, j = j_offset + lj, k = k_offset + lk
        
        # Strassen coefficients
        # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1)
        # M2: A(1,0)+A(1,1) * B(0,0)
        # M3: A(0,0) * B(0,1)-B(1,1)
        # M4: A(1,1) * B(1,0)-B(0,0)
        # M5: A(0,0)+A(0,1) * B(1,1)
        # M6: A(1,0)-A(0,0) * B(0,0)+B(0,1)
        # M7: A(0,1)-A(1,1) * B(1,0)+B(1,1)
        
        # U[li, lj, r]
        # V[lj, lk, r]
        # W[lk, li, r]
        
        # r=0 (M1)
        U[0, 0, rank_start+0] = 1; U[1, 1, rank_start+0] = 1
        V[0, 0, rank_start+0] = 1; V[1, 1, rank_start+0] = 1
        W[0, 0, rank_start+0] = 1; W[0, 1, rank_start+0] = 1 # C00, C10
        
        # r=1 (M2)
        U[1, 0, rank_start+1] = 1; U[1, 1, rank_start+1] = 1
        V[0, 0, rank_start+1] = 1
        W[0, 1, rank_start+1] = 1 # C10
        
        # r=2 (M3)
        U[0, 0, rank_start+2] = 1
        V[0, 1, rank_start+2] = 1; V[1, 1, rank_start+2] = -1
        W[1, 0, rank_start+2] = 1 # C01
        
        # r=3 (M4)
        U[1, 1, rank_start+3] = 1
        V[1, 0, rank_start+3] = 1; V[0, 0, rank_start+3] = -1
        W[0, 0, rank_start+3] = 1; W[1, 0, rank_start+3] = 1; W[1, 1, rank_start+3] = 1 # C00, C01, C11
        
        # r=4 (M5)
        U[0, 0, rank_start+4] = 1; U[0, 1, rank_start+4] = 1
        V[1, 1, rank_start+4] = 1
        W[0, 0, rank_start+4] = -1 # C00
        
        # r=5 (M6)
        U[1, 0, rank_start+5] = 1; U[0, 0, rank_start+5] = -1
        V[0, 0, rank_start+5] = 1; V[0, 1, rank_start+5] = 1
        W[1, 1, rank_start+5] = 1 # C11
        
        # r=6 (M7)
        U[0, 1, rank_start+6] = 1; U[1, 1, rank_start+6] = -1
        V[1, 0, rank_start+6] = 1; V[1, 1, rank_start+6] = 1
        W[0, 0, rank_start+6] = -1; W[1, 1, rank_start+6] = -1 # C00, C11

        # Map to global
        for li in range(2):
            for lj in range(2):
                for r in range(7):
                    val = U[li, lj, rank_start+r]
                    if val != 0:
                        gi = i_offset + li
                        gj = j_offset + lj
                        U[gi * m + gj, rank_start+r] += val
        
        for lj in range(2):
            for lk in range(2):
                for r in range(7):
                    val = V[lj, lk, rank_start+r]
                    if val != 0:
                        gj = j_offset + lj
                        gk = k_offset + lk
                        V[gj * p + gk, rank_start+r] += val
        
        for lk in range(2):
            for li in range(2):
                for r in range(7):
                    val = W[lk, li, rank_start+r]
                    if val != 0:
                        gk = k_offset + lk
                        gi = i_offset + li
                        W[gk * n + gi, rank_start+r] += val

    # Block 1: A_L (2x2) x B_T (2x6)
    # i_offset=0, j_offset=0
    # k splits: 0, 2, 4
    rank_ptr = 0
    for k_off in [0, 2, 4]:
        add_strassen_block(0, 0, k_off, rank_ptr)
        rank_ptr += 7
        
    # Block 2: A_R1 (2x1) x B_B2 (1x6)
    # i_offset=0, j=2
    # k is 0..5
    # Rank 1
    U[0 * m + 2, rank_ptr] = 1
    U[1 * m + 2, rank_ptr] = 1
    for k in range(6):
        V[2 * p + k, rank_ptr] = 1
        W[k * n + 0, rank_ptr] = 1
        W[k * n + 1, rank_ptr] = 1
    rank_ptr += 1
    
    # Block 3: A_R2 (2x2) x B_B1 (2x6)
    # i_offset=0, j_offset=3
    # k splits: 0, 2, 4
    for k_off in [0, 2, 4]:
        add_strassen_block(0, 3, k_off, rank_ptr)
        rank_ptr += 7

    return {"U": U, "V": V, "W": W}
