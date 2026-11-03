import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Rank 81: 49 (Strassen 4x4x4) + 32 (Schoolbook 4x4x2)
    R = 81
    U = np.zeros((16, R))  # A is 4x4, flat index i*4+j
    V = np.zeros((24, R))  # B is 4x6, flat index j*6+k
    W = np.zeros((24, R))  # C is 4x6, W[k*4+i, r] where k=col_C, i=row_C
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    
    # Outer Strassen coefficients for 2x2 blocks
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3 in 2x2 grid)
    # B blocks: B00, B01, B10, B11
    # 7 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # alpha_outer[s][block_idx]
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    alpha_outer = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # beta_outer[s][block_idx]
    # block_idx: 0=B00, 1=B01, 2=B10, 3=B11
    beta_outer = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Reconstruction of C blocks from M1..M7
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # recon_outer[c_block_idx][s]
    # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
    recon_outer = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 0, 1]   # C11
    ]
    
    # Inner Strassen coefficients for 2x2 matrices
    # Same structure as outer
    alpha_inner = alpha_outer
    beta_inner = beta_outer
    
    # Reconstruction of Z (2x2 matrix) from N1..N7
    # Z00 = N1 + N4 - N5 - N7
    # Z01 = N3 + N5
    # Z10 = N2 + N4
    # Z11 = N1 - N2 + N3 + N7
    recon_inner = [
        [1, 0, 0, 1, -1, 0, -1], # Z00
        [0, 0, 1, 0, 1, 0, 0],   # Z01
        [0, 1, 0, 1, 0, 0, 0],   # Z10
        [1, -1, 1, 0, 0, 0, 1]   # Z11
    ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Calculate U coefficients (A)
            # A index i,j. i = 2*br + dr, j = 2*bc + dc
            # Coeff = sum over dr,dc,br,bc where i=2br+dr, j=2bc+dc
            # of alpha_inner[t][dr*2+dc] * alpha_outer[s][br*2+bc]
            
            # Note: alpha_inner[t] is indexed by local 2x2 position (dr, dc) -> index dr*2+dc?
            # In my alpha_outer definition, index is block_idx (0..3).
            # For inner, the "blocks" are scalars.
            # So alpha_inner[t][local_idx] where local_idx = dr*2+dc.
            
            for i in range(4):
                for j in range(4):
                    # Determine br, bc, dr, dc
                    br = i // 2
                    dr = i % 2
                    bc = j // 2
                    dc = j % 2
                    
                    local_idx_A = dr * 2 + dc
                    block_idx_A = br * 2 + bc
                    
                    coeff = alpha_inner[t][local_idx_A] * alpha_outer[s][block_idx_A]
                    if coeff != 0:
                        U[i * 4 + j, col] += coeff
            
            # Calculate V coefficients (B)
            # B index j,k. j = 2*br + er, k = 2*bc + ec
            # Coeff = sum over er,ec,br,bc where j=2br+er, k=2bc+ec
            # of beta_inner[t][er*2+ec] * beta_outer[s][br*2+bc]
            
            for j in range(4):
                for k in range(4): # Only first 4 cols of B
                    br = j // 2
                    er = j % 2
                    bc = k // 2
                    ec = k % 2
                    
                    local_idx_B = er * 2 + ec
                    block_idx_B = br * 2 + bc
                    
                    coeff = beta_inner[t][local_idx_B] * beta_outer[s][block_idx_B]
                    if coeff != 0:
                        V[j * 6 + k, col] += coeff
            
            # Calculate W coefficients (C)
            # C index i,k. i = 2*br + dr, k = 2*bc + dc
            # Z_s[dr, dc] contributes to C blocks.
            # Z_s[dr, dc] = sum_t recon_inner[dr*2+dc][t] * N_t
            # C_block[br, bc][dr, dc] = sum_s recon_outer[br*2+bc][s] * Z_s[dr, dc]
            
            # So for a specific C element i,k:
            # br = i//2, dr = i%2
            # bc = k//2, dc = k%2
            # Coeff = recon_outer[br*2+bc][s] * recon_inner[dr*2+dc][t]
            
            for i in range(4):
                for k in range(4): # Only first 4 cols of C
                    br = i // 2
                    dr = i % 2
                    bc = k // 2
                    dc = k % 2
                    
                    c_block_idx = br * 2 + bc
                    z_local_idx = dr * 2 + dc
                    
                    coeff = recon_outer[c_block_idx][s] * recon_inner[z_local_idx][t]
                    if coeff != 0:
                        W[k * 4 + i, col] += coeff
            
            col += 1
    
    # --- Part 2: 4x4x2 Schoolbook (Rank 32) ---
    # C[i, 4+k] = sum_j A[i, j] B[j, 4+k]
    for i in range(4):
        for k in range(2):
            for j in range(4):
                u_idx = i * 4 + j
                v_idx = j * 6 + (4 + k)
                w_idx = (4 + k) * 4 + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                
                col += 1
                
    # Ensure col == R
    # assert col == R, f"Col {col} != R {R}"
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
