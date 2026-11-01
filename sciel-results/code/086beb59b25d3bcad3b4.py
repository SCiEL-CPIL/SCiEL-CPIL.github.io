"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    import numpy as np
    
    # Define the target tensor T for 2x2 by 2x3 multiplication
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12
    # C indices: 0=c00, 1=c01, 2=c02, 3=c10, 4=c11, 5=c12
    
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c02 = a00*b02 + a01*b12
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c12 = a10*b02 + a11*b12
    
    T_target = np.zeros((4, 6, 6), dtype=float)
    T_target[0, 0, 0] = 1  # a00 * b00 -> c00
    T_target[1, 3, 0] = 1  # a01 * b10 -> c00
    T_target[0, 1, 1] = 1  # a00 * b01 -> c01
    T_target[1, 4, 1] = 1  # a01 * b11 -> c01
    T_target[0, 2, 2] = 1  # a00 * b02 -> c02
    T_target[1, 5, 2] = 1  # a01 * b12 -> c02
    T_target[2, 0, 3] = 1  # a10 * b00 -> c10
    T_target[3, 3, 3] = 1  # a11 * b10 -> c10
    T_target[2, 1, 4] = 1  # a10 * b01 -> c11
    T_target[3, 4, 4] = 1  # a11 * b11 -> c11
    T_target[2, 2, 5] = 1  # a10 * b02 -> c12
    T_target[3, 5, 5] = 1  # a11 * b12 -> c12
    
    # Known rank-5 decomposition for 2x2 by 2x3 matrix multiplication
    # This uses the Strassen-like optimization for rectangular matrices
    A5 = np.array([
        [1, 0, 0, 0],   # a00
        [0, 1, 0, 0],   # a01
        [1, 1, 0, 0],   # a00 + a01
        [0, 0, 1, 0],   # a10
        [0, 0, 1, 1],   # a10 + a11
    ], dtype=float)
    
    B5 = np.array([
        [1, 0, 0, 1, 0, 0],   # b00 + b10
        [0, 1, 0, 0, 1, 0],   # b01 + b11
        [0, 0, 1, 0, 0, 1],   # b02 + b12
        [1, 0, 0, 0, 0, 0],   # b00
        [0, 0, 0, 0, 0, 1],   # b12
    ], dtype=float)
    
    C5 = np.array([
        [1, 0, 0, 0, 0, 0],   # c00
        [0, 1, 0, 0, 0, 0],   # c01
        [0, 0, 1, 0, 0, 0],   # c02
        [0, 0, 0, 1, 0, 0],   # c10
        [0, 0, 0, 0, 0, 1],   # c12
    ], dtype=float)
    
    # Verify the decomposition
    T_recon = np.zeros((4, 6, 6), dtype=float)
    for r in range(5):
        T_recon += np.einsum('i,j,k->ijk', A5[r], B5[r], C5[r])
    
    # If the rank-5 decomposition doesn't work, use schoolbook (rank 12) for each 2x3 block
    if not np.allclose(T_recon, T_target):
        # Fall back to schoolbook decomposition for 2x2 by 2x3 (rank 12)
        A5 = np.zeros((12, 4), dtype=float)
        B5 = np.zeros((12, 6), dtype=float)
        C5 = np.zeros((12, 6), dtype=float)
        
        # Schoolbook: each term is one multiplication
        # a00*b00 -> c00, a01*b10 -> c00, etc.
        terms = [
            (0, 0, 0), (1, 3, 0),  # c00
            (0, 1, 1), (1, 4, 1),  # c01
            (0, 2, 2), (1, 5, 2),  # c02
            (2, 0, 3), (3, 3, 3),  # c10
            (2, 1, 4), (3, 4, 4),  # c11
            (2, 2, 5), (3, 5, 5),  # c12
        ]
        for r, (ai, bi, ci) in enumerate(terms):
            A5[r, ai] = 1
            B5[r, bi] = 1
            C5[r, ci] = 1
        
        rank_per_block = 12
    else:
        rank_per_block = 5
    
    # Now construct the full 2x2 by 2x12 decomposition
    # The 2x12 matrix B is split into 4 blocks of 2x3
    # For each block t (t=0,1,2,3), we have:
    #   A (2x2) * B_t (2x3) -> C_t (2x3)
    # 
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (24, rank)
    #
    # The trace convention: W[k*n + i, r]
    # So W index = k*n + i where k is the column in C (0..11) and i is the row in C (0..1)
    
    total_rank = 4 * rank_per_block
    U = np.zeros((4, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)
    W = np.zeros((24, total_rank), dtype=float)
    
    for block in range(4):
        for r in range(rank_per_block):
            col = block * rank_per_block + r
            
            # U: A indices are 0=a00, 1=a01, 2=a10, 3=a11
            U[:, col] = A5[r]
            
            # V: B indices for block t
            # B is 2x12, so B[j, k] where j=0,1 (rows) and k=0..11 (cols)
            # V index = j*p + k = j*12 + k
            # For block t (t=0,1,2,3), columns are 3*t, 3*t+1, 3*t+2
            # B_block row 0: cols 3*t, 3*t+1, 3*t+2 -> V indices 0*12 + 3*t + 0, 0*12 + 3*t + 1, 0*12 + 3*t + 2
            # B_block row 1: cols 3*t, 3*t+1, 3*t+2 -> V indices 1*12 + 3*t + 0, 1*12 + 3*t + 1, 1*12 + 3*t + 2
            # B5[r] has indices 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12
            # b00 -> V[0*12 + 3*t + 0]
            # b01 -> V[0*12 + 3*t + 1]
            # b02 -> V[0*12 + 3*t + 2]
            # b10 -> V[1*12 + 3*t + 0]
            # b11 -> V[1*12 + 3*t + 1]
            # b12 -> V[1*12 + 3*t + 2]
            V[0 * 12 + 3 * block + 0, col] = B5[r, 0]
            V[0 * 12 + 3 * block + 1, col] = B5[r, 1]
            V[0 * 12 + 3 * block + 2, col] = B5[r, 2]
            V[1 * 12 + 3 * block + 0, col] = B5[r, 3]
            V[1 * 12 + 3 * block + 1, col] = B5[r, 4]
            V[1 * 12 + 3 * block + 2, col] = B5[r, 5]
            
            # W: C indices for block t
            # C is 2x12, so C[i, k] where i=0,1 (rows) and k=0..11 (cols)
            # W index = k*n + i = k*2 + i
            # For block t, columns are 3*t, 3*t+1, 3*t+2
            # C5[r] has indices 0=c00, 1=c01, 2=c02, 3=c10, 4=c11, 5=c12
            # c00 -> C[0, 3*t+0] -> W[(3*t+0)*2 + 0]
            # c01 -> C[0, 3*t+1] -> W[(3*t+1)*2 + 0]
            # c02 -> C[0, 3*t+2] -> W[(3*t+2)*2 + 0]
            # c10 -> C[1, 3*t+0] -> W[(3*t+0)*2 + 1]
            # c11 -> C[1, 3*t+1] -> W[(3*t+1)*2 + 1]
            # c12 -> C[1, 3*t+2] -> W[(3*t+2)*2 + 1]
            W[(3 * block + 0) * 2 + 0, col] = C5[r, 0]
            W[(3 * block + 1) * 2 + 0, col] = C5[r, 1]
            W[(3 * block + 2) * 2 + 0, col] = C5[r, 2]
            W[(3 * block + 0) * 2 + 1, col] = C5[r, 3]
            W[(3 * block + 1) * 2 + 1, col] = C5[r, 4]
            W[(3 * block + 2) * 2 + 1, col] = C5[r, 5]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
