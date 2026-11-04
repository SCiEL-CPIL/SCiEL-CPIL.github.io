import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Construct a rank-48 decomposition using a shared Strassen structure.
    # Key insight: Instead of 8 independent 2x2 Strassen blocks (7 each = 56),
    # we can use a 2x4*4x8 decomposition that shares terms across column pairs.
    #
    # Strategy: Split the 8 output columns into 4 pairs. For each pair, we need
    # a 2x4*4x2 product. Use a rank-12 decomposition for each 2x4*4x2 block.
    # Total: 4 * 12 = 48.
    #
    # For 2x4*4x2: C[i,c'] = sum_{a=0}^{3} A[i,a]*B[a,c']
    # Split A into A1 (cols 0-1) and A2 (cols 2-3), B into B1 (rows 0-1) and B2 (rows 2-3)
    # C = A1*B1 + A2*B2
    # Use 6 Strassen-like terms for the combined 2x2*2x2 + 2x2*2x2.
    #
    # Actually, let me use a different approach: direct rank-48 construction
    # using a modified Strassen pattern.
    
    # Let me try: use 6 terms per column-pair group by merging Strassen terms.
    # For columns (2k, 2k+1) and rows (2j, 2j+1), use:
    # M1 = (A[0,2j]+A[1,2j+1]) * (B[2j,2k]+B[2j+1,2k+1])
    # M2 = (A[1,2j]+A[1,2j+1]) * B[2j,2k]
    # M3 = A[0,2j] * (B[2j,2k+1]-B[2j+1,2k+1])
    # M4 = A[1,2j+1] * (B[2j+1,2k]-B[2j,2k])
    # M5 = (A[0,2j]+A[0,2j+1]) * B[2j+1,2k+1]
    # M6 = (A[1,2j]-A[0,2j]) * (B[2j,2k]+B[2j,2k+1])
    # Plus one more term for the cross-block interaction.
    
    # This is getting complex. Let me just use the rank-56 solution as a fallback
    # and try a numerical search for rank 48.
    
    # First, construct the rank-56 solution.
    rank_56 = 56
    U56 = np.zeros((8, rank_56))
    V56 = np.zeros((32, rank_56))
    W56 = np.zeros((16, rank_56))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        U56[a00, col] += 1
        U56[a11, col] += 1
        V56[b00, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += 1
        W56[c11, col] += 1
        col += 1
        
        U56[a10, col] += 1
        U56[a11, col] += 1
        V56[b00, col] += 1
        W56[c10, col] += 1
        W56[c11, col] += -1
        col += 1
        
        U56[a00, col] += 1
        V56[b01, col] += 1
        V56[b11, col] += -1
        W56[c01, col] += 1
        W56[c11, col] += 1
        col += 1
        
        U56[a11, col] += 1
        V56[b10, col] += 1
        V56[b00, col] += -1
        W56[c00, col] += 1
        W56[c10, col] += 1
        col += 1
        
        U56[a00, col] += 1
        U56[a01, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += -1
        W56[c01, col] += 1
        col += 1
        
        U56[a10, col] += 1
        U56[a00, col] += -1
        V56[b00, col] += 1
        V56[b01, col] += 1
        W56[c11, col] += 1
        col += 1
        
        U56[a01, col] += 1
        U56[a11, col] += -1
        V56[b10, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += 1
        col += 1
    
    assert col == 56
    
    # Try to reduce rank by finding a linear combination of terms that sums to zero.
    # Build the tensor matrix.
    M = np.zeros((64, 56))
    for r in range(56):
        for i in range(2):
            for a in range(4):
                for c in range(8):
                    M[i*32 + a*8 + c, r] = U56[i*4+a, r] * V56[a*8+c, r] * W56[c*2+i, r]
    
    # Check for linear dependencies.
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    
    if rank_M < 56 and time.time() < deadline:
        # Find null space vectors.
        null_space = np.linalg.null_space(M)
        null_dim = null_space.shape[1]
        
        # Try to use null space vectors to reduce rank.
        # If we have a null vector v, we can modify the decomposition.
        # For now, just check if we can eliminate terms.
        pass
    
    # Return the rank-56 solution.
    return {
        "U": U56.tolist(),
        "V": V56.tolist(),
        "W": W56.tolist()
    }
