"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Winograd-based decomposition for 2x3x9 matrix multiplication.
    # Split the inner dimension m=3 into blocks of 2+1, and the output dimension p=9 into 4*2+1.
    # For each 2x2x2 block, use Winograd's rank-7 identity.
    # Total rank = (number of 2x2 blocks) * 7 + (number of 2x1 blocks) * 2
    # 2x2 blocks: 4 blocks (for p=0..7, each 2x2)
    # 2x1 blocks: 1 block (for p=8)
    # Total rank = 4*7 + 1*2 = 30
    
    n, m, p = DIMENSIONS
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 27 x 30
    W = [[0] * rank for _ in range(p * n)]  # 18 x 30
    
    # Winograd 2x2x2 identity (rank 7)
    # U factors (2x2 A matrix):
    # u0: a00 + a11
    # u1: a10 + a11
    # u2: a00
    # u3: a00 + a01
    # u4: a01 - a11
    # u5: a01
    # u6: a10
    # V factors (2x2 B matrix):
    # v0: b00 + b11
    # v1: b11
    # v2: b01 - b11
    # v3: b00
    # v4: b00 + b01
    # v5: b10 + b11
    # v6: b10 - b11
    # W factors (2x2 C matrix, index k*n+i):
    # c00 (k=0,i=0): +P0, +P3, +P5, -P1, -P4
    # c10 (k=0,i=1): -P2, -P4, +P5, +P6
    # c01 (k=1,i=0): -P0, +P2, +P3, -P4
    # c11 (k=1,i=1): +P0, +P1, +P4, -P5, -P6
    
    def winograd_2x2x2(base_col, a_base, b_base, c_base):
        """Add a 2x2x2 Winograd block starting at column base_col.
        a_base: index in U for a00 (i=0,j=0)
        b_base: index in V for b00 (j=0,k=0)  
        c_base: index in W for c00 (k=0,i=0)
        """
        # U: a00=a_base, a01=a_base+1, a10=a_base+2, a11=a_base+3
        u0, u1, u2, u3, u4, u5, u6 = base_col, base_col+1, base_col+2, base_col+3, base_col+4, base_col+5, base_col+6
        
        # U coefficients
        U[a_base][u0] += 1      # a00
        U[a_base+1][u3] += 1    # a01
        U[a_base+1][u4] += 1    # a01
        U[a_base+1][u5] += 1    # a01
        U[a_base+2][u1] += 1    # a10
        U[a_base+2][u6] += 1    # a10
        U[a_base+3][u0] += 1    # a11
        U[a_base+3][u1] += 1    # a11
        U[a_base+3][u4] -= 1    # -a11
        U[a_base+3][u6] += 0    # no a11 in u6
        
        # V: b00=b_base, b01=b_base+1, b10=b_base+2, b11=b_base+3
        V[b_base][u0] += 1      # b00
        V[b_base][u3] += 1      # b00
        V[b_base][u4] += 1      # b00
        V[b_base+1][u2] += 1    # b01
        V[b_base+1][u4] += 1    # b01
        V[b_base+2][u5] += 1    # b10
        V[b_base+2][u6] += 1    # b10
        V[b_base+3][u0] += 1    # b11
        V[b_base+3][u1] += 1    # b11
        V[b_base+3][u2] -= 1    # -b11
        V[b_base+3][u5] += 1    # b11
        V[b_base+3][u6] -= 1    # -b11
        
        # W: c00=c_base, c10=c_base+1, c01=c_base+2, c11=c_base+3
        # c00 = P0 + P3 + P5 - P1 - P4
        W[c_base][u0] += 1
        W[c_base][u3] += 1
        W[c_base][u5] += 1
        W[c_base][u1] -= 1
        W[c_base][u4] -= 1
        
        # c10 = -P2 - P4 + P5 + P6
        W[c_base+1][u2] -= 1
        W[c_base+1][u4] -= 1
        W[c_base+1][u5] += 1
        W[c_base+1][u6] += 1
        
        # c01 = -P0 + P2 + P3 - P4
        W[c_base+2][u0] -= 1
        W[c_base+2][u2] += 1
        W[c_base+2][u3] += 1
        W[c_base+2][u4] -= 1
        
        # c11 = P0 + P1 + P4 - P5 - P6
        W[c_base+3][u0] += 1
        W[c_base+3][u1] += 1
        W[c_base+3][u4] += 1
        W[c_base+3][u5] -= 1
        W[c_base+3][u6] -= 1
    
    col = 0
    # Process 4 blocks of 2x2 (p=0,1), (p=2,3), (p=4,5), (p=6,7)
    for block in range(4):
        p_start = block * 2
        # a_base: for m=3, j=0..2, so a indices are i*3+j
        # We use a00, a01, a10, a11 for each 2x2 block in the m dimension
        # But we only have 2 rows (n=2) and 3 columns (m=3)
        # For each 2x2x2 block, we use a specific 2x2 submatrix of A
        # Let's use the first 2 columns (j=0,1) for all blocks
        a_base = 0  # a00 = A[0,0], a01 = A[0,1], a10 = A[1,0], a11 = A[1,1]
        b_base = p_start  # b00 = B[0,p_start], b01 = B[0,p_start+1], b10 = B[1,p_start], b11 = B[1,p_start+1]
        c_base = p_start * n  # c00 = C[p_start,0], c10 = C[p_start,1], c01 = C[p_start+1,0], c11 = C[p_start+1,1]
        
        winograd_2x2x2(col, a_base, b_base, c_base)
        col += 7
    
    # Now add the remaining terms for j=2 (third column of A)
    # For each k (0..8), we need A[i,2] * B[2,k]
    # This is a 2x1x9 block, which has rank 2*1*9 = 18 in schoolbook
    # But we can use a simpler approach: for each k, add a rank-2 block
    # Actually, let's just use schoolbook for the remaining part
    
    # Reset and do a cleaner approach
    # Let me restart with a correct implementation
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each 2x2 block in the p dimension, use Winograd
    for block in range(4):
        p_start = block * 2
        # Use A[:, 0:2] (first 2 columns) for the 2x2 part
        # a_base = 0 means A[0,0], A[0,1], A[1,0], A[1,1]
        a_base = 0
        # b_base: B[0:p_start+2, :] but we need B[j, k] for j=0,1 and k=p_start, p_start+1
        b_base = p_start  # B[0,p_start], B[0,p_start+1], B[1,p_start], B[1,p_start+1]
        # c_base: C[p_start:p_start+2, :] 
        c_base = p_start * n  # C[p_start,0], C[p_start,1], C[p_start+1,0], C[p_start+1,1]
        
        winograd_2x2x2(col, a_base, b_base, c_base)
        col += 7
    
    # Now add the contribution from A[:, 2] * B[2, :]
    # This is A[i,2] * B[2,k] for i=0,1 and k=0..8
    # Schoolbook: rank = 2*1*9 = 18, but we only have rank 30-28=2 left
    # This doesn't work. We need a different approach.
    
    # Let me use a simpler, correct approach: schoolbook with rank 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
