"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 3x4x8 into 8 blocks of 3x2x2
    # Each 3x2x2 block uses rank 8 (schoolbook for 2x2x2 extended to 3 rows)
    # Total rank = 8 * 8 = 64
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # J is the block index for the 4 columns of A / rows of B (0, 1)
    # K is the block index for the 8 columns of B (0, 1, 2, 3)
    for J in range(2):
        for K in range(4):
            # 8 schoolbook terms for the 2x2x2 core
            # v is local row/col index in J block (0, 1)
            # z is local col index in K block (0, 1)
            for i in range(n): # 3 rows
                for v in range(2): # 2 cols in J block
                    for z in range(2): # 2 cols in K block
                        # This is not quite right. The 3x2x2 rank 8 decomposition
                        # shares the B factors.
                        # Let's use the explicit 8 factors for 2x2x2 and replicate for i.
                        pass
                        
            # Correct approach:
            # For each (J, K), we have a 3x2x2 multiplication.
            # We use 8 factors.
            # Factor c (0..7) corresponds to (v, z) in schoolbook 2x2x2.
            # v is 0 or 1, z is 0 or 1.
            # c = v * 2 + z
            for c_local in range(8):
                v = c_local // 2
                z = c_local % 2
                
                # U: A[i, 2J + v]
                for i in range(n):
                    U[i * m + (2 * J + v)][col] = 1
                    
                # V: B[2J + v, 4K + z]
                V[(2 * J + v) * p + (4 * K + z)][col] = 1
                
                # W: C[i, 4K + z]
                for i in range(n):
                    W[(4 * K + z) * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
