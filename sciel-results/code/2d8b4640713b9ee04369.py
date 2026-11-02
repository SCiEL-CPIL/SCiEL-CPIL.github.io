"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen-based decomposition
    # Split 2x4x6 into two 2x2x6 blocks
    # Split each 2x2x6 into three 2x2x2 blocks
    # Total 6 blocks of 2x2x2
    # Rank of 2x2x2 is 7
    # Total rank = 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u: (i', j', coeff)
    # v: (j', k', coeff)
    # w: (i', k', coeff) -> W index is k*n + i
    
    strassen_u = [
        [(0,0,1), (1,1,1)],       # P1
        [(1,0,1), (1,1,1)],       # P2
        [(0,0,1)],                # P3
        [(1,1,1)],                # P4
        [(0,0,1), (0,1,1)],       # P5
        [(1,0,1), (0,0,-1)],      # P6
        [(0,1,1), (1,1,-1)]       # P7
    ]
    
    strassen_v = [
        [(0,0,1), (1,1,1)],       # P1
        [(0,0,1)],                # P2
        [(0,1,1), (1,1,-1)],      # P3
        [(1,0,1), (0,0,-1)],      # P4
        [(1,1,1)],                # P5
        [(0,0,1), (0,1,1)],       # P6
        [(1,0,1), (1,1,1)]        # P7
    ]
    
    strassen_w = [
        [(0,0,1), (1,1,1)],       # P1
        [(1,0,1)],                # P2
        [(0,1,1)],                # P3
        [(0,0,1), (1,0,1)],       # P4
        [(0,0,-1), (0,1,1)],      # P5
        [(1,1,1)],                # P6
        [(0,0,1), (1,1,-1)]       # P7
    ]
    
    # Blocks: (i_off, j_off, k_off)
    # A is 2x4. B is 4x6.
    # A_left (cols 0,1) * B_top (rows 0,1) -> C (rows 0,1)
    # A_left (cols 0,1) * B_bot (rows 2,3) -> C (rows 0,1)
    # Wait, B is 4x6. Rows 0,1,2,3. Cols 0..5.
    # A is 2x4. Rows 0,1. Cols 0..3.
    # C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    # Split j into {0,1} and {2,3}.
    # C = A[:,0:2] B[0:2,:] + A[:,2:4] B[2:4,:]
    
    # Term 1: A1 (2x2) * B1 (2x6)
    # B1 is rows 0,1 of B.
    # Split B1 into B1[:,0:2], B1[:,2:4], B1[:,4:6].
    # Each is 2x2.
    # So 3 blocks of 2x2x2.
    # i_off=0, j_off=0.
    # k_off = 0, 2, 4.
    
    # Term 2: A2 (2x2) * B2 (2x6)
    # A2 is cols 2,3 of A.
    # B2 is rows 2,3 of B.
    # Split B2 into B2[:,0:2], B2[:,2:4], B2[:,4:6].
    # Each is 2x2.
    # So 3 blocks of 2x2x2.
    # i_off=0, j_off=2.
    # k_off = 0, 2, 4.
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    col = 0
    for i_off, j_off, k_off in blocks:
        for s in range(7):
            # Fill U
            for i_prime, j_prime, coeff in strassen_u[s]:
                i = i_off + i_prime
                j = j_off + j_prime
                U[i * m + j][col] += coeff
            
            # Fill V
            for j_prime, k_prime, coeff in strassen_v[s]:
                j = j_off + j_prime
                k = k_off + k_prime
                V[j * p + k][col] += coeff
            
            # Fill W
            for i_prime, k_prime, coeff in strassen_w[s]:
                i = i_off + i_prime
                k = k_off + k_prime
                W[k * n + i][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
