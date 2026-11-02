"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, c):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1

    # Helper to add a Strassen term for a 2x2 block
    # i_offset: row offset for A and C
    # j_offset: col offset for A and row offset for B
    # k_offset: col offset for B and C
    def add_strassen_block(i_off, j_off, k_off, start_col):
        # 7 terms
        # u, v, w vectors for the 2x2 block
        # A indices: (0,0), (0,1), (1,0), (1,1) -> global (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B indices: (0,0), (0,1), (1,0), (1,1) -> global (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C indices: (0,0), (0,1), (1,0), (1,1) -> global (k_off, i_off), (k_off+1, i_off), (k_off, i_off+1), (k_off+1, i_off+1)
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # P1
            ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
            # P2
            ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
            # P3
            ((1, 0, 0, 0), (0, 1, 0, 1), (0, 1, 0, -1)),
            # P4
            ((0, 0, 0, 1), (0, 0, 1, -1), (-1, 0, 0, 0)),
            # P5
            ((1, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1)),
            # P6
            ((0, 0, 1, -1), (1, 1, 0, 0), (1, 0, 0, 0)),
            # P7
            ((0, 1, 0, -1), (0, 0, 1, 1), (0, 1, 0, 0)),
        ]
        
        for t in range(7):
            c = start_col + t
            u, v, w = terms[t]
            
            # Map u to U
            # u[0] -> A[i_off, j_off]
            # u[1] -> A[i_off, j_off+1]
            # u[2] -> A[i_off+1, j_off]
            # u[3] -> A[i_off+1, j_off+1]
            if u[0] != 0: U[(i_off) * m + (j_off)][c] = u[0]
            if u[1] != 0: U[(i_off) * m + (j_off + 1)][c] = u[1]
            if u[2] != 0: U[(i_off + 1) * m + (j_off)][c] = u[2]
            if u[3] != 0: U[(i_off + 1) * m + (j_off + 1)][c] = u[3]
            
            # Map v to V
            # v[0] -> B[j_off, k_off]
            # v[1] -> B[j_off, k_off+1]
            # v[2] -> B[j_off+1, k_off]
            # v[3] -> B[j_off+1, k_off+1]
            if v[0] != 0: V[(j_off) * p + (k_off)][c] = v[0]
            if v[1] != 0: V[(j_off) * p + (k_off + 1)][c] = v[1]
            if v[2] != 0: V[(j_off + 1) * p + (k_off)][c] = v[2]
            if v[3] != 0: V[(j_off + 1) * p + (k_off + 1)][c] = v[3]
            
            # Map w to W
            # w[0] -> C[k_off, i_off] -> index k*n + i
            # w[1] -> C[k_off+1, i_off]
            # w[2] -> C[k_off, i_off+1]
            # w[3] -> C[k_off+1, i_off+1]
            if w[0] != 0: W[(k_off) * n + (i_off)][c] = w[0]
            if w[1] != 0: W[(k_off + 1) * n + (i_off)][c] = w[1]
            if w[2] != 0: W[(k_off) * n + (i_off + 1)][c] = w[2]
            if w[3] != 0: W[(k_off + 1) * n + (i_off + 1)][c] = w[3]
            
        return start_col + 7

    # Block 1: k=0 (2x3x1)
    # Schoolbook for all i, j
    for i in range(n):
        for j in range(m):
            add_schoolbook(i, j, 0, col)
            col += 1
            
    # Block 2: k in {1, 2} (2x3x2)
    # Split A into A_left (j=0,1) and A_right (j=2)
    # Split B into B_top (j=0,1) and B_bottom (j=2)
    # C = A_left B_top + A_right B_bottom
    
    # Part A: A_right B_bottom (2x1 * 1x2)
    # i in {0,1}, k in {1,2}
    # A index: i, 2
    # B index: 2, k
    for i in range(n):
        for k_local in range(2):
            k_global = 1 + k_local
            add_schoolbook(i, 2, k_global, col)
            col += 1
            
    # Part B: A_left B_top (2x2 * 2x2)
    # Use Strassen
    # i_off=0, j_off=0, k_off=1
    col = add_strassen_block(0, 0, 1, col)
    
    # Block 3: k in {3, 4} (2x3x2)
    # Same structure as Block 2
    # Part A: A_right B_bottom
    for i in range(n):
        for k_local in range(2):
            k_global = 3 + k_local
            add_schoolbook(i, 2, k_global, col)
            col += 1
            
    # Part B: A_left B_top
    # i_off=0, j_off=0, k_off=3
    col = add_strassen_block(0, 0, 3, col)
    
    # Check if col == rank
    if col != rank:
        # Fallback if construction failed
        return {"U": [[0]*30 for _ in range(6)], "V": [[0]*30 for _ in range(15)], "W": [[0]*30 for _ in range(10)]}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
