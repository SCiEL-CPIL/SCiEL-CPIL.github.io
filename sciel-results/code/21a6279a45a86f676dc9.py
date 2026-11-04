"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen-like blocks
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 Strassen terms for a 2x2 x 2x2 block
    # Indices:
    # A: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # B: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # C: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # Helper to add a term
        def add_term(u_entries, v_entries, w_entries):
            nonlocal col
            for idx, val in u_entries.items():
                U[idx][col] += val
            for idx, val in v_entries.items():
                V[idx][col] += val
            for idx, val in w_entries.items():
                W[idx][col] += val
            col += 1

        # P1 = (A[i0, j0] + A[i0+1, j0+1]) * (B[j0, k0] + B[j0+1, k0+1])
        add_term(
            {i0*m + j0: 1, (i0+1)*m + (j0+1): 1},
            {j0*p + k0: 1, (j0+1)*p + (k0+1): 1},
            {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        )
        
        # P2 = (A[i0+1, j0] + A[i0+1, j0+1]) * B[j0, k0]
        add_term(
            {(i0+1)*m + j0: 1, (i0+1)*m + (j0+1): 1},
            {j0*p + k0: 1},
            {k0*n + (i0+1): 1, (k0+1)*n + (i0+1): -1}
        )
        
        # P3 = A[i0, j0] * (B[j0, k0+1] - B[j0+1, k0+1])
        add_term(
            {i0*m + j0: 1},
            {j0*p + (k0+1): 1, (j0+1)*p + (k0+1): -1},
            {(k0+1)*n + i0: 1, (k0+1)*n + (i0+1): 1}
        )
        
        # P4 = A[i0+1, j0+1] * (B[j0+1, k0] - B[j0, k0])
        add_term(
            {(i0+1)*m + (j0+1): 1},
            {(j0+1)*p + k0: 1, j0*p + k0: -1},
            {k0*n + i0: 1, k0*n + (i0+1): 1}
        )
        
        # P5 = (A[i0, j0] + A[i0, j0+1]) * B[j0+1, k0+1]
        add_term(
            {i0*m + j0: 1, i0*m + (j0+1): 1},
            {(j0+1)*p + (k0+1): 1},
            {k0*n + i0: -1, (k0+1)*n + i0: 1}
        )
        
        # P6 = (A[i0+1, j0] - A[i0, j0]) * (B[j0, k0] + B[j0, k0+1])
        add_term(
            {(i0+1)*m + j0: 1, i0*m + j0: -1},
            {j0*p + k0: 1, j0*p + (k0+1): 1},
            {(k0+1)*n + (i0+1): 1}
        )
        
        # P7 = (A[i0, j0+1] - A[i0+1, j0+1]) * (B[j0+1, k0] + B[j0+1, k0+1])
        add_term(
            {i0*m + (j0+1): 1, (i0+1)*m + (j0+1): -1},
            {(j0+1)*p + k0: 1, (j0+1)*p + (k0+1): 1},
            {k0*n + i0: 1}
        )

    # Main Block 1: i0=0, j0=0
    # Sub-blocks for k: 0-1, 2-3, 4-5
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Main Block 2: i0=0, j0=2
    # Sub-blocks for k: 0-1, 2-3, 4-5
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
