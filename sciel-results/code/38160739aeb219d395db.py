"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen block helper
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        col += 1
        
        # M2
        U[(i0+1) * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] -= 1
        W[(k0+1) * n + i0][col] += 1
        col += 1
        
        # M4
        U[(i0+1) * m + (j0+1)][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + (i0+1)][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + (j0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0+1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        col += 1
        
        # M7
        U[i0 * m + (j0+1)][col] += 1
        U[(i0+1) * m + (j0+1)][col] -= 1
        V[(j0+1) * p + k0][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

    # Schoolbook 2x1x2 block helper
    def add_small(i0, j0, k0):
        nonlocal col
        # A[i0, j0] * B[j0, k0] -> C[i0, k0]
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        # A[i0, j0] * B[j0, k0+1] -> C[i0, k0+1]
        U[i0 * m + j0][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1
        col += 1
        
        # A[i0+1, j0] * B[j0, k0] -> C[i0+1, k0]
        U[(i0+1) * m + j0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        col += 1
        
        # A[i0+1, j0] * B[j0, k0+1] -> C[i0+1, k0+1]
        U[(i0+1) * m + j0][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        col += 1

    # Blocks
    # j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    # j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2)
    # j in {0,1}, k in {4,5}
    add_strassen(0, 0, 4)
    
    # j in {2,3}, k in {0,1}
    add_strassen(0, 2, 0)
    # j in {2,3}, k in {2,3}
    add_strassen(0, 2, 2)
    # j in {2,3}, k in {4,5}
    add_strassen(0, 2, 4)
    
    # j in {4}, k in {0,1}
    add_small(0, 4, 0)
    # j in {4}, k in {2,3}
    add_small(0, 4, 2)
    # j in {4}, k in {4,5}
    add_small(0, 4, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
