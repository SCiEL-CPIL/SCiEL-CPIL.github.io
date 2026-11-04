import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # U indices
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        # V indices
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        # W indices
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # 1. (x00+x11)(y00+y11)
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # 2. (x10+x11)y00
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # 3. x00(y10-y11)
        U[u00][col] += 1
        V[v10][col] += 1
        V[v11][col] += -1
        W[w11][col] += -1
        col += 1
        
        # 4. x11(y01-y00)
        U[u11][col] += 1
        V[v01][col] += 1
        V[v00][col] += -1
        W[w00][col] += -1
        col += 1
        
        # 5. (x00+x01)y11
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # 6. (x10-x00)(y00+y01)
        U[u10][col] += 1
        U[u00][col] += -1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w00][col] += 1
        col += 1
        
        # 7. (x01-x11)(y10+y11)
        U[u01][col] += 1
        U[u11][col] += -1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w01][col] += 1
        col += 1

    def add_single_block(i0, i1, j0, j1, k0):
        nonlocal col
        # U indices
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        # V indices
        v00 = j0 * p + k0
        v10 = j1 * p + k0
        
        # W indices
        w00 = k0 * n + i0
        w10 = k0 * n + i1
        
        # 1. x00 y00 -> z00
        U[u00][col] += 1
        V[v00][col] += 1
        W[w00][col] += 1
        col += 1
        
        # 2. x01 y10 -> z00
        U[u01][col] += 1
        V[v10][col] += 1
        W[w00][col] += 1
        col += 1
        
        # 3. x10 y00 -> z10
        U[u10][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # 4. x11 y10 -> z10
        U[u11][col] += 1
        V[v10][col] += 1
        W[w10][col] += 1
        col += 1

    # Block 1: A cols 0,1; B rows 0,1
    # i=0,1; j=0,1
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 0, 1, 2, 3)
    add_strassen_block(0, 1, 0, 1, 4, 5)
    add_single_block(0, 1, 0, 1, 6)
    
    # Block 2: A cols 2,3; B rows 2,3
    # i=0,1; j=2,3
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    add_strassen_block(0, 1, 2, 3, 4, 5)
    add_single_block(0, 1, 2, 3, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
