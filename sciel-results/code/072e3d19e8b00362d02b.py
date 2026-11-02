"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition:
    # A (2x4) = [A1 (2x2), A2 (2x2)]
    # B (4x7) = [B1 (2x7); B2 (2x7)]
    # C = A1*B1 + A2*B2
    # Each Ai*Bi (2x2 * 2x7) is decomposed into:
    # 3 Strassen blocks (2x2 * 2x2) + 1 Schoolbook block (2x2 * 2x1)
    # Rank per block: 3*7 + 4 = 25
    # Total rank: 50
    
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # U indices for A[i0:i0+2, j0:j0+2]
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + j0 + 1
        
        # V indices for B[j0:j0+2, k0:k0+2]
        v00 = j0 * p + k0
        v01 = j0 * p + k0 + 1
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + k0 + 1
        
        # W indices for C[i0:i0+2, k0:k0+2]
        # W index is k*n + i
        w00 = k0 * n + i0
        w01 = k0 * n + i0 + 1
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + i0 + 1
        
        # m1 = (x00 + x11)(y00 + y11)
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1 # z00
        W[w11][col] += 1 # z11
        col += 1
        
        # m2 = (x10 + x11)y00
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1 # z10
        col += 1
        
        # m3 = x00(y01 - y11)
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] -= 1
        W[w01][col] += 1 # z01
        col += 1
        
        # m4 = x11(y10 - y00)
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] -= 1
        W[w00][col] += 1 # z00
        W[w10][col] += 1 # z10
        col += 1
        
        # m5 = (x00 + x01)y11
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w01][col] += 1 # z01
        col += 1
        
        # m6 = (x10 - x00)(y00 + y01)
        U[u10][col] += 1
        U[u00][col] -= 1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1 # z11
        col += 1
        
        # m7 = (x01 - x11)(y10 + y11)
        U[u01][col] += 1
        U[u11][col] -= 1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1 # z00
        col += 1
        
    def add_schoolbook_2x2x1(i0, j0, k0):
        nonlocal col
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + j0 + 1
        
        v00 = j0 * p + k0
        v10 = (j0 + 1) * p + k0
        
        w00 = k0 * n + i0
        w10 = k0 * n + i0 + 1
        
        # z00 = x00 y00 + x01 y10
        U[u00][col] += 1
        V[v00][col] += 1
        W[w00][col] += 1
        col += 1
        
        U[u01][col] += 1
        V[v10][col] += 1
        W[w00][col] += 1
        col += 1
        
        # z10 = x10 y00 + x11 y10
        U[u10][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        U[u11][col] += 1
        V[v10][col] += 1
        W[w10][col] += 1
        col += 1
        
    # Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_schoolbook_2x2x1(0, 0, 6)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:7]
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    add_schoolbook_2x2x1(0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
