"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 multiplication
    # x00, x01, x10, x11 and y00, y01, y10, y11
    # Products M1..M7
    
    # Block definitions:
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def apply_strassen(u_rows, v_rows, w_rows, offset):
        # u_rows: [u00, u01, u10, u11]
        # v_rows: [v00, v01, v10, v11]
        # w_rows: [w00, w01, w10, w11]
        
        # M1 = (x00 + x11)(y00 + y11)
        col = offset
        U[u_rows[0]][col] += 1
        U[u_rows[3]][col] += 1
        V[v_rows[0]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1 # Z00
        W[w_rows[3]][col] += 1 # Z11
        
        # M2 = (x10 + x11)y00
        col = offset + 1
        U[u_rows[2]][col] += 1
        U[u_rows[3]][col] += 1
        V[v_rows[0]][col] += 1
        W[w_rows[2]][col] += 1 # Z10
        W[w_rows[3]][col] -= 1 # Z11
        
        # M3 = x00(y01 - y11)
        col = offset + 2
        U[u_rows[0]][col] += 1
        V[v_rows[1]][col] += 1
        V[v_rows[3]][col] -= 1
        W[w_rows[1]][col] += 1 # Z01
        W[w_rows[3]][col] += 1 # Z11
        
        # M4 = x11(y10 - y00)
        col = offset + 3
        U[u_rows[3]][col] += 1
        V[v_rows[2]][col] += 1
        V[v_rows[0]][col] -= 1
        W[w_rows[0]][col] += 1 # Z00
        W[w_rows[2]][col] += 1 # Z10
        
        # M5 = (x00 + x01)y11
        col = offset + 4
        U[u_rows[0]][col] += 1
        U[u_rows[1]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] -= 1 # Z00
        W[w_rows[1]][col] += 1 # Z01
        
        # M6 = (x10 - x00)(y00 + y01)
        col = offset + 5
        U[u_rows[2]][col] += 1
        U[u_rows[0]][col] -= 1
        V[v_rows[0]][col] += 1
        V[v_rows[1]][col] += 1
        W[w_rows[3]][col] += 1 # Z11
        
        # M7 = (x01 - x11)(y10 + y11)
        col = offset + 6
        U[u_rows[1]][col] += 1
        U[u_rows[3]][col] -= 1
        V[v_rows[2]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1 # Z00

    # Block 0: A[0:2, 0:2] * B[0:2, 0:2]
    # A rows 0,1; cols 0,1. B rows 0,1; cols 0,1. C rows 0,1; cols 0,1.
    u0 = [0, 1, 4, 5]
    v0 = [0, 1, 4, 5]
    w0 = [0, 1, 2, 3]
    apply_strassen(u0, v0, w0, 0)

    # Block 1: A[0:2, 0:2] * B[0:2, 2:4]
    # A rows 0,1; cols 0,1. B rows 0,1; cols 2,3. C rows 0,1; cols 2,3.
    u1 = [0, 1, 4, 5]
    v1 = [2, 3, 6, 7]
    w1 = [4, 5, 6, 7]
    apply_strassen(u1, v1, w1, 7)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2]
    # A rows 0,1; cols 2,3. B rows 2,3; cols 0,1. C rows 0,1; cols 0,1.
    u2 = [2, 3, 6, 7]
    v2 = [8, 9, 12, 13]
    w2 = [0, 1, 2, 3]
    apply_strassen(u2, v2, w2, 14)

    # Block 3: A[0:2, 2:4] * B[2:4, 2:4]
    # A rows 0,1; cols 2,3. B rows 2,3; cols 2,3. C rows 0,1; cols 2,3.
    u3 = [2, 3, 6, 7]
    v3 = [10, 11, 14, 15]
    w3 = [4, 5, 6, 7]
    apply_strassen(u3, v3, w3, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
