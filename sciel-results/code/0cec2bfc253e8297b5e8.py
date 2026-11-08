"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We aim for rank 36 using block decomposition.
    # A (2x4) split into A_L (2x2), A_R (2x2)
    # B (4x5) split into B_T (2x5), B_B (2x5)
    # C (2x5) = A_L B_T + A_R B_B
    
    # B_T (2x5) split into B_T1 (2x2), B_T2 (2x2), B_T3 (2x1)
    # B_B (2x5) split into B_B1 (2x2), B_B2 (2x2), B_B3 (2x1)
    
    # Costs:
    # A_L B_T1: 2x2x2 -> 7 (Strassen)
    # A_L B_T2: 2x2x2 -> 7 (Strassen)
    # A_L B_T3: 2x2x1 -> 4 (Standard)
    # A_R B_B1: 2x2x2 -> 7 (Strassen)
    # A_R B_B2: 2x2x2 -> 7 (Strassen)
    # A_R B_B3: 2x2x1 -> 4 (Standard)
    # Total: 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_2x2(y, z, w_offsets):
        """
        y: list of 4 (i,j) indices for the 2x2 Y block
        z: list of 4 (j,k) indices for the 2x2 Z block
        w_offsets: list of 4 (k,i) indices for the 2x2 result positions
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w_offsets[2])][col] += 1
        W[w_idx(*w_offsets[3])][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*w_offsets[1])][col] += 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*w_offsets[0])][col] += 1
        W[w_idx(*w_offsets[2])][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] -= 1
        W[w_idx(*w_offsets[1])][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        col += 1

    def standard_2x2x1(y, z, w_offsets):
        """
        y: list of 4 (i,j) indices for the 2x2 Y block
        z: list of 2 (j,k) indices for the 2x1 Z block
        w_offsets: list of 2 (k,i) indices for the 2x1 result positions
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z0, z1 = z
        
        # C0 = y00*z0 + y01*z1
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z0)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        col += 1
        
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z1)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        col += 1
        
        # C1 = y10*z0 + y11*z1
        U[u_idx(*y10)][col] = 1
        V[v_idx(*z0)][col] = 1
        W[w_idx(*w_offsets[1])][col] += 1
        col += 1
        
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z1)][col] = 1
        W[w_idx(*w_offsets[1])][col] += 1
        col += 1

    # Define blocks
    # A_L: rows 0,1 cols 0,1
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    # A_R: rows 0,1 cols 2,3
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # B_T: rows 0,1, all cols
    # B_T1: cols 0,1
    B_T1 = [(0,0), (0,1), (1,0), (1,1)]
    # B_T2: cols 2,3
    B_T2 = [(0,2), (0,3), (1,2), (1,3)]
    # B_T3: col 4
    B_T3 = [(0,4), (1,4)]
    
    # B_B: rows 2,3, all cols
    # B_B1: cols 0,1
    B_B1 = [(2,0), (2,1), (3,0), (3,1)]
    # B_B2: cols 2,3
    B_B2 = [(2,2), (2,3), (3,2), (3,3)]
    # B_B3: col 4
    B_B3 = [(2,4), (3,4)]
    
    # W offsets for C (2x5)
    # C[:, 0:2]
    C_01 = [(0,0), (1,0), (0,1), (1,1)]
    # C[:, 2:4]
    C_23 = [(2,0), (3,0), (2,1), (3,1)]
    # C[:, 4]
    C_4 = [(4,0), (4,1)]
    
    # Execute
    strassen_2x2(A_L, B_T1, C_01)
    strassen_2x2(A_L, B_T2, C_23)
    standard_2x2x1(A_L, B_T3, C_4)
    
    strassen_2x2(A_R, B_B1, C_01)
    strassen_2x2(A_R, B_B2, C_23)
    standard_2x2x1(A_R, B_B3, C_4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
