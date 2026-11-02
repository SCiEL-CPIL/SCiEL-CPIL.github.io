"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_col, x_idx, y_idx, z_idx):
        # x_idx: [00, 01, 10, 11]
        # y_idx: [00, 01, 10, 11]
        # z_idx: [00, 01, 10, 11]
        
        # M1 = (x00+x11)(y00+y11) -> z00 += 1, z11 += 1
        c = base_col
        U[x_idx[0]][c] += 1; U[x_idx[3]][c] += 1
        V[y_idx[0]][c] += 1; V[y_idx[3]][c] += 1
        W[z_idx[0]][c] += 1; W[z_idx[3]][c] += 1
        c += 1
        
        # M2 = (x10+x11)y00 -> z10 += 1, z11 -= 1
        U[x_idx[2]][c] += 1; U[x_idx[3]][c] += 1
        V[y_idx[0]][c] += 1
        W[z_idx[2]][c] += 1; W[z_idx[3]][c] += -1
        c += 1
        
        # M3 = x00(y01-y11) -> z01 += 1, z11 += 1
        U[x_idx[0]][c] += 1
        V[y_idx[1]][c] += 1; V[y_idx[3]][c] += -1
        W[z_idx[1]][c] += 1; W[z_idx[3]][c] += 1
        c += 1
        
        # M4 = x11(y10-y00) -> z00 += 1, z10 += 1
        U[x_idx[3]][c] += 1
        V[y_idx[2]][c] += 1; V[y_idx[0]][c] += -1
        W[z_idx[0]][c] += 1; W[z_idx[2]][c] += 1
        c += 1
        
        # M5 = (x00+x01)y11 -> z00 -= 1, z01 += 1
        U[x_idx[0]][c] += 1; U[x_idx[1]][c] += 1
        V[y_idx[3]][c] += 1
        W[z_idx[0]][c] += -1; W[z_idx[1]][c] += 1
        c += 1
        
        # M6 = (x10-x00)(y00+y01) -> z11 += 1
        U[x_idx[2]][c] += 1; U[x_idx[0]][c] += -1
        V[y_idx[0]][c] += 1; V[y_idx[1]][c] += 1
        W[z_idx[3]][c] += 1
        c += 1
        
        # M7 = (x01-x11)(y10+y11) -> z00 += 1, z01 += 1
        U[x_idx[1]][c] += 1; U[x_idx[3]][c] += -1
        V[y_idx[2]][c] += 1; V[y_idx[3]][c] += 1
        W[z_idx[0]][c] += 1; W[z_idx[1]][c] += 1
        c += 1
        
        return c

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,4,5
    # B indices: 0,1,4,5
    # C indices: 0,1,2,3
    x1 = [0, 1, 4, 5]
    y1 = [0, 1, 4, 5]
    z1 = [0, 1, 2, 3]
    col = add_strassen_block(0, x1, y1, z1)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A indices: 2,3,6,7
    # B indices: 8,9,12,13
    # C indices: 0,1,2,3
    x2 = [2, 3, 6, 7]
    y2 = [8, 9, 12, 13]
    z2 = [0, 1, 2, 3]
    col = add_strassen_block(col, x2, y2, z2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: 0,1,4,5
    # B indices: 2,3,6,7
    # C indices: 4,5,6,7
    x3 = [0, 1, 4, 5]
    y3 = [2, 3, 6, 7]
    z3 = [4, 5, 6, 7]
    col = add_strassen_block(col, x3, y3, z3)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A indices: 2,3,6,7
    # B indices: 10,11,14,15
    # C indices: 4,5,6,7
    x4 = [2, 3, 6, 7]
    y4 = [10, 11, 14, 15]
    z4 = [4, 5, 6, 7]
    col = add_strassen_block(col, x4, y4, z4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
