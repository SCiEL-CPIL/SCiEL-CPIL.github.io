import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # A is 2x4, split into A1 (2x2) and A2 (2x2) along columns.
    # B is 4x5, split into B1 (2x5) and B2 (2x5) along rows.
    # B1, B2 are further split into 2x2 and 2x1 blocks along columns.
    # C = A1*B1 + A2*B2
    # C1 (2x2) = A1*B11 + A2*B21
    # C2 (2x2) = A1*B12 + A2*B22
    # C3 (2x1) = A1*B13 + A2*B23
    
    # Total Rank:
    # C1: 2 Strassen blocks (2x2x2) -> 7 + 7 = 14
    # C2: 2 Strassen blocks (2x2x2) -> 7 + 7 = 14
    # C3: 2 Schoolbook blocks (2x2x1) -> 4 + 4 = 8
    # Total Rank = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(A_i_off, A_j_off, B_i_off, B_j_off, C_i_off, C_k_off):
        nonlocal col
        # A is 2x2 at [A_i_off:A_i_off+2, A_j_off:A_j_off+2]
        # B is 2x2 at [B_i_off:B_i_off+2, B_j_off:B_j_off+2]
        # C is 2x2 at [C_i_off:C_i_off+2, C_k_off:C_k_off+2]
        
        # Strassen factors:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b00-b01)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # c00 = m1 - m4 + m5 - m7
        # c01 = m2 + m4
        # c10 = m3 + m7
        # c11 = m1 - m2 + m3 + m6
        
        # Indices in A: (A_i_off + di, A_j_off + dj) -> index = (A_i_off+di)*m + (A_j_off+dj)
        # Indices in B: (B_i_off + di, B_j_off + dj) -> index = (B_i_off+di)*p + (B_j_off+dj)
        # Indices in C: (C_i_off + di, C_k_off + dk) -> index = (C_k_off+dk)*n + (C_i_off+di)
        
        def get_U_idx(di, dj):
            return (A_i_off + di) * m + (A_j_off + dj)
            
        def get_V_idx(di, dj):
            return (B_i_off + di) * p + (B_j_off + dj)
            
        def get_W_idx(di, dk):
            return (C_k_off + dk) * n + (C_i_off + di)

        # m1
        # A: (0,0)+1, (1,1)+1
        # B: (0,0)+1, (1,1)+1
        # C: (0,0)+1, (1,1)+1
        U[get_U_idx(0,0)][col] += 1
        U[get_U_idx(1,1)][col] += 1
        V[get_V_idx(0,0)][col] += 1
        V[get_V_idx(1,1)][col] += 1
        W[get_W_idx(0,0)][col] += 1
        W[get_W_idx(1,1)][col] += 1
        col += 1
        
        # m2
        # A: (1,0)+1, (1,1)+1
        # B: (0,0)+1
        # C: (0,1)+1, (1,1)-1
        U[get_U_idx(1,0)][col] += 1
        U[get_U_idx(1,1)][col] += 1
        V[get_V_idx(0,0)][col] += 1
        W[get_W_idx(0,1)][col] += 1
        W[get_W_idx(1,1)][col] -= 1
        col += 1
        
        # m3
        # A: (0,0)+1
        # B: (1,0)+1, (1,1)-1
        # C: (1,0)+1, (1,1)+1
        U[get_U_idx(0,0)][col] += 1
        V[get_V_idx(1,0)][col] += 1
        V[get_V_idx(1,1)][col] -= 1
        W[get_W_idx(1,0)][col] += 1
        W[get_W_idx(1,1)][col] += 1
        col += 1
        
        # m4
        # A: (1,1)+1
        # B: (0,0)+1, (0,1)-1
        # C: (0,0)-1, (0,1)+1
        U[get_U_idx(1,1)][col] += 1
        V[get_V_idx(0,0)][col] += 1
        V[get_V_idx(0,1)][col] -= 1
        W[get_W_idx(0,0)][col] -= 1
        W[get_W_idx(0,1)][col] += 1
        col += 1
        
        # m5
        # A: (0,0)+1, (0,1)+1
        # B: (1,1)+1
        # C: (0,0)+1, (1,1)+1
        U[get_U_idx(0,0)][col] += 1
        U[get_U_idx(0,1)][col] += 1
        V[get_V_idx(1,1)][col] += 1
        W[get_W_idx(0,0)][col] += 1
        W[get_W_idx(1,1)][col] += 1
        col += 1
        
        # m6
        # A: (1,0)+1, (0,0)-1
        # B: (0,0)+1, (0,1)+1
        # C: (1,1)+1
        U[get_U_idx(1,0)][col] += 1
        U[get_U_idx(0,0)][col] -= 1
        V[get_V_idx(0,0)][col] += 1
        V[get_V_idx(0,1)][col] += 1
        W[get_W_idx(1,1)][col] += 1
        col += 1
        
        # m7
        # A: (0,1)+1, (1,1)-1
        # B: (1,0)+1, (1,1)+1
        # C: (0,0)-1, (1,0)+1
        U[get_U_idx(0,1)][col] += 1
        U[get_U_idx(1,1)][col] -= 1
        V[get_V_idx(1,0)][col] += 1
        V[get_V_idx(1,1)][col] += 1
        W[get_W_idx(0,0)][col] -= 1
        W[get_W_idx(1,0)][col] += 1
        col += 1

    def add_schoolbook_2x2x1(A_i_off, A_j_off, B_i_off, B_j_off, C_i_off, C_k_off):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        
        def get_U_idx(di, dj):
            return (A_i_off + di) * m + (A_j_off + dj)
            
        def get_V_idx(di, dj):
            return (B_i_off + di) * p + (B_j_off + dj)
            
        def get_W_idx(di, dk):
            return (C_k_off + dk) * n + (C_i_off + di)

        # Term 1: a00 * b0 -> c0
        U[get_U_idx(0,0)][col] += 1
        V[get_V_idx(0,0)][col] += 1
        W[get_W_idx(0,0)][col] += 1
        col += 1
        
        # Term 2: a01 * b1 -> c0
        U[get_U_idx(0,1)][col] += 1
        V[get_V_idx(1,0)][col] += 1
        W[get_W_idx(0,0)][col] += 1
        col += 1
        
        # Term 3: a10 * b0 -> c1
        U[get_U_idx(1,0)][col] += 1
        V[get_V_idx(0,0)][col] += 1
        W[get_W_idx(1,0)][col] += 1
        col += 1
        
        # Term 4: a11 * b1 -> c1
        U[get_U_idx(1,1)][col] += 1
        V[get_V_idx(1,0)][col] += 1
        W[get_W_idx(1,0)][col] += 1
        col += 1

    # Block 1: C1 = A1*B11 + A2*B21
    # A1: i=0, j=0. B11: i=0, j=0. C1: i=0, k=0.
    add_strassen(0, 0, 0, 0, 0, 0)
    # A2: i=0, j=2. B21: i=2, j=0. C1: i=0, k=0.
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Block 2: C2 = A1*B12 + A2*B22
    # A1: i=0, j=0. B12: i=0, j=2. C2: i=0, k=2.
    add_strassen(0, 0, 0, 2, 0, 2)
    # A2: i=0, j=2. B22: i=2, j=2. C2: i=0, k=2.
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Block 3: C3 = A1*B13 + A2*B23
    # A1: i=0, j=0. B13: i=0, j=4. C3: i=0, k=4.
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)
    # A2: i=0, j=2. B23: i=2, j=4. C3: i=0, k=4.
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
