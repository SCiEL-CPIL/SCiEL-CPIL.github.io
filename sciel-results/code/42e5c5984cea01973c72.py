"""Rewritten program for matrix multiplication 3x4x4."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-44 tensor decomposition for 3x4x4 matrix multiplication
    using a mix of Strassen (2x2x2) and schoolbook (1x2x2) blocks.
    """
    n, m, p = DIMENSIONS
    # Total rank: 4 Strassen blocks (7 each) + 4 Schoolbook blocks (4 each)
    rank = 4 * 7 + 4 * 4
    
    # U shape: (n*m, rank) = (12, 44)
    # V shape: (m*p, rank) = (16, 44)
    # W shape: (p*n, rank) = (12, 44)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_r, A_c, B_r, B_c, C_r, C_c):
        nonlocal col
        # A is 2x2 at (A_r, A_c)
        # B is 2x2 at (B_r, B_c)
        # C is 2x2 at (C_r, C_c)
        
        # Helper to get U index
        def u_idx(r, c):
            return (A_r + r) * m + (A_c + c)
            
        # Helper to get V index
        def v_idx(r, c):
            return (B_r + r) * p + (B_c + c)
            
        # Helper to get W index
        def w_idx(r, c):
            return (C_c + c) * n + (C_r + r)
            
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[u_idx(0,0)][col] += 1.0
        U[u_idx(1,1)][col] += 1.0
        V[v_idx(0,0)][col] += 1.0
        V[v_idx(1,1)][col] += 1.0
        W[w_idx(0,0)][col] += 1.0
        W[w_idx(1,1)][col] += 1.0
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[u_idx(1,0)][col] += 1.0
        U[u_idx(1,1)][col] += 1.0
        V[v_idx(0,0)][col] += 1.0
        W[w_idx(1,0)][col] += 1.0
        W[w_idx(1,1)][col] -= 1.0
        col += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[u_idx(0,0)][col] += 1.0
        V[v_idx(0,1)][col] += 1.0
        V[v_idx(1,1)][col] -= 1.0
        W[w_idx(0,1)][col] += 1.0
        W[w_idx(1,1)][col] += 1.0
        col += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[u_idx(1,1)][col] += 1.0
        V[v_idx(1,0)][col] += 1.0
        V[v_idx(0,0)][col] -= 1.0
        W[w_idx(0,0)][col] += 1.0
        W[w_idx(1,0)][col] += 1.0
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        U[u_idx(0,0)][col] += 1.0
        U[u_idx(0,1)][col] += 1.0
        V[v_idx(1,1)][col] += 1.0
        W[w_idx(0,1)][col] += 1.0
        W[w_idx(0,0)][col] -= 1.0
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U[u_idx(1,0)][col] += 1.0
        U[u_idx(0,0)][col] -= 1.0
        V[v_idx(0,0)][col] += 1.0
        V[v_idx(0,1)][col] += 1.0
        W[w_idx(1,1)][col] += 1.0
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U[u_idx(0,1)][col] += 1.0
        U[u_idx(1,1)][col] -= 1.0
        V[v_idx(1,0)][col] += 1.0
        V[v_idx(1,1)][col] += 1.0
        W[w_idx(0,0)][col] += 1.0
        col += 1

    def add_schoolbook_block(A_r, A_c, B_r, B_c, C_r, C_c):
        nonlocal col
        # A is 1x2 at (A_r, A_c)
        # B is 2x2 at (B_r, B_c)
        # C is 1x2 at (C_r, C_c)
        
        # Helper to get U index
        def u_idx(r, c):
            return (A_r + r) * m + (A_c + c)
            
        # Helper to get V index
        def v_idx(r, c):
            return (B_r + r) * p + (B_c + c)
            
        # Helper to get W index
        def w_idx(r, c):
            return (C_c + c) * n + (C_r + r)
            
        # Term 1: A0 B0 -> C00
        U[u_idx(0,0)][col] += 1.0
        V[v_idx(0,0)][col] += 1.0
        W[w_idx(0,0)][col] += 1.0
        col += 1
        
        # Term 2: A1 B1 -> C00
        U[u_idx(0,1)][col] += 1.0
        V[v_idx(1,0)][col] += 1.0
        W[w_idx(0,0)][col] += 1.0
        col += 1
        
        # Term 3: A0 B2 -> C01
        U[u_idx(0,0)][col] += 1.0
        V[v_idx(0,1)][col] += 1.0
        W[w_idx(0,1)][col] += 1.0
        col += 1
        
        # Term 4: A1 B3 -> C01
        U[u_idx(0,1)][col] += 1.0
        V[v_idx(1,1)][col] += 1.0
        W[w_idx(0,1)][col] += 1.0
        col += 1

    # Block 1: A_top_left (0:2, 0:2) * B_left_top (0:2, 0:2) -> C_top_left (0:2, 0:2)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A_top_right (0:2, 2:4) * B_left_bot (2:4, 0:2) -> C_top_left (0:2, 0:2)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 3: A_top_left (0:2, 0:2) * B_right_top (0:2, 2:4) -> C_top_right (0:2, 2:4)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # Block 4: A_top_right (0:2, 2:4) * B_right_bot (2:4, 2:4) -> C_top_right (0:2, 2:4)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block 5: A_bot_left (2:3, 0:2) * B_left_top (0:2, 0:2) -> C_bot_left (2:3, 0:2)
    add_schoolbook_block(2, 0, 0, 0, 2, 0)
    
    # Block 6: A_bot_right (2:3, 2:4) * B_left_bot (2:4, 0:2) -> C_bot_left (2:3, 0:2)
    add_schoolbook_block(2, 2, 2, 0, 2, 0)
    
    # Block 7: A_bot_left (2:3, 0:2) * B_right_top (0:2, 2:4) -> C_bot_right (2:3, 2:4)
    add_schoolbook_block(2, 0, 0, 2, 2, 2)
    
    # Block 8: A_bot_right (2:3, 2:4) * B_right_bot (2:4, 2:4) -> C_bot_right (2:3, 2:4)
    add_schoolbook_block(2, 2, 2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
