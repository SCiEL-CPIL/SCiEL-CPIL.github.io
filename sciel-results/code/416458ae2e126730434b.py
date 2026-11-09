import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # We use a block decomposition.
    # Split A (4x4) into four 2x2 blocks: A00, A01, A10, A11
    # Split B (4x6) into four 2x3 blocks: B00, B01, B10, B11
    # Each 2x3 block is split into 2x2 and 2x1.
    # C (4x6) is formed by 2x3 blocks.
    
    # Total rank will be 8 * (7 + 4) = 88.
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen 2x2x2 decomposition
    # A indices: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # L(A) forms for Strassen
    # M1: a+d -> (1, 0, 0, 1)
    # M2: c+d -> (0, 0, 1, 1)
    # M3: a   -> (1, 0, 0, 0)
    # M4: d   -> (0, 0, 0, 1)
    # M5: a+b -> (1, 1, 0, 0)
    # M6: c-a -> (-1, 0, 1, 0)
    # M7: b-d -> (0, 1, 0, -1)
    
    L_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    R_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Contributions to C (p,q,r,s)
    # p: M1 + M4 - M5 + M7
    # q: M3 + M5
    # r: M2 + M4
    # s: M1 - M2 + M3 + M6
    
    W_strassen = [
        [1, 0, 0, 1],   # M1 -> p, s
        [0, 0, 1, -1],  # M2 -> r, -s
        [0, 1, 0, 1],   # M3 -> q, s
        [1, 0, 1, 0],   # M4 -> p, r
        [-1, 1, 0, 0],  # M5 -> -p, q
        [0, 0, 0, 1],   # M6 -> s
        [1, 0, 0, 0]    # M7 -> p
    ]
    # Note: W_strassen[k][i] is the coefficient of C_i for product k.
    # C indices 0,1,2,3 correspond to p,q,r,s.
    
    # Schoolbook 2x2x1 decomposition
    # A indices: 0,1,2,3
    # B indices: 0,1 (x, y)
    # C indices: 0,1 (u, v)
    
    L_sb = [
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1]  # d
    ]
    
    R_sb = [
        [1, 0], # x
        [0, 1], # y
        [1, 0], # x
        [0, 1]  # y
    ]
    
    W_sb = [
        [1, 0], # u
        [1, 0], # u
        [0, 1], # v
        [0, 1]  # v
    ]
    
    # Mapping of block indices to global indices
    # A block (bi, bj) where bi,bj in {0,1}
    # A global index for element (r,c) in block is (bi*2+r)*m + (bj*2+c)
    # B block (bi, bj) where bi in {0,1}, bj in {0,1,2} (col block)
    # B global index for element (r,c) in block is (bi*2+r)*p + (bj*3+c)
    # C block (ci, cj) where ci in {0,1}, cj in {0,1,2}
    # C global index for element (r,c) in block is (cj*3+c)*n + (ci*2+r)
    # Wait, W index is k*n + i. k is col, i is row.
    # C row i, col k.
    # Block (ci, cj) has rows ci*2..ci*2+1, cols cj*3..cj*3+2.
    # Global C index for (r, c) in block is (cj*3+c)*n + (ci*2+r).
    
    # The 8 terms are:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Each B block is 2x3. Split into B2 (2x2) and B1 (2x1).
    # B2 cols 0,1. B1 col 2.
    
    terms = [
        # (A_block, B_block, C_block)
        ((0,0), (0,0), (0,0)),
        ((0,1), (1,0), (0,0)),
        ((0,0), (0,1), (0,1)),
        ((0,1), (1,1), (0,1)),
        ((1,0), (0,0), (1,0)),
        ((1,1), (1,0), (1,0)),
        ((1,0), (0,1), (1,1)),
        ((1,1), (1,1), (1,1))
    ]
    
    for A_blk, B_blk, C_blk in terms:
        ai, aj = A_blk
        bi, bj = B_blk
        ci, cj = C_blk
        
        # Process 2x2 part of B (cols 0,1 of the 2x3 block)
        # B2 is 2x2.
        for k in range(7):
            # U part
            for r in range(2):
                for c in range(2):
                    if L_strassen[k][r*2+c] != 0:
                        u_idx = (ai*2+r)*m + (aj*2+c)
                        add_term(u_idx, L_strassen[k][r*2+c], -1, 0, -1, 0) # Placeholder, need to fill V and W
                        # We can't add term by term like this easily because we need to fill U, V, W for the SAME column.
                        # Let's collect indices.
                        
        # Reset col for this term? No, col is global.
        # We need to fill U, V, W for the current column.
        
    # The above loop structure is flawed for filling matrices.
    # Let's iterate k, and for each k, determine U, V, W indices and values.
    
    col = 0
    for A_blk, B_blk, C_blk in terms:
        ai, aj = A_blk
        bi, bj = B_blk
        ci, cj = C_blk
        
        # Part 1: Strassen 2x2x2
        for k in range(7):
            # U indices
            for r in range(2):
                for c in range(2):
                    val = L_strassen[k][r*2+c]
                    if val != 0:
                        u_idx = (ai*2+r)*m + (aj*2+c)
                        U[u_idx][col] = val
            
            # V indices (B2 is cols 0,1 of B_blk)
            for r in range(2):
                for c in range(2):
                    val = R_strassen[k][r*2+c]
                    if val != 0:
                        v_idx = (bi*2+r)*p + (bj*3+c)
                        V[v_idx][col] = val
                        
            # W indices (C_blk)
            for r in range(2):
                for c in range(2):
                    val = W_strassen[k][r*2+c]
                    if val != 0:
                        w_idx = (cj*3+c)*n + (ci*2+r)
                        W[w_idx][col] = val
            col += 1
            
        # Part 2: Schoolbook 2x2x1
        for k in range(4):
            # U indices
            for r in range(2):
                for c in range(2):
                    val = L_sb[k][r*2+c]
                    if val != 0:
                        u_idx = (ai*2+r)*m + (aj*2+c)
                        U[u_idx][col] = val
                        
            # V indices (B1 is col 2 of B_blk)
            for r in range(2):
                val = R_sb[k][r] # B1 has 2 rows, 1 col. R_sb[k] is [x] or [y]
                if val != 0:
                    v_idx = (bi*2+r)*p + (bj*3+2)
                    V[v_idx][col] = val
                    
            # W indices (C_blk, col 2)
            for r in range(2):
                val = W_sb[k][r]
                if val != 0:
                    w_idx = (cj*3+2)*n + (ci*2+r)
                    W[w_idx][col] = val
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
