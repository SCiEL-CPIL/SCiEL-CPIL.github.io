"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 84 decomposition
    # Split B (4x6) into three 4x2 blocks
    # Each 4x2 block multiplication uses rank 28 (4 Strassen 2x2x2 products)
    
    # Helper: Generate 7 Strassen terms for 2x2x2 multiplication
    # Returns list of (u, v, w) where u, v, w are lists of 4 elements
    # u corresponds to X (2x2), v to Y (2x2), w to Z (2x2)
    # Indexing: 0:11, 1:12, 2:21, 3:22
    
    def get_strassen_terms():
        terms = []
        # M1 = (x00+x11)(y00+y11) -> c00, c01, c10, c11 contributions
        # c00: +1, c01: 0, c10: 0, c11: +1 ? No.
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 - M6
        
        # Let's define u, v, w for each M
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] ? 
        # Wait, M1 contributes to c00 (+1) and c11 (+1).
        # w1 = [1, 0, 0, 1]
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # M2 contributes to c10 (+1) and c11 (-1).
        # w2 = [0, 0, 1, -1]
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        
        # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # M3 contributes to c01 (+1) and c11 (+1).
        # w3 = [0, 1, 0, 1]
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
        
        # M4: u=[0,0,0,1], v=[-1,0,1,0], w=[1,0,1,0]
        # M4 contributes to c00 (+1) and c10 (+1).
        # w4 = [1, 0, 1, 0]
        terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
        
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
        # M5 contributes to c00 (-1) and c01 (+1).
        # w5 = [-1, 1, 0, 0]
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        
        # M6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,-1]
        # M6 contributes to c11 (-1).
        # w6 = [0, 0, 0, -1]
        terms.append(([1,0,-1,0], [1,1,0,0], [0,0,0,1])) # Note: u sign flip, w sign flip cancels?
        # Original: (a21-a11)(b11+b12) = a21b11 + a21b12 - a11b11 - a11b12
        # u = [ -1, 0, 1, 0 ]? No, a11 is index 0, a21 is index 2.
        # u = [-1, 0, 1, 0]
        # v = [1, 1, 0, 0]
        # w = [0, 0, 0, -1]
        # Let's re-verify M6 contribution.
        # c11 has -M6. So if term is u,v,w, it adds w to c11 with coeff -1?
        # No, the decomposition is sum of u_k v_k w_k.
        # So if c11 = ... - M6, then the term for M6 in the sum must have w_11 = -1.
        # So u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,-1] is correct.
        
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        # M7 contributes to c00 (+1).
        # w7 = [1, 0, 0, 0]
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # Build U, V, W for one 4x4x2 block
    # U size 16x28, V size 8x28, W size 8x28
    # A indices: 0..15 (4x4)
    # B indices: 0..7 (4x2) -> Top 0..3, Bot 4..7
    # C indices: 0..7 (4x2) -> Top 0..3, Bot 4..7
    
    def build_4x4x2_block():
        U_block = [[0]*28 for _ in range(16)]
        V_block = [[0]*28 for _ in range(8)]
        W_block = [[0]*28 for _ in range(8)]
        
        col = 0
        # Products:
        # 1. A11 * B_top -> C_top_part1
        # 2. A12 * B_bot -> C_top_part2
        # 3. A21 * B_top -> C_bot_part1
        # 4. A22 * B_bot -> C_bot_part2
        
        # A11: rows 0,1 cols 0,1 -> indices 0,1,4,5
        # A12: rows 0,1 cols 2,3 -> indices 2,3,6,7
        # A21: rows 2,3 cols 0,1 -> indices 8,9,12,13
        # A22: rows 2,3 cols 2,3 -> indices 10,11,14,15
        
        # B_top: rows 0,1 cols 0,1 -> indices 0,1,2,3
        # B_bot: rows 2,3 cols 0,1 -> indices 4,5,6,7
        
        # C_top: rows 0,1 cols 0,1 -> indices 0,1,2,3
        # C_bot: rows 2,3 cols 0,1 -> indices 4,5,6,7
        
        def map_strassen_to_block(A_indices, B_indices, C_indices, offset):
            # A_indices: list of 4 global U indices corresponding to 0,1,2,3 local
            # B_indices: list of 4 global V indices
            # C_indices: list of 4 global W indices
            for u_loc, v_loc, w_loc in strassen_terms:
                for i in range(4):
                    if u_loc[i] != 0:
                        U_block[A_indices[i]][offset] = u_loc[i]
                    if v_loc[i] != 0:
                        V_block[B_indices[i]][offset] = v_loc[i]
                    if w_loc[i] != 0:
                        W_block[C_indices[i]][offset] = w_loc[i]
                offset += 1
            return offset
        
        # 1. A11 * B_top
        A11_idx = [0, 1, 4, 5]
        B_top_idx = [0, 1, 2, 3]
        C_top_idx = [0, 1, 2, 3]
        col = map_strassen_to_block(A11_idx, B_top_idx, C_top_idx, col)
        
        # 2. A12 * B_bot
        A12_idx = [2, 3, 6, 7]
        B_bot_idx = [4, 5, 6, 7]
        C_top_idx = [0, 1, 2, 3] # Adds to C_top
        col = map_strassen_to_block(A12_idx, B_bot_idx, C_top_idx, col)
        
        # 3. A21 * B_top
        A21_idx = [8, 9, 12, 13]
        B_top_idx = [0, 1, 2, 3]
        C_bot_idx = [4, 5, 6, 7]
        col = map_strassen_to_block(A21_idx, B_top_idx, C_bot_idx, col)
        
        # 4. A22 * B_bot
        A22_idx = [10, 11, 14, 15]
        B_bot_idx = [4, 5, 6, 7]
        C_bot_idx = [4, 5, 6, 7]
        col = map_strassen_to_block(A22_idx, B_bot_idx, C_bot_idx, col)
        
        return U_block, V_block, W_block

    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Create 3 blocks
    for i in range(3):
        Ub, Vb, Wb = build_4x4x2_block()
        U_blocks.append(Ub)
        # V indices for block i:
        # B is 4x6. Block i uses columns 2*i, 2*i+1.
        # V global index: row * 6 + col.
        # Local V indices 0..3 correspond to rows 0,1 cols 0,1 of the 4x2 block.
        # Global V indices:
        # Local 0 (r0,c0) -> Global (0, 2*i)
        # Local 1 (r0,c1) -> Global (0, 2*i+1)
        # Local 2 (r1,c0) -> Global (1, 2*i)
        # Local 3 (r1,c1) -> Global (1, 2*i+1)
        # Local 4 (r2,c0) -> Global (2, 2*i)
        # Local 5 (r2,c1) -> Global (2, 2*i+1)
        # Local 6 (r3,c0) -> Global (3, 2*i)
        # Local 7 (r3,c1) -> Global (3, 2*i+1)
        
        V_global = [[0]*28 for _ in range(24)]
        W_global = [[0]*28 for _ in range(24)]
        
        # Map V
        for r in range(4):
            for c in range(2):
                local_idx = r * 2 + c
                global_idx = r * 6 + (2*i + c)
                for k in range(28):
                    V_global[global_idx][k] = Vb[local_idx][k]
                    
        # Map W
        # W size p*n = 6*4 = 24.
        # W[k*n+i, r] convention?
        # Task says: W[k*n+i, r].
        # k is col of B (0..5), i is row of A (0..3).
        # Our local W is for 4x2 C.
        # Local W indices 0..3 correspond to C rows 0,1 cols 0,1.
        # Local 0 (r0,c0) -> Global (k=2*i, i=0) -> Index (2*i)*4 + 0
        # Local 1 (r0,c1) -> Global (k=2*i+1, i=0) -> Index (2*i+1)*4 + 0
        # Local 2 (r1,c0) -> Global (k=2*i, i=1) -> Index (2*i)*4 + 1
        # Local 3 (r1,c1) -> Global (k=2*i+1, i=1) -> Index (2*i+1)*4 + 1
        # Local 4 (r2,c0) -> Global (k=2*i, i=2) -> Index (2*i)*4 + 2
        # Local 5 (r2,c1) -> Global (k=2*i+1, i=2) -> Index (2*i+1)*4 + 2
        # Local 6 (r3,c0) -> Global (k=2*i, i=3) -> Index (2*i)*4 + 3
        # Local 7 (r3,c1) -> Global (k=2*i+1, i=3) -> Index (2*i+1)*4 + 3
        
        for r in range(4):
            for c in range(2):
                local_idx = r * 2 + c
                global_k = 2*i + c
                global_i = r
                global_idx = global_k * 4 + global_i
                for k in range(28):
                    W_global[global_idx][k] = Wb[local_idx][k]
                    
        V_blocks.append(V_global)
        W_blocks.append(W_global)

    # Concatenate
    U = [row[:] for row in U_blocks[0]]
    for b in U_blocks[1:]:
        for i in range(16):
            for j in range(28):
                U[i][j + 28 * U_blocks.index(b)] = b[i][j]
                
    # This concatenation logic is messy. Let's do it properly.
    U_final = [[0]*84 for _ in range(16)]
    V_final = [[0]*84 for _ in range(24)]
    W_final = [[0]*84 for _ in range(24)]
    
    for i in range(3):
        offset = i * 28
        for r in range(16):
            for c in range(28):
                U_final[r][offset + c] = U_blocks[i][r][c]
        for r in range(24):
            for c in range(28):
                V_final[r][offset + c] = V_blocks[i][r][c]
                W_final[r][offset + c] = W_blocks[i][r][c]
                
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
