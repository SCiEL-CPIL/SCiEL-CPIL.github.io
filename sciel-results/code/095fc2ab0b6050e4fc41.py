import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff):
        nonlocal col
        for u_idx in u_indices:
            U[u_idx[0]][col] += coeff * u_idx[1]
        for v_idx in v_indices:
            V[v_idx[0]][col] += coeff * v_idx[1]
        for w_idx in w_indices:
            W[w_idx[0]][col] += coeff * w_idx[1]
        col += 1

    # Part 1: 4x4x4 Strassen
    # Blocks of A (4x4)
    # A00: rows 0,1 cols 0,1 -> U indices: (0,1), (1,1), (4,1), (5,1)
    # A01: rows 0,1 cols 2,3 -> U indices: (2,1), (3,1), (6,1), (7,1)
    # A10: rows 2,3 cols 0,1 -> U indices: (8,1), (9,1), (12,1), (13,1)
    # A11: rows 2,3 cols 2,3 -> U indices: (10,1), (11,1), (14,1), (15,1)
    
    A_blocks = {
        '00': [(0, 1), (1, 1), (4, 1), (5, 1)],
        '01': [(2, 1), (3, 1), (6, 1), (7, 1)],
        '10': [(8, 1), (9, 1), (12, 1), (13, 1)],
        '11': [(10, 1), (11, 1), (14, 1), (15, 1)]
    }
    
    # Blocks of B1 (4x4, cols 0-3)
    # B00: rows 0,1 cols 0,1 -> V indices: (0,1), (1,1), (7,1), (8,1)
    # B01: rows 0,1 cols 2,3 -> V indices: (2,1), (3,1), (9,1), (10,1)
    # B10: rows 2,3 cols 0,1 -> V indices: (14,1), (15,1), (21,1), (22,1)
    # B11: rows 2,3 cols 2,3 -> V indices: (16,1), (17,1), (23,1), (24,1)
    
    B_blocks = {
        '00': [(0, 1), (1, 1), (7, 1), (8, 1)],
        '01': [(2, 1), (3, 1), (9, 1), (10, 1)],
        '10': [(14, 1), (15, 1), (21, 1), (22, 1)],
        '11': [(16, 1), (17, 1), (23, 1), (24, 1)]
    }
    
    # Blocks of C1 (4x4, cols 0-3)
    # C00: rows 0,1 cols 0,1 -> W indices: (0,1), (1,1), (4,1), (5,1)
    # C01: rows 0,1 cols 2,3 -> W indices: (2,1), (3,1), (6,1), (7,1)
    # C10: rows 2,3 cols 0,1 -> W indices: (8,1), (9,1), (12,1), (13,1)
    # C11: rows 2,3 cols 2,3 -> W indices: (10,1), (11,1), (14,1), (15,1)
    
    C_blocks = {
        '00': [(0, 1), (1, 1), (4, 1), (5, 1)],
        '01': [(2, 1), (3, 1), (6, 1), (7, 1)],
        '10': [(8, 1), (9, 1), (12, 1), (13, 1)],
        '11': [(10, 1), (11, 1), (14, 1), (15, 1)]
    }
    
    # Strassen products for 4x4x4
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    def get_block_sum(b1, b2, sign1=1, sign2=1):
        return [(idx, coeff * sign) for block in [b1, b2] for idx, coeff in A_blocks[block] for sign in [sign1 if block == b1 else sign2]]
        # This logic is flawed, simpler to just concatenate lists with coeffs
        
    def combine_blocks(b1, c1, b2, c2):
        res = []
        for idx, coeff in A_blocks[b1]:
            res.append((idx, coeff * c1))
        for idx, coeff in A_blocks[b2]:
            res.append((idx, coeff * c2))
        return res
        
    def combine_blocks_B(b1, c1, b2, c2):
        res = []
        for idx, coeff in B_blocks[b1]:
            res.append((idx, coeff * c1))
        for idx, coeff in B_blocks[b2]:
            res.append((idx, coeff * c2))
        return res

    # Inner Strassen terms for 2x2x2
    # Returns list of (u_list, v_list, w_list, coeff)
    # u_list, v_list, w_list are lists of (index, coeff)
    
    def inner_strassen_terms(u_blocks, v_blocks):
        # u_blocks is a list of (block_name, coeff)
        # v_blocks is a list of (block_name, coeff)
        # We need to form the 7 products
        # But wait, the inner Strassen operates on the 2x2 matrices themselves.
        # The "X" matrix is formed by the linear combination of A blocks.
        # The "Y" matrix is formed by the linear combination of B blocks.
        # Then we apply the 7 Strassen terms to X and Y.
        
        # Let's construct X and Y as lists of (global_index, coeff)
        X = []
        for block_name, coeff in u_blocks:
            for idx, c in A_blocks[block_name]:
                X.append((idx, c * coeff))
                
        Y = []
        for block_name, coeff in v_blocks:
            for idx, c in B_blocks[block_name]:
                Y.append((idx, c * coeff))
                
        # Now apply 7 Strassen terms to X and Y
        # X entries: x00, x01, x10, x11
        # Y entries: y00, y01, y10, y11
        # We need to map the local 2x2 indices to the global indices in X and Y.
        # The order in A_blocks is (0,1), (1,1), (4,1), (5,1) which corresponds to x00, x01, x10, x11?
        # A00 block: rows 0,1 cols 0,1.
        # Entry (0,0) of A00 is global (0,0) -> U[0].
        # Entry (0,1) of A00 is global (0,1) -> U[1].
        # Entry (1,0) of A00 is global (1,0) -> U[4].
        # Entry (1,1) of A00 is global (1,1) -> U[5].
        # So the 4 entries of a 2x2 block are in order: (r0c0, r0c1, r1c0, r1c1).
        
        terms = []
        
        # P1: (x00 + x11)(y00 + y11)
        u_p1 = [X[0], X[3]]
        v_p1 = [Y[0], Y[3]]
        w_p1 = [C_blocks['00'][0], C_blocks['11'][0]] # C00(0,0) and C11(0,0)? No.
        # C00 is a 2x2 block. C00(0,0) is global C(0,0).
        # C11 is a 2x2 block. C11(0,0) is global C(2,2).
        # Wait, C_blocks['00'] contains indices for the whole 2x2 block.
        # We need to add to specific entries of C.
        # Z00 gets +1 from P1. Z00 is the top-left entry of the C block.
        # The C block is 2x2. Its entries are C(0,0), C(0,1), C(1,0), C(1,1) relative to block start.
        # For C00, global indices are 0,1,4,5. Top-left is 0.
        # For C11, global indices are 10,11,14,15. Top-left is 10.
        # So w_p1 should be [(0, 1), (10, 1)]
        
        # Let's refine W indices for each Z entry
        # Z00: C00[0] (global 0)
        # Z01: C00[1] (global 1)
        # Z10: C00[2] (global 4)
        # Z11: C00[3] (global 5)
        # This is for the C00 block.
        # For C11 block:
        # Z00: C11[0] (global 10)
        # Z01: C11[1] (global 11)
        # Z10: C11[2] (global 14)
        # Z11: C11[3] (global 15)
        
        # We need to know which C block is being updated.
        # The outer Strassen product M_l updates specific C blocks.
        # M1 updates C00 and C11.
        # M2 updates C10 and C11.
        # M3 updates C01 and C11.
        # M4 updates C00 and C10.
        # M5 updates C00 and C01.
        # M6 updates C11.
        # M7 updates C00.
        
        # This is getting complicated to hardcode.
        # Let's just write the 49 terms explicitly by iterating.
        
        pass

    # Simpler approach:
    # Define the 7 outer Strassen products' inputs (X, Y) and outputs (C blocks).
    # For each, call a function that adds 7 inner Strassen terms.
    
    outer_products = [
        # (X_blocks, Y_blocks, C_blocks_to_update)
        # X_blocks: list of (block_name, coeff)
        # Y_blocks: list of (block_name, coeff)
        # C_blocks: list of (block_name, coeff_dict) where coeff_dict maps Z_entry to coeff
    ]
    
    # M1 = (A00 + A11)(B00 + B11) -> C00 += P1, C11 += P1
    # Inner P1: X=x00+x11, Y=y00+y11. Z00+=1, Z11+=1.
    # So for M1, we add:
    # Term 1: U(A00+A11), V(B00+B11), W(C00[00]+=1, C11[00]+=1) -- Wait, P1 contributes to Z00 and Z11 of the RESULT of the 2x2x2.
    # The result of the 2x2x2 is added to C00 and C11.
    # So C00[00] += 1, C00[11] += 1, C11[00] += 1, C11[11] += 1.
    
    # This manual expansion is error-prone.
    # Let's use the structure.
    
    # I will implement the general Strassen decomposition for 4x4x4 by nesting the 2x2x2 decomposition.
    
    def add_strassen_2x2x2(U, V, W, X_indices, Y_indices, Z_indices, col_start):
        # X_indices: list of 4 (idx, coeff) for x00, x01, x10, x11
        # Y_indices: list of 4 (idx, coeff) for y00, y01, y10, y11
        # Z_indices: list of 4 (idx, coeff) for z00, z01, z10, z11
        # This function adds 7 terms starting at col_start.
        # Returns new col index.
        
        col = col_start
        
        # P1: x00+x11, y00+y11 -> z00+1, z11+1
        u1 = [X_indices[0], X_indices[3]]
        v1 = [Y_indices[0], Y_indices[3]]
        w1 = [(Z_indices[0][0], Z_indices[0][1]), (Z_indices[3][0], Z_indices[3][1])]
        for u in u1: U[u[0]][col] += u[1]
        for v in v1: V[v[0]][col] += v[1]
        for w in w1: W[w[0]][col] += w[1]
        col += 1
        
        # P2: x10+x11, y00 -> z10+1, z11-1
        u2 = [X_indices[2], X_indices[3]]
        v2 = [Y_indices[0]]
        w2 = [(Z_indices[2][0], Z_indices[2][1]), (Z_indices[3][0], -Z_indices[3][1])]
        for u in u2: U[u[0]][col] += u[1]
        for v in v2: V[v[0]][col] += v[1]
        for w in w2: W[w[0]][col] += w[1]
        col += 1
        
        # P3: x00, y01-y11 -> z01+1, z11+1
        u3 = [X_indices[0]]
        v3 = [(Y_indices[1][0], Y_indices[1][1]), (Y_indices[3][0], -Y_indices[3][1])]
        w3 = [(Z_indices[1][0], Z_indices[1][1]), (Z_indices[3][0], Z_indices[3][1])]
        for u in u3: U[u[0]][col] += u[1]
        for v in v3: V[v[0]][col] += v[1]
        for w in w3: W[w[0]][col] += w[1]
        col += 1
        
        # P4: x11, y10-y00 -> z00+1, z10+1
        u4 = [X_indices[3]]
        v4 = [(Y_indices[2][0], Y_indices[2][1]), (Y_indices[0][0], -Y_indices[0][1])]
        w4 = [(Z_indices[0][0], Z_indices[0][1]), (Z_indices[2][0], Z_indices[2][1])]
        for u in u4: U[u[0]][col] += u[1]
        for v in v4: V[v[0]][col] += v[1]
        for w in w4: W[w[0]][col] += w[1]
        col += 1
        
        # P5: x00+x01, y11 -> z00-1, z01+1
        u5 = [X_indices[0], X_indices[1]]
        v5 = [Y_indices[3]]
        w5 = [(Z_indices[0][0], -Z_indices[0][1]), (Z_indices[1][0], Z_indices[1][1])]
        for u in u5: U[u[0]][col] += u[1]
        for v in v5: V[v[0]][col] += v[1]
        for w in w5: W[w[0]][col] += w[1]
        col += 1
        
        # P6: x10-x00, y00+y01 -> z11+1
        u6 = [(X_indices[2][0], X_indices[2][1]), (X_indices[0][0], -X_indices[0][1])]
        v6 = [Y_indices[0], Y_indices[1]]
        w6 = [(Z_indices[3][0], Z_indices[3][1])]
        for u in u6: U[u[0]][col] += u[1]
        for v in v6: V[v[0]][col] += v[1]
        for w in w6: W[w[0]][col] += w[1]
        col += 1
        
        # P7: x01-x11, y10+y11 -> z00+1
        u7 = [(X_indices[1][0], X_indices[1][1]), (X_indices[3][0], -X_indices[3][1])]
        v7 = [Y_indices[2], Y_indices[3]]
        w7 = [(Z_indices[0][0], Z_indices[0][1])]
        for u in u7: U[u[0]][col] += u[1]
        for v in v7: V[v[0]][col] += v[1]
        for w in w7: W[w[0]][col] += w[1]
        col += 1
        
        return col

    # Now construct the 7 outer products
    # We need X_indices, Y_indices, Z_indices for each.
    # X_indices is formed from A_blocks.
    # Y_indices is formed from B_blocks.
    # Z_indices is formed from C_blocks.
    
    # M1: X=A00+A11, Y=B00+B11, Z=C00+C11
    # X: A00[0..3] + A11[0..3]
    X1 = [ (A_blocks['00'][i][0], A_blocks['00'][i][1] + A_blocks['11'][i][1]) for i in range(4) ]
    Y1 = [ (B_blocks['00'][i][0], B_blocks['00'][i][1] + B_blocks['11'][i][1]) for i in range(4) ]
    # Z: C00 and C11.
    # The result of the 2x2x2 is added to C00 and C11.
    # So Z_indices should represent a "virtual" 2x2 matrix that is added to both.
    # But add_strassen_2x2x2 adds to specific indices in Z_indices.
    # We need to pass Z_indices such that Z00 maps to C00[0] and C11[0]?
    # No, the function adds to Z_indices[k].
    # If we want to add to two different blocks, we need to modify the function or pass a list of targets.
    
    # Let's modify add_strassen_2x2x2 to accept a list of Z targets.
    # Or simpler: call it twice? No, the U and V parts are the same.
    # We can just pass Z_indices as a list of 4 lists of (idx, coeff).
    
    def add_strassen_2x2x2_multi(U, V, W, X_indices, Y_indices, Z_targets, col_start):
        # Z_targets: list of 4 items. Each item is a list of (idx, coeff).
        col = col_start
        
        # P1
        u1 = [X_indices[0], X_indices[3]]
        v1 = [Y_indices[0], Y_indices[3]]
        w1 = []
        for target in Z_targets[0]: w1.append((target[0], target[1]))
        for target in Z_targets[3]: w1.append((target[0], target[1]))
        for u in u1: U[u[0]][col] += u[1]
        for v in v1: V[v[0]][col] += v[1]
        for w in w1: W[w[0]][col] += w[1]
        col += 1
        
        # P2
        u2 = [X_indices[2], X_indices[3]]
        v2 = [Y_indices[0]]
        w2 = []
        for target in Z_targets[2]: w2.append((target[0], target[1]))
        for target in Z_targets[3]: w2.append((target[0], -target[1]))
        for u in u2: U[u[0]][col] += u[1]
        for v in v2: V[v[0]][col] += v[1]
        for w in w2: W[w[0]][col] += w[1]
        col += 1
        
        # P3
        u3 = [X_indices[0]]
        v3 = [(Y_indices[1][0], Y_indices[1][1]), (Y_indices[3][0], -Y_indices[3][1])]
        w3 = []
        for target in Z_targets[1]: w3.append((target[0], target[1]))
        for target in Z_targets[3]: w3.append((target[0], target[1]))
        for u in u3: U[u[0]][col] += u[1]
        for v in v3: V[v[0]][col] += v[1]
        for w in w3: W[w[0]][col] += w[1]
        col += 1
        
        # P4
        u4 = [X_indices[3]]
        v4 = [(Y_indices[2][0], Y_indices[2][1]), (Y_indices[0][0], -Y_indices[0][1])]
        w4 = []
        for target in Z_targets[0]: w4.append((target[0], target[1]))
        for target in Z_targets[2]: w4.append((target[0], target[1]))
        for u in u4: U[u[0]][col] += u[1]
        for v in v4: V[v[0]][col] += v[1]
        for w in w4: W[w[0]][col] += w[1]
        col += 1
        
        # P5
        u5 = [X_indices[0], X_indices[1]]
        v5 = [Y_indices[3]]
        w5 = []
        for target in Z_targets[0]: w5.append((target[0], -target[1]))
        for target in Z_targets[1]: w5.append((target[0], target[1]))
        for u in u5: U[u[0]][col] += u[1]
        for v in v5: V[v[0]][col] += v[1]
        for w in w5: W[w[0]][col] += w[1]
        col += 1
        
        # P6
        u6 = [(X_indices[2][0], X_indices[2][1]), (X_indices[0][0], -X_indices[0][1])]
        v6 = [Y_indices[0], Y_indices[1]]
        w6 = []
        for target in Z_targets[3]: w6.append((target[0], target[1]))
        for u in u6: U[u[0]][col] += u[1]
        for v in v6: V[v[0]][col] += v[1]
        for w in w6: W[w[0]][col] += w[1]
        col += 1
        
        # P7
        u7 = [(X_indices[1][0], X_indices[1][1]), (X_indices[3][0], -X_indices[3][1])]
        v7 = [Y_indices[2], Y_indices[3]]
        w7 = []
        for target in Z_targets[0]: w7.append((target[0], target[1]))
        for u in u7: U[u[0]][col] += u[1]
        for v in v7: V[v[0]][col] += v[1]
        for w in w7: W[w[0]][col] += w[1]
        col += 1
        
        return col

    # Define Z targets for each outer product
    # C00 block indices: 0,1,4,5
    # C01 block indices: 2,3,6,7
    # C10 block indices: 8,9,12,13
    # C11 block indices: 10,11,14,15
    
    def get_Z_targets(blocks):
        # blocks is a list of block names
        targets = [[], [], [], []]
        for block in blocks:
            idxs = C_blocks[block]
            targets[0].append((idxs[0][0], idxs[0][1]))
            targets[1].append((idxs[1][0], idxs[1][1]))
            targets[2].append((idxs[2][0], idxs[2][1]))
            targets[3].append((idxs[3][0], idxs[3][1]))
        return targets

    # M1
    col = add_strassen_2x2x2_multi(U, V, W, X1, Y1, get_Z_targets(['00', '11']), col)
    
    # M2: X=A10+A11, Y=B00, Z=C10+C11
    X2 = [ (A_blocks['10'][i][0], A_blocks['10'][i][1] + A_blocks['11'][i][1]) for i in range(4) ]
    Y2 = B_blocks['00']
    col = add_strassen_2x2x2_multi(U, V, W, X2, Y2, get_Z_targets(['10', '11']), col)
    
    # M3: X=A00, Y=B01-B11, Z=C01+C11
    X3 = A_blocks['00']
    Y3 = [ (B_blocks['01'][i][0], B_blocks['01'][i][1] - B_blocks['11'][i][1]) for i in range(4) ]
    col = add_strassen_2x2x2_multi(U, V, W, X3, Y3, get_Z_targets(['01', '11']), col)
    
    # M4: X=A11, Y=B10-B00, Z=C00+C10
    X4 = A_blocks['11']
    Y4 = [ (B_blocks['10'][i][0], B_blocks['10'][i][1] - B_blocks['00'][i][1]) for i in range(4) ]
    col = add_strassen_2x2x2_multi(U, V, W, X4, Y4, get_Z_targets(['00', '10']), col)
    
    # M5: X=A00+A01, Y=B11, Z=C00+C01
    X5 = [ (A_blocks['00'][i][0], A_blocks['00'][i][1] + A_blocks['01'][i][1]) for i in range(4) ]
    Y5 = B_blocks['11']
    col = add_strassen_2x2x2_multi(U, V, W, X5, Y5, get_Z_targets(['00', '01']), col)
    
    # M6: X=A10-A00, Y=B00+B01, Z=C11
    X6 = [ (A_blocks['10'][i][0], A_blocks['10'][i][1] - A_blocks['00'][i][1]) for i in range(4) ]
    Y6 = [ (B_blocks['00'][i][0], B_blocks['00'][i][1] + B_blocks['01'][i][1]) for i in range(4) ]
    col = add_strassen_2x2x2_multi(U, V, W, X6, Y6, get_Z_targets(['11']), col)
    
    # M7: X=A01-A11, Y=B10+B11, Z=C00
    X7 = [ (A_blocks['01'][i][0], A_blocks['01'][i][1] - A_blocks['11'][i][1]) for i in range(4) ]
    Y7 = [ (B_blocks['10'][i][0], B_blocks['10'][i][1] + B_blocks['11'][i][1]) for i in range(4) ]
    col = add_strassen_2x2x2_multi(U, V, W, X7, Y7, get_Z_targets(['00']), col)
    
    # Part 2: 4x4x3 Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[4*i + j][col] += 1
                V[7*j + k][col] += 1
                W[4*k + i][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
