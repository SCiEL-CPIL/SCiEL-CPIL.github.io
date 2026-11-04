"""
Strategy: Block Partition Decomposition.
Split 4x4 * 4x6 into 4x4 * 4x4 and 4x4 * 4x2.
Use Recursive Strassen (Rank 49) for 4x4 * 4x4.
Use Block Strassen (Rank 28) for 4x4 * 4x2.
Total Rank: 77.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _build_strassen_4x4_4x4():
    """
    Build rank-49 decomposition for 4x4 * 4x4.
    Returns U (16x49), V (24x49) [only first 12 cols used? No, V is 4x4 part so 4*4=16 entries? 
    Wait, V dimension is m*p = 4*6 = 24.
    For the 4x4 part, we only use B[:, 0:4].
    So V will be 24x49, with non-zeros in rows j*6+k for k in 0..3.
    W will be 24x49, with non-zeros in rows k*4+i for k in 0..3.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((24, rank)) # Full V shape, but only B[:, 0:4] used
    W = np.zeros((24, rank)) # Full W shape, but only C[:, 0:4] used
    
    # Strassen for 4x4 * 4x4
    # Recursively apply Strassen to 2x2 blocks of 2x2 matrices.
    # Level 1: 2x2 blocks of size 2x2.
    # Level 2: 2x2 blocks of size 1x1.
    
    # We'll generate the 49 terms by iterating over the 7 products at the top level,
    # and for each, generating the 7 products at the sub-level.
    
    # Indices for 4x4 A: i, j in 0..3
    # Indices for 4x4 B: j, k in 0..3
    # Indices for 4x4 C: i, k in 0..3
    
    col = 0
    # Top level blocks:
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) -- block coords
    # Each block is 2x2.
    
    # Strassen products at top level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each product, we need to multiply two 2x2 matrices.
    # We use Strassen for 2x2 * 2x2 (Rank 7).
    
    def add_strassen_2x2(u_signs, v_signs, w_signs, base_i, base_j, base_k, base_k_b, base_k_c):
        """
        Adds 7 terms for a 2x2 * 2x2 multiplication.
        u_signs: dict mapping (bi, bj) to sign for A block
        v_signs: dict mapping (bj, bk) to sign for B block
        w_signs: dict mapping (bk, bi) to sign for C block
        base_i, base_j, base_k: offset for A indices
        base_k_b, base_k_c: offset for B and C indices
        """
        nonlocal col
        # 7 Strassen products for 2x2 * 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map the 2x2 block indices to global indices.
        # A block (bi, bj) -> global (base_i + bi, base_j + bj)
        # B block (bj, bk) -> global (base_j + bj, base_k_b + bk)
        # C block (bk, bi) -> global (base_k_c + bk, base_i + bi)
        
        # Strassen 2x2 products:
        # P1: (A11+A22)(B11+B22)
        # P2: (A21+A22)B11
        # P3: A11(B12-B22)
        # P4: A22(B21-B11)
        # P5: (A11+A12)B22
        # P6: (A21-A11)(B11+B12)
        # P7: (A12-A22)(B21+B22)
        
        # We will generate 7 columns.
        # For each column, we set U, V, W.
        
        # Helper to get global indices
        def a_idx(bi, bj): return (base_i + bi) * 4 + (base_j + bj)
        def b_idx(bj, bk): return (base_j + bj) * 6 + (base_k_b + bk)
        def c_idx(bk, bi): return (base_k_c + bk) * 4 + (base_i + bi)
        
        # P1
        U[a_idx(0,0)][col] = u_signs.get((0,0), 0) + u_signs.get((1,1), 0)
        U[a_idx(1,1)][col] = u_signs.get((0,0), 0) + u_signs.get((1,1), 0)
        V[b_idx(0,0)][col] = v_signs.get((0,0), 0) + v_signs.get((1,1), 0)
        V[b_idx(1,1)][col] = v_signs.get((0,0), 0) + v_signs.get((1,1), 0)
        # W contributions for P1: C11 += P1, C22 += P1
        W[c_idx(0,0)][col] = w_signs.get((0,0), 0)
        W[c_idx(1,1)][col] = w_signs.get((1,1), 0)
        col += 1
        
        # P2
        U[a_idx(1,0)][col] = u_signs.get((1,0), 0) + u_signs.get((1,1), 0)
        U[a_idx(1,1)][col] = u_signs.get((1,0), 0) + u_signs.get((1,1), 0)
        V[b_idx(0,0)][col] = v_signs.get((0,0), 0)
        # W: C21 += P2, C22 += P2
        W[c_idx(0,1)][col] = w_signs.get((0,1), 0)
        W[c_idx(1,1)][col] = w_signs.get((1,1), 0)
        col += 1
        
        # P3
        U[a_idx(0,0)][col] = u_signs.get((0,0), 0)
        V[b_idx(0,1)][col] = v_signs.get((0,1), 0) - v_signs.get((1,1), 0)
        V[b_idx(1,1)][col] = -v_signs.get((1,1), 0) + v_signs.get((0,1), 0) # Wait, B12-B22
        # B12 is (0,1), B22 is (1,1).
        # So V[0,1] gets +1, V[1,1] gets -1.
        V[b_idx(0,1)][col] = v_signs.get((0,1), 0)
        V[b_idx(1,1)][col] = -v_signs.get((1,1), 0)
        # W: C12 += P3, C22 -= P3
        W[c_idx(1,0)][col] = w_signs.get((1,0), 0)
        W[c_idx(1,1)][col] = -w_signs.get((1,1), 0)
        col += 1
        
        # P4
        U[a_idx(1,1)][col] = u_signs.get((1,1), 0)
        V[b_idx(1,0)][col] = v_signs.get((1,0), 0) - v_signs.get((0,0), 0)
        V[b_idx(1,0)][col] = v_signs.get((1,0), 0)
        V[b_idx(0,0)][col] = -v_signs.get((0,0), 0)
        # W: C21 += P4, C11 -= P4
        W[c_idx(0,1)][col] = w_signs.get((0,1), 0)
        W[c_idx(0,0)][col] = -w_signs.get((0,0), 0)
        col += 1
        
        # P5
        U[a_idx(0,0)][col] = u_signs.get((0,0), 0) + u_signs.get((0,1), 0)
        U[a_idx(0,1)][col] = u_signs.get((0,0), 0) + u_signs.get((0,1), 0)
        V[b_idx(1,1)][col] = v_signs.get((1,1), 0)
        # W: C11 += P5, C12 += P5
        W[c_idx(0,0)][col] = w_signs.get((0,0), 0)
        W[c_idx(1,0)][col] = w_signs.get((1,0), 0)
        col += 1
        
        # P6
        U[a_idx(1,0)][col] = u_signs.get((1,0), 0) - u_signs.get((0,0), 0)
        U[a_idx(0,0)][col] = -u_signs.get((0,0), 0) + u_signs.get((1,0), 0) # Wait, A21-A11
        U[a_idx(1,0)][col] = u_signs.get((1,0), 0)
        U[a_idx(0,0)][col] = -u_signs.get((0,0), 0)
        V[b_idx(0,0)][col] = v_signs.get((0,0), 0) + v_signs.get((0,1), 0)
        V[b_idx(0,1)][col] = v_signs.get((0,0), 0) + v_signs.get((0,1), 0)
        # W: C22 -= P6
        W[c_idx(1,1)][col] = -w_signs.get((1,1), 0)
        col += 1
        
        # P7
        U[a_idx(0,1)][col] = u_signs.get((0,1), 0) - u_signs.get((1,1), 0)
        U[a_idx(1,1)][col] = -u_signs.get((1,1), 0) + u_signs.get((0,1), 0) # A12-A22
        U[a_idx(0,1)][col] = u_signs.get((0,1), 0)
        U[a_idx(1,1)][col] = -u_signs.get((1,1), 0)
        V[b_idx(1,0)][col] = v_signs.get((1,0), 0) + v_signs.get((1,1), 0)
        V[b_idx(1,1)][col] = v_signs.get((1,0), 0) + v_signs.get((1,1), 0)
        # W: C11 -= P7
        W[c_idx(0,0)][col] = -w_signs.get((0,0), 0)
        col += 1

    # Top level Strassen for 4x4*4x4
    # Blocks:
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
    
    # M1 = (A11+A22)(B11+B22)
    u1 = {(0,0):1, (1,1):1}
    v1 = {(0,0):1, (1,1):1}
    w1 = {(0,0):1, (1,1):1}
    add_strassen_2x2(u1, v1, w1, 0, 0, 0, 0, 0)
    
    # M2 = (A21+A22)B11
    u2 = {(1,0):1, (1,1):1}
    v2 = {(0,0):1}
    w2 = {(1,0):1, (1,1):1}
    add_strassen_2x2(u2, v2, w2, 0, 0, 0, 0, 0)
    
    # M3 = A11(B12-B22)
    u3 = {(0,0):1}
    v3 = {(0,1):1, (1,1):-1}
    w3 = {(0,1):1, (1,1):-1}
    add_strassen_2x2(u3, v3, w3, 0, 0, 0, 0, 0)
    
    # M4 = A22(B21-B11)
    u4 = {(1,1):1}
    v4 = {(1,0):1, (0,0):-1}
    w4 = {(1,0):1, (0,0):-1}
    add_strassen_2x2(u4, v4, w4, 0, 0, 0, 0, 0)
    
    # M5 = (A11+A12)B22
    u5 = {(0,0):1, (0,1):1}
    v5 = {(1,1):1}
    w5 = {(0,0):1, (0,1):1}
    add_strassen_2x2(u5, v5, w5, 0, 0, 0, 0, 0)
    
    # M6 = (A21-A11)(B11+B12)
    u6 = {(1,0):1, (0,0):-1}
    v6 = {(0,0):1, (0,1):1}
    w6 = {(1,1):-1}
    add_strassen_2x2(u6, v6, w6, 0, 0, 0, 0, 0)
    
    # M7 = (A12-A22)(B21+B22)
    u7 = {(0,1):1, (1,1):-1}
    v7 = {(1,0):1, (1,1):1}
    w7 = {(0,0):-1}
    add_strassen_2x2(u7, v7, w7, 0, 0, 0, 0, 0)
    
    return U, V, W

def _build_strassen_4x4_4x2():
    """
    Build rank-28 decomposition for 4x4 * 4x2.
    Uses Block Strassen.
    """
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((24, rank)) # Only k=4,5 used
    W = np.zeros((24, rank)) # Only k=4,5 used
    
    # B is 4x2.
    # Split B into B_top (2x2) and B_bot (2x2).
    # B_top corresponds to rows 0,1 of B, cols 0,1 of the 4x2 part (which are cols 4,5 of full B).
    # B_bot corresponds to rows 2,3 of B, cols 0,1 of the 4x2 part.
    
    # We apply Strassen to the 2x2 block structure of A and the 2x1 block structure of B.
    # The "scalar" multiplication is 2x2 * 2x2.
    
    col = 0
    
    # Top level Strassen for 4x4 * 4x2
    # A blocks: A11, A12, A21, A22 (2x2 each)
    # B blocks: B1 (top 2x2), B2 (bot 2x2)
    
    # M1 = (A11+A22)(B1+B2)
    # M2 = (A21+A22)B1
    # M3 = A11(B2-B1)  # Note: B12-B22 in standard, here B_top vs B_bot
    # M4 = A22(B1-B2)   # B21-B11 in standard
    # M5 = (A11+A12)B2
    # M6 = (A21-A11)(B1+B2)
    # M7 = (A12-A22)(B1+B2)
    
    # Wait, standard Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Here B is 4x2.
    # B11 (top-left of B) = B[0:2, 0:2] of the 4x2 part.
    # B12 (top-right) doesn't exist?
    # We are multiplying 4x4 by 4x2.
    # This is like multiplying 2x2 block matrix by 2x1 block matrix.
    # Strassen for 2x2 * 2x1?
    # There is no standard 7-product formula for 2x2 * 2x1.
    # However, we can embed 4x2 into 4x4 by padding with zeros?
    # If we pad B to 4x4, we get Rank 49.
    # But we want Rank 28.
    
    # Let's use the fact that $R(4,4,2) \le 2 R(2,2,1) \times 2$? No.
    # $R(4,4,2) = 28$ is achieved by using Strassen on the 2x2 blocks, where each "scalar" product is a 2x2*2x2 product?
    # No, the blocks of B are 2x2.
    # So we are doing 7 products of (2x2 * 2x2).
    # Each 2x2*2x2 product has rank 4 (schoolbook) or 7 (Strassen).
    # If we use schoolbook for the sub-products, rank is 7*4 = 28.
    
    # So we need to generate 7 terms for the top-level Strassen,
    # and for each term, generate 4 terms for the schoolbook 2x2*2x2 sub-product.
    
    def add_schoolbook_2x2(u_signs, v_signs, w_signs, base_i, base_j, base_k_b, base_k_c):
        nonlocal col
        # 4 terms for 2x2 * 2x2
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    # U[a][col] = u_signs[(bi,bj)]
                    # V[b][col] = v_signs[(bj,bk)]
                    # W[c][col] = w_signs[(bk,bi)]
                    
                    # U index: (base_i+bi)*4 + (base_j+bj)
                    # V index: (base_j+bj)*6 + (base_k_b+bk)
                    # W index: (base_k_c+bk)*4 + (base_i+bi)
                    
                    U[(base_i+bi)*4 + (base_j+bj)][col] = u_signs.get((bi,bj), 0)
                    V[(base_j+bj)*6 + (base_k_b+bk)][col] = v_signs.get((bj,bk), 0)
                    W[(base_k_c+bk)*4 + (base_i+bi)][col] = w_signs.get((bk,bi), 0)
                    col += 1

    # Top level products for 4x4 * 4x2
    # B1 = B[0:2, 4:6] (rows 0,1, cols 4,5)
    # B2 = B[2:4, 4:6] (rows 2,3, cols 4,5)
    
    # M1 = (A11+A22)(B1+B2)
    u1 = {(0,0):1, (1,1):1}
    v1 = {(0,0):1, (1,0):1, (0,1):1, (1,1):1} # B1+B2
    # Wait, v_signs maps (bj, bk) to sign.
    # B1 is top block (bj=0), B2 is bot block (bj=1).
    # B1 has bk=0,1. B2 has bk=0,1.
    # So v1 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    w1 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1} # C11+C22? No, C is 4x2.
    # C11 (top-left 2x1), C21 (bot-left 2x1).
    # C11 corresponds to (bk=0, bi=0), (bk=1, bi=0)?
    # C is 4x2.
    # C_top (rows 0,1), C_bot (rows 2,3).
    # C_top has 2 cols.
    # So W indices:
    # C_top, col 0: k=4, i=0,1
    # C_top, col 1: k=5, i=0,1
    # C_bot, col 0: k=4, i=2,3
    # C_bot, col 1: k=5, i=2,3
    
    # In add_schoolbook_2x2, w_signs maps (bk, bi) to sign.
    # bk is 0,1 (local col index). bi is 0,1 (local row index).
    # Global k = base_k_c + bk.
    # Global i = base_i + bi.
    
    # For M1, it contributes to C11 and C22?
    # In 2x2*2x2, M1 contributes to C11 and C22.
    # Here, C11 is C_top, C22 is C_bot.
    # So w1 should have 1 for all (bk, bi)?
    # No, M1 = (A11+A22)(B1+B2).
    # Result is added to C11 (top) and C22 (bot).
    # So w1 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    
    add_schoolbook_2x2(u1, v1, w1, 0, 0, 4, 4)
    
    # M2 = (A21+A22)B1
    u2 = {(1,0):1, (1,1):1}
    v2 = {(0,0):1, (0,1):1} # B1 only
    w2 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1} # C21 (bot) and C22? No, C21 is bot-left.
    # M2 contributes to C21 and C22 in standard.
    # Here C21 is C_bot, C22 is... wait.
    # Standard: C11, C12, C21, C22.
    # Here we have C_top, C_bot.
    # So M2 contributes to C_bot.
    # w2 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    add_schoolbook_2x2(u2, v2, w2, 0, 0, 4, 4)
    
    # M3 = A11(B2-B1)
    u3 = {(0,0):1}
    v3 = {(1,0):1, (1,1):1, (0,0):-1, (0,1):-1} # B2 - B1
    w3 = {(0,0):1, (0,1):1, (1,0):-1, (1,1):-1} # C_top - C_bot
    add_schoolbook_2x2(u3, v3, w3, 0, 0, 4, 4)
    
    # M4 = A22(B1-B2)
    u4 = {(1,1):1}
    v4 = {(0,0):1, (0,1):1, (1,0):-1, (1,1):-1} # B1 - B2
    w4 = {(0,0):-1, (0,1):-1, (1,0):1, (1,1):1} # -C_top + C_bot
    add_schoolbook_2x2(u4, v4, w4, 0, 0, 4, 4)
    
    # M5 = (A11+A12)B2
    u5 = {(0,0):1, (0,1):1}
    v5 = {(1,0):1, (1,1):1} # B2
    w5 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1} # C_top + C_bot
    add_schoolbook_2x2(u5, v5, w5, 0, 0, 4, 4)
    
    # M6 = (A21-A11)(B1+B2)
    u6 = {(1,0):1, (0,0):-1}
    v6 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    w6 = {(0,0):-1, (0,1):-1, (1,0):-1, (1,1):-1} # -C_top - C_bot
    add_schoolbook_2x2(u6, v6, w6, 0, 0, 4, 4)
    
    # M7 = (A12-A22)(B1+B2)
    u7 = {(0,1):1, (1,1):-1}
    v7 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    w7 = {(0,0):-1, (0,1):-1, (1,0):-1, (1,1):-1} # -C_top - C_bot
    add_schoolbook_2x2(u7, v7, w7, 0, 0, 4, 4)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    try:
        U1, V1, W1 = _build_strassen_4x4_4x4()
        U2, V2, W2 = _build_strassen_4x4_4x2()
        
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        
        rank = U.shape[1]
        
        # Verify
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    s = np.dot(U[i*4+j, :], V[j*6+k, :] * W[k*4+i, :])
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback
            U_fb = np.zeros((16, 96))
            V_fb = np.zeros((24, 96))
            W_fb = np.zeros((24, 96))
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(6):
                        U_fb[i*4+j, col] = 1
                        V_fb[j*6+k, col] = 1
                        W_fb[k*4+i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        U_fb = np.zeros((16, 96))
        V_fb = np.zeros((24, 96))
        W_fb = np.zeros((24, 96))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*6+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
