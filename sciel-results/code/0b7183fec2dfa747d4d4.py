import time
import numpy as np

def _strassen_2x2_u(block_indices_u, block_indices_v):
    """
    For a 2x2x2 Strassen term, given which A-blocks and B-blocks are involved,
    return the coefficient vector for a 4x4 matrix (16 entries).
    block_indices_u: list of (row_block, col_block) for A
    block_indices_v: list of (row_block, col_block) for B
    Each entry in the lists has a sign (+1 or -1).
    """
    vec = [0] * 16
    for (rb, cb), sign in block_indices_u:
        vec[rb * 2 + cb] = sign  # This is wrong for 4x4
    return vec

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = 4, 4, 7
    
    # Strategy: 4x4x4 (Strassen rank 49) + 4x4x3 (schoolbook rank 48) = rank 97
    # 
    # For 4x4x4: view A, B, C as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen at block level (rank 7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product, which via Strassen needs rank 7.
    # So total rank = 7 * 7 = 49.
    #
    # For the full 4x4x7:
    # U: 16 x 97 (n*m = 16, rank = 97)
    # V: 28 x 97 (m*p = 28, rank = 97)  
    # W: 28 x 97 (p*n = 28, rank = 97)
    
    # Construct the 49-term decomposition for 4x4x4
    # Each term: u (16 entries for A), v (16 entries for B), w (16 entries for C)
    # A[i,j] at index i*4+j, B[j,k] at index j*4+k, C[i,k] at index i*4+k
    # But W uses trace convention: W[k*n+i] for C[i,k], so C[i,k] is at index k*4+i in W
    
    # Strassen 2x2 (inner) terms: for 2x2 matrices X,Y,Z with Z=X*Y
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7, Z01 = M3+M5, Z10 = M2+M4, Z11 = M1-M2+M3+M6
    
    def strassen_2x2_terms():
        """Return 7 terms for 2x2x2 Strassen. Each term is (u4, v4, w4) for 2x2 matrices."""
        # 2x2 matrix: X[i,j] at index i*2+j
        # u: coefficients for X (4 entries), v: for Y (4 entries), w: for Z (4 entries)
        terms = []
        # M1 = (X00+X11)(Y00+Y11)
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))  # Z00+Z11... wait
        # Actually Z00 = M1+M4-M5+M7, so M1 contributes +1 to Z00
        # Z11 = M1-M2+M3+M6, so M1 contributes +1 to Z11
        # So w for M1: Z00=+1, Z11=+1 => [1,0,0,1]
        # Let me redo this properly.
        
        # Term 1: M1 = (X00+X11)(Y00+Y11)
        # Contributes to: Z00 (+1), Z11 (+1)
        # u: X00=+1, X11=+1 => [1,0,0,1]
        # v: Y00=+1, Y11=+1 => [1,0,0,1]
        # w: Z00=+1, Z11=+1 => [1,0,0,1]
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # Term 2: M2 = (X10+X11)Y00
        # Contributes to: Z10 (+1), Z11 (-1)
        # u: X10=+1, X11=+1 => [0,0,1,1]
        # v: Y00=+1 => [1,0,0,0]
        # w: Z10=+1, Z11=-1 => [0,0,1,-1]
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        
        # Term 3: M3 = X00(Y01-Y11)
        # Contributes to: Z01 (+1), Z11 (+1)
        # u: X00=+1 => [1,0,0,0]
        # v: Y01=+1, Y11=-1 => [0,1,0,-1]
        # w: Z01=+1, Z11=+1 => [0,1,0,1]
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
        
        # Term 4: M4 = X11(Y10-Y00)
        # Contributes to: Z00 (+1), Z10 (+1)
        # u: X11=+1 => [0,0,0,1]
        # v: Y10=+1, Y00=-1 => [-1,0,1,0]
        # w: Z00=+1, Z10=+1 => [1,0,1,0]
        terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
        
        # Term 5: M5 = (X00+X01)Y11
        # Contributes to: Z00 (-1), Z01 (+1)
        # u: X00=+1, X01=+1 => [1,1,0,0]
        # v: Y11=+1 => [0,0,0,1]
        # w: Z00=-1, Z01=+1 => [-1,1,0,0]
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        
        # Term 6: M6 = (X10-X00)(Y00+Y01)
        # Contributes to: Z11 (+1)
        # u: X10=+1, X00=-1 => [-1,0,1,0]
        # v: Y00=+1, Y01=+1 => [1,1,0,0]
        # w: Z11=+1 => [0,0,0,1]
        terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        
        # Term 7: M7 = (X01-X11)(Y10+Y11)
        # Contributes to: Z00 (+1)
        # u: X01=+1, X11=-1 => [0,1,0,-1]
        # v: Y10=+1, Y11=+1 => [0,0,1,1]
        # w: Z00=+1 => [1,0,0,0]
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        
        return terms
    
    # Now build the 49-term 4x4x4 decomposition
    # Top-level Strassen blocks:
    # A00, A01, A10, A11 are 2x2 blocks
    # A00: rows 0-1, cols 0-1 => indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 => indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 => indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 => indices 10,11,14,15
    #
    # B00: rows 0-1, cols 0-1 => indices 0,1,4,5
    # B01: rows 0-1, cols 2-3 => indices 2,3,6,7
    # B10: rows 2-3, cols 0-1 => indices 8,9,12,13
    # B11: rows 2-3, cols 2-3 => indices 10,11,14,15
    #
    # C00: rows 0-1, cols 0-1 => indices 0,1,4,5
    # C01: rows 0-1, cols 2-3 => indices 2,3,6,7
    # C10: rows 2-3, cols 0-1 => indices 8,9,12,13
    # C11: rows 2-3, cols 2-3 => indices 10,11,14,15
    
    # Define block index mappings for 4x4
    def block_indices_4x4(block_row, block_col):
        """Return the 4 indices in a 4x4 matrix (row-major) for a 2x2 block at (block_row, block_col)."""
        # block_row: 0 or 1, block_col: 0 or 1
        base_row = block_row * 2
        base_col = block_col * 2
        return [base_row * 4 + base_col, base_row * 4 + base_col + 1,
                (base_row + 1) * 4 + base_col, (base_row + 1) * 4 + base_col + 1]
    
    # For 2x2 matrix X with indices [0,1,2,3] = [X00, X01, X10, X11]
    # Map to 4x4 block at (br, bc): 
    # X00 -> 4x4[br*4+bc], X01 -> 4x4[br*4+bc+1], X10 -> 4x4[(br+1)*4+bc], X11 -> 4x4[(br+1)*4+bc+1]
    
    def expand_2x2_to_4x4(vec2, block_row, block_col):
        """Expand a 2x2 coefficient vector (4 entries) to 4x4 (16 entries) for the specified block."""
        vec4 = [0] * 16
        base_r = block_row * 2
        base_c = block_col * 2
        # vec2[0]=X00, vec2[1]=X01, vec2[2]=X10, vec2[3]=X11
        vec4[base_r * 4 + base_c] = vec2[0]
        vec4[base_r * 4 + base_c + 1] = vec2[1]
        vec4[(base_r + 1) * 4 + base_c] = vec2[2]
        vec4[(base_r + 1) * 4 + base_c + 1] = vec2[3]
        return vec4
    
    def combine_blocks_4x4(vecs):
        """Combine multiple 16-entry vectors by addition."""
        result = [0] * 16
        for v in vecs:
            for i in range(16):
                result[i] += v[i]
        return result
    
    # Top-level Strassen for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each top-level term Mi, we need to compute the 2x2x2 Strassen of the block products.
    # Each Mi involves linear combinations of A-blocks and B-blocks, resulting in a 2x2 matrix product.
    
    # Define the top-level terms as (a_blocks, b_blocks) where a_blocks is a list of (block, sign) for A
    top_level_terms = [
        # (a_block_signs, b_block_signs)
        # M1 = (A00+A11)(B00+B11)
        ([(0,0,+1), (1,1,+1)], [(0,0,+1), (1,1,+1)]),
        # M2 = (A10+A11)B00
        ([(1,0,+1), (1,1,+1)], [(0,0,+1)]),
        # M3 = A00(B01-B11)
        ([(0,0,+1)], [(0,1,+1), (1,1,-1)]),
        # M4 = A11(B10-B00)
        ([(1,1,+1)], [(1,0,+1), (0,0,-1)]),
        # M5 = (A00+A01)B11
        ([(0,0,+1), (0,1,+1)], [(1,1,+1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1,0,+1), (0,0,-1)], [(0,0,+1), (0,1,+1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0,1,+1), (1,1,-1)], [(1,0,+1), (1,1,+1)]),
    ]
    
    # C block contributions from each Mi:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_contribs = [
        [(0,0,+1), (1,0,-1), (1,1,-1)],  # Wait, this is wrong
        # Let me redo: C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
    ]
    
    # For each top-level term Mi and each inner Strassen term j (j=0..6),
    # we get a full 4x4x4 term.
    
    inner_strassen = strassen_2x2_terms()
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    for t_idx, (a_signs, b_signs) in enumerate(top_level_terms):
        # Compute the A-block combination: X = sum of a_block * A_block
        # X is a 2x2 matrix (in block space)
        # Similarly Y = sum of b_block * B_block
        
        # For the inner Strassen, X plays the role of the "A" matrix and Y plays the "B" matrix.
        # Each inner Strassen term j gives (u4, v4, w4) for 2x2 matrices.
        
        for j, (uj, vj, wj) in enumerate(inner_strassen):
            # uj, vj, wj are 4-entry vectors for 2x2 matrices
            # We need to expand these to 4x4.
            
            # uj corresponds to coefficients for X (the combined A-blocks)
            # X is a 2x2 matrix where X[i,j] = sum of signs * A_block[bi,bj] for (bi,bj,sign) in a_signs
            # So the coefficient for A[bi,bj] in the full 4x4 is uj[i,j] * sign for each (bi,bj,sign) in a_signs
            
            u_full = [0] * 16
            for (bi, bj, sign) in a_signs:
                # uj[i,j] gives the coefficient for X[i,j]
                # X[i,j] = sum over a_signs: sign * A_block[bi,bj]
                # So A_block[bi,bj] gets coefficient uj[i,j] * sign
                for ii in range(2):
                    for jj in range(2):
                        coeff = uj[ii * 2 + jj] * sign
                        if coeff != 0:
                            # A_block[bi,bj] is at 4x4 position (bi*2+ii, bj*2+jj)
                            idx = (bi * 2 + ii) * 4 + (bj * 2 + jj)
                            u_full[idx] += coeff
            
            v_full = [0] * 16
            for (bi, bj, sign) in b_signs:
                for ii in range(2):
                    for jj in range(2):
                        coeff = vj[ii * 2 + jj] * sign
                        if coeff != 0:
                            idx = (bi * 2 + ii) * 4 + (bj * 2 + jj)
                            v_full[idx] += coeff
            
            # wj corresponds to coefficients for Z (the result block)
            # Z is a 2x2 matrix that contributes to C blocks based on t_idx
            # C_block_contribs[t_idx] gives which C-blocks Mi contributes to
            c_block_map = {
                0: [(0,0,+1), (1,1,+1)],   # M1 -> C00(+1), C11(+1)
                1: [(1,0,+1), (1,1,-1)],   # M2 -> C10(+1), C11(-1)
                2: [(0,1,+1), (1,1,+1)],   # M3 -> C01(+1), C11(+1)
                3: [(0,0,+1), (1,0,+1)],   # M4 -> C00(+1), C10(+1)
                4: [(0,0,-1), (0,1,+1)],   # M5 -> C00(-1), C01(+1)
                5: [(1,1,+1)],             # M6 -> C11(+1)
                6: [(0,0,+1)],             # M7 -> C00(+1)
            }
            
            w_full = [0] * 16
            for (ci, cj, sign) in c_block_map[t_idx]:
                for ii in range(2):
                    for jj in range(2):
                        coeff = wj[ii * 2 + jj] * sign
                        if coeff != 0:
                            idx = (ci * 2 + ii) * 4 + (cj * 2 + jj)
                            w_full[idx] += coeff
            
            U_444.append(u_full)
            V_444.append(v_full)
            W_444.append(w_full)
    
    # Now verify the 4x4x4 decomposition
    U_444_arr = np.array(U_444, dtype=float)
    V_444_arr = np.array(V_444, dtype=float)
    W_444_arr = np.array(W_444, dtype=float)
    rank_444 = len(U_444)
    
    valid_444 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for c in range(rank_444):
                    s += U_444_arr[i*4+j, c] * V_444_arr[j*4+k, c] * W_444_arr[k*4+i, c]
                if abs(s - 1.0) > 1e-10:
                    valid_444 = False
                    break
            if not valid_444:
                break
        if not valid_444:
            break
    
    if not valid_444:
        # Fallback: schoolbook for 4x4x4
        U_444 = []
        V_444 = []
        W_444 = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = [0] * 16
                    v = [0] * 16
                    w = [0] * 16
                    u[i*4+j] = 1
                    v[j*4+k] = 1
                    w[k*4+i] = 1
                    U_444.append(u)
                    V_444.append(v)
                    W_444.append(w)
        rank_444 = len(U_444)
    
    if time.time() > deadline:
        # Just return schoolbook for the whole thing
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Schoolbook for 4x4x3 (columns 4-6)
    U_443 = []
    V_443 = []
    W_443 = []
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = [0] * 16
                v = [0] * 12
                w = [0] * 12
                u[i*4+j] = 1
                v[j*3+k] = 1
                w[k*4+i] = 1
                U_443.append(u)
                V_443.append(v)
                W_443.append(w)
    rank_443 = len(U_443)
    
    # Combine for full 4x4x7
    total_rank = rank_444 + rank_443
    
    full_U = []
    for i in range(16):
        full_U.append(U_444[i] + U_443[i])
    
    full_V = []
    for j in range(4):
        for k in range(7):
            row = [0] * total_rank
            if k < 4:
                for c in range(rank_444):
                    row[c] = V_444[j*4+k][c]
            else:
                for c in range(rank_443):
                    row[rank_444 + c] = V_443[j*3 + (k-4)][c]
            full_V.append(row)
    
    full_W = []
    for k in range(7):
        for i in range(4):
            row = [0] * total_rank
            if k < 4:
                for c in range(rank_444):
                    row[c] = W_444[k*4+i][c]
            else:
                for c in range(rank_443):
                    row[rank_444 + c] = W_443[(k-4)*4+i][c]
            full_W.append(row)
    
    # Final verification
    if time.time() < deadline:
        full_U_arr = np.array(full_U, dtype=float)
        full_V_arr = np.array(full_V, dtype=float)
        full_W_arr = np.array(full_W, dtype=float)
        
        valid_full = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    s = 0.0
                    for c in range(total_rank):
                        s += full_U_arr[i*4+j, c] * full_V_arr[j*7+k, c] * full_W_arr[k*4+i, c]
                    if abs(s - 1.0) > 1e-10:
                        valid_full = False
                        break
                if not valid_full:
                    break
            if not valid_full:
                break
        
        if not valid_full:
            # Fallback to full schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": full_U, "V": full_V, "W": full_W}
