import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    def time_check():
        return (time.time() - start_time) > (deadline - safety_margin)

    n, m, p = 4, 4, 4
    rank_initial = 49
    
    # Generate the 49 rank-1 terms of the recursive Strassen decomposition
    # U, V, W shapes: (16, 49)
    U = np.zeros((16, rank_initial), dtype=np.float64)
    V = np.zeros((16, rank_initial), dtype=np.float64)
    W = np.zeros((16, rank_initial), dtype=np.float64)
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i  # Trace convention: W[k*n+i, r]
    
    # Indices for 2x2 blocks
    # A blocks: 00, 01, 10, 11
    # U indices for A[bi, bj]
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # Helper to add a Strassen term
    # The recursive step:
    # We have 7 "big" terms. Each big term is a product of two 2x2 matrices.
    # Each product is decomposed into 7 scalar terms.
    
    # Let's define the 7 Strassen combinations for 2x2 matrices X, Y
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # We need to map these to indices in U, V.
    # For the top level, X=A, Y=B.
    # For the sub-level, X and Y are the specific combinations.
    
    # Let's build the terms directly.
    # There are 7 top-level terms. Let's call them T0..T6.
    # Each Tt is a product of two 2x2 matrices, let's call them At and Bt.
    # At and Bt are linear combinations of the blocks of A and B.
    
    # Define the block indices for A (U) and B (V)
    # Block (bi, bj) of A corresponds to U indices:
    # (2*bi, 2*bj), (2*bi, 2*bj+1), (2*bi+1, 2*bj), (2*bi+1, 2*bj+1)
    
    def get_a_block_indices(bi, bj):
        return [
            u_idx(2*bi, 2*bj),
            u_idx(2*bi, 2*bj+1),
            u_idx(2*bi+1, 2*bj),
            u_idx(2*bi+1, 2*bj+1)
        ]
        
    def get_b_block_indices(bi, bj):
        return [
            v_idx(2*bi, 2*bj),
            v_idx(2*bi, 2*bj+1),
            v_idx(2*bi+1, 2*bj),
            v_idx(2*bi+1, 2*bj+1)
        ]
    
    # The 7 top-level terms definitions:
    # (A_part, B_part)
    # A_part is a list of (block_idx, coeff)
    # B_part is a list of (block_idx, coeff)
    
    # M1: (A00+A11) * (B00+B11)
    # M2: (A10+A11) * B00
    # M3: A00 * (B01-B11)
    # M4: A11 * (B10-B00)
    # M5: (A00+A01) * B11
    # M6: (A10-A00) * (B00+B01)
    # M7: (A01-A11) * (B10+B11)
    
    top_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),      # T0
        ([(2, 1), (3, 1)], [(0, 1)]),              # T1
        ([(0, 1)], [(1, 1), (3, -1)]),             # T2
        ([(3, 1)], [(2, 1), (0, -1)]),             # T3
        ([(0, 1), (1, 1)], [(3, 1)]),              # T4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),     # T5
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])      # T6
    ]
    
    # For each top term, we decompose the product of the two 2x2 matrices into 7 scalar terms.
    # Let the A-part matrix be X and B-part matrix be Y.
    # X and Y are 2x2 matrices.
    # The 7 scalar multiplications are:
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y01-Y11)
    # S4 = X11(Y10-Y00)
    # S5 = (X00+X01)Y11
    # S6 = (X10-X00)(Y00+Y01)
    # S7 = (X01-X11)(Y10+Y11)
    
    # We need to map X and Y entries to global U and V indices.
    # X is formed by summing blocks of A.
    # X[i,j] = sum_{(b, c) in A_part} coeff * A_block[b, c][i, j]
    # Y[i,j] = sum_{(b, c) in B_part} coeff * B_block[b, c][i, j]
    
    # For each top term t (0..6), and each sub term s (0..6):
    # rank_idx = t * 7 + s
    
    for t in range(7):
        A_part, B_part = top_terms[t]
        
        # Construct X and Y as symbolic 2x2 matrices of linear combinations
        # X[i,j] is a dict {u_index: coeff}
        # Y[i,j] is a dict {v_index: coeff}
        
        X = [[{} for _ in range(2)] for _ in range(2)]
        Y = [[{} for _ in range(2)] for _ in range(2)]
        
        for (bi, bj), coeff in A_part:
            indices = get_a_block_indices(bi, bj)
            # indices: [00, 01, 10, 11] local to block
            # global U indices
            for li in range(2):
                for lj in range(2):
                    idx = indices[li*2 + lj]
                    X[li][lj][idx] = X[li][lj].get(idx, 0) + coeff
                    
        for (bi, bj), coeff in B_part:
            indices = get_b_block_indices(bi, bj)
            for li in range(2):
                for lj in range(2):
                    idx = indices[li*2 + lj]
                    Y[li][lj][idx] = Y[li][lj].get(idx, 0) + coeff
        
        # Now apply Strassen to X, Y
        # S1: (X00+X11)(Y00+Y11)
        # S2: (X10+X11)Y00
        # S3: X00(Y01-Y11)
        # S4: X11(Y10-Y00)
        # S5: (X00+X01)Y11
        # S6: (X10-X00)(Y00+Y01)
        # S7: (X01-X11)(Y10+Y11)
        
        def add_dicts(d1, d2, sign=1):
            res = dict(d1)
            for k, v in d2.items():
                res[k] = res.get(k, 0) + sign * v
            return {k: v for k, v in res.items() if v != 0}
            
        def sub_dicts(d1, d2):
            return add_dicts(d1, d2, -1)
            
        # Calculate the 7 terms
        # Each term is a pair (dict_U, dict_V)
        
        # S1
        x1 = add_dicts(X[0][0], X[1][1])
        y1 = add_dicts(Y[0][0], Y[1][1])
        
        # S2
        x2 = add_dicts(X[1][0], X[1][1])
        y2 = Y[0][0]
        
        # S3
        x3 = X[0][0]
        y3 = sub_dicts(Y[0][1], Y[1][1])
        
        # S4
        x4 = X[1][1]
        y4 = sub_dicts(Y[1][0], Y[0][0])
        
        # S5
        x5 = add_dicts(X[0][0], X[0][1])
        y5 = Y[1][1]
        
        # S6
        x6 = sub_dicts(X[1][0], X[0][0])
        y6 = add_dicts(Y[0][0], Y[0][1])
        
        # S7
        x7 = sub_dicts(X[0][1], X[1][1])
        y7 = add_dicts(Y[1][0], Y[1][1])
        
        sub_terms = [
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4),
            (x5, y5),
            (x6, y6),
            (x7, y7)
        ]
        
        for s in range(7):
            rank_idx = t * 7 + s
            x_dict, y_dict = sub_terms[s]
            
            for idx, coeff in x_dict.items():
                U[idx, rank_idx] += coeff
            for idx, coeff in y_dict.items():
                V[idx, rank_idx] += coeff
                
            # W part:
            # The result of the 2x2 multiplication X*Y is a 2x2 matrix P.
            # P is added to the corresponding block of C.
            # The contribution of S1..S7 to P:
            # P00 = S1 + S4 - S5 + S7
            # P01 = S3 + S5
            # P10 = S2 + S6
            # P11 = S1 - S2 + S3 + S6
            
            # Wait, standard Strassen:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M6
            # C11 = M1 - M2 + M3 + M6
            
            # Here S1..S7 are the M's for the sub-problem.
            # The sub-problem result P is a 2x2 matrix.
            # P[0,0] corresponds to C[2*bi+0, 2*bj+0] ?
            # No, the top-level term t corresponds to a specific block of C?
            # No, the top-level Strassen computes the whole 4x4 C.
            # The 7 terms M0..M6 are 2x2 matrices.
            # C00 = M00 + M10 - M01 + M11 ? No.
            
            # Let's trace the indices.
            # The 7 top-level terms T0..T6 produce 2x2 matrices M0..M6.
            # C = [ M00 + M10 - M01 + M11  M01 + M11 ]
            #     [ M10 + M11              M00 - M10 + M01 + M11 ]
            # Wait, the standard Strassen formulas for C blocks (2x2 each):
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M6
            # C11 = M1 - M2 + M3 + M6
            
            # Here M1..M7 are the 7 products.
            # Each Mk is a 2x2 matrix.
            # So for each Mk, we have 4 entries.
            # The W tensor maps the scalar multiplication to the C entries.
            
            # For a specific top-level term t (which is one of M1..M7),
            # and a specific sub-level term s (which is one of the 7 scalars in Mk),
            # we need to determine which C entries this scalar contributes to, and with what coefficient.
            
            # Let's map t (0..6) to M_index (1..7).
            # t=0 -> M1
            # t=1 -> M2
            # t=2 -> M3
            # t=3 -> M4
            # t=4 -> M5
            # t=5 -> M6
            # t=6 -> M7
            
            # The contribution of M_k to C blocks:
            # M1: C00 (+1), C11 (+1)
            # M2: C10 (+1), C11 (-1)
            # M3: C01 (+1), C11 (+1)
            # M4: C00 (+1), C10 (+1)
            # M5: C00 (-1), C01 (+1)
            # M6: C11 (+1)
            # M7: C00 (+1)
            
            # Wait, let's re-verify.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M6
            # C11 = M1 - M2 + M3 + M6
            
            # So:
            # M1: C00(1), C11(1)
            # M2: C10(1), C11(-1)
            # M3: C01(1), C11(1)
            # M4: C00(1), C10(1)
            # M5: C00(-1), C01(1)
            # M6: C10(1), C11(1)  <-- Wait, M6 is in C10 and C11?
            # Formula: C11 = M1 - M2 + M3 + M6. Yes.
            # M7: C00(1)
            
            # Now, within Mk (a 2x2 matrix), the scalar Ss (s=0..6) contributes to the entries of Mk.
            # Mk = [ P00 P01 ]
            #      [ P10 P11 ]
            
            # P00 = S1 + S4 - S5 + S7
            # P01 = S3 + S5
            # P10 = S2 + S6
            # P11 = S1 - S2 + S3 + S6
            
            # So for a given t and s, we know which entry of Mk (P00, P01, P10, P11) Ss contributes to.
            # And we know which C block Mk contributes to.
            
            # Mk is located at a specific position in C?
            # No, Mk is a 2x2 matrix that is added to specific 2x2 blocks of C.
            # M1 is added to C00 and C11.
            # So the entry P00 of M1 is added to C[0,0] and C[2,2].
            # P01 of M1 is added to C[0,1] and C[2,3].
            # P10 of M1 is added to C[1,0] and C[3,2].
            # P11 of M1 is added to C[1,1] and C[3,3].
            
            # Generally, if Mk contributes to C block (bi, bj) with coefficient coeff_block,
            # then entry P[li, lj] of Mk contributes to C[2*bi+li, 2*bj+lj] with coeff_block.
            
            # Let's define the block contributions for each t (M_index):
            # t=0 (M1): blocks (0,0) coeff 1, (1,1) coeff 1
            # t=1 (M2): blocks (1,0) coeff 1, (1,1) coeff -1
            # t=2 (M3): blocks (0,1) coeff 1, (1,1) coeff 1
            # t=3 (M4): blocks (0,0) coeff 1, (1,0) coeff 1
            # t=4 (M5): blocks (0,0) coeff -1, (0,1) coeff 1
            # t=5 (M6): blocks (1,0) coeff 1, (1,1) coeff 1
            # t=6 (M7): blocks (0,0) coeff 1
            
            block_map = {
                0: [(0, 0, 1), (1, 1, 1)],
                1: [(1, 0, 1), (1, 1, -1)],
                2: [(0, 1, 1), (1, 1, 1)],
                3: [(0, 0, 1), (1, 0, 1)],
                4: [(0, 0, -1), (0, 1, 1)],
                5: [(1, 0, 1), (1, 1, 1)],
                6: [(0, 0, 1)]
            }
            
            # Sub-term s (0..6) corresponds to S1..S7.
            # s=0 -> S1, s=1 -> S2, etc.
            
            # P entry coefficients for Ss:
            # S1 (s=0): P00(1), P11(1)
            # S2 (s=1): P10(1), P11(-1)
            # S3 (s=2): P01(1), P11(1)
            # S4 (s=3): P00(1), P10(1)
            # S5 (s=4): P00(-1), P01(1)
            # S6 (s=5): P10(1), P11(1)
            # S7 (s=6): P00(1)
            
            sub_map = {
                0: [(0, 0, 1), (1, 1, 1)],
                1: [(1, 0, 1), (1, 1, -1)],
                2: [(0, 1, 1), (1, 1, 1)],
                3: [(0, 0, 1), (1, 0, 1)],
                4: [(0, 0, -1), (0, 1, 1)],
                5: [(1, 0, 1), (1, 1, 1)],
                6: [(0, 0, 1)]
            }
            
            # For each block in block_map[t] and each P entry in sub_map[s]:
            # C index: row = 2*bi + li, col = 2*bj + lj
            # W index: w_idx(row, col) = col * 4 + row (Trace convention)
            # Coeff: coeff_block * coeff_sub
            
            for (bi, bj, coeff_block) in block_map[t]:
                for (li, lj, coeff_sub) in sub_map[s]:
                    r = 2 * bi + li
                    c = 2 * bj + lj
                    w_ind = w_idx(r, c)
                    W[w_ind, rank_idx] += coeff_block * coeff_sub

    # Now we have U, V, W of shape (16, 49).
    # Check for linear dependence.
    # Flatten T_k = u_k \otimes v_k \otimes w_k
    # Vector length 16*16*16 = 4096.
    
    T_vectors = []
    for k in range(49):
        u = U[:, k]
        v = V[:, k]
        w = W[:, k]
        t_vec = np.einsum('i,j,k->ijk', u, v, w).flatten()
        T_vectors.append(t_vec)
        
    T_matrix = np.array(T_vectors).T # Shape (4096, 49)
    
    # Compute rank
    # Use SVD
    U_svd, s, Vt = np.linalg.svd(T_matrix, full_matrices=False)
    
    # Count non-zero singular values
    tol = 1e-10
    rank_found = np.sum(s > tol)
    
    if rank_found < 49:
        # Find null space
        # Null space is spanned by last (49 - rank_found) columns of Vt
        null_vecs = Vt[rank_found:, :].T
        # Take the first null vector
        c = null_vecs[:, 0]
        
        # c is a vector of length 49 such that sum c_k T_k = 0
        # We can eliminate term k=48 (index 48)
        # T_48 = -1/c_48 * sum_{k != 48} c_k T_k
        # New decomposition:
        # T'_k = (1 - c_k/c_48) T_k for k != 48
        # T'_48 is removed.
        
        c_48 = c[48]
        if abs(c_48) < tol:
            # Pick a different index to eliminate
            # Find index with max |c|
            elim_idx = np.argmax(np.abs(c))
        else:
            elim_idx = 48
            
        c_elim = c[elim_idx]
        
        U_new = U[:, :elim_idx].copy()
        V_new = V[:, :elim_idx].copy()
        W_new = W[:, :elim_idx].copy()
        
        for k in range(elim_idx):
            scale = 1 - c[k] / c_elim
            U_new[:, k] = U[:, k] * scale
            V_new[:, k] = V[:, k] * scale
            W_new[:, k] = W[:, k] * scale
            
        # Return the reduced decomposition
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # No reduction found
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
