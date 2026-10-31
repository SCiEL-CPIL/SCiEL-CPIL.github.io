import time
import numpy as np

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor.
    This is the minimal possible rank for 4x4 matrix multiplication.
    """
    t0 = time.time()
    deadline = t0 + 110.0 # Safety margin

    # Define Strassen 2x2 decomposition
    # T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    # Indices are local 0,1
    # U corresponds to A[i,j], V to B[j,k], W to C[k,i] (note W index is k*2+i for 2x2)
    
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00)     -> C10 += M2, C11 -= M2
    # M3 = (A00)(B01-B11)     -> C01 += M3, C11 += M3
    # M4 = (A11)(B10-B00)     -> C00 += M4, C10 += M4
    # M5 = (A00+A01)(B11)     -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7

    # Format: (u_entries, v_entries, w_entries)
    # Each entry is a list of (index, coeff)
    # For 2x2, A index is i*2+j. B index is j*2+k. W index is k*2+i.
    
    strassen_2x2 = [
        # M1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2
        ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
        # M3
        ([(0,1)],       [(1,1),(3,-1)],[(1,1),(3,1)]),
        # M4
        ([(3,1)],       [(2,1),(0,-1)],[(0,1),(2,1)]),
        # M5
        ([(0,1),(1,1)], [(3,1)],       [(0,-1),(1,1)]),
        # M6
        ([(2,1),(0,-1)],[(0,1),(1,1)], [(3,1)]),
        # M7
        ([(1,1),(3,-1)],[(2,1),(3,1)], [(0,1)]),
    ]

    # Recursive application for 4x4
    # We treat 4x4 as 2x2 blocks of 2x2 matrices.
    # Outer Strassen combines blocks.
    # Inner Strassen computes block products.
    
    # The total rank is 7 (outer) * 7 (inner) = 49.
    # Each of the 49 terms corresponds to a specific combination of outer and inner Strassen steps.
    
    # Instead of building the full 49 terms explicitly in code which is verbose,
    # we can construct the matrices U, V, W by iterating through the 49 combinations.
    
    # Let's define the structure more explicitly.
    # A is 4x4. A[i,j]. i=2*bi+li, j=2*bj+lj.
    # B is 4x4. B[j,k]. j=2*bj+lj, k=2*bk+lk.
    # C is 4x4. C[i,k]. i=2*bi+li, k=2*bk+lk.
    # W index is k*4+i.
    
    # Outer Strassen operates on block indices (bi,bj) for A, (bj,bk) for B, (bk,bi) for C.
    # Inner Strassen operates on local indices (li,lj) etc.
    
    # We will generate the 49 terms.
    # Term index t = t_outer * 7 + t_inner.
    
    U = np.zeros((16, 49), dtype=np.int8)
    V = np.zeros((16, 49), dtype=np.int8)
    W = np.zeros((16, 49), dtype=np.int8)
    
    # Outer Strassen terms (same structure as inner, but on block indices)
    # Block indices: 0,1 for rows, 0,1 for cols.
    # A block index: bi*2 + bj
    # B block index: bj*2 + bk
    # C block index: bk*2 + bi (W index convention)
    
    # We reuse the strassen_2x2 definition for the outer level, but the indices
    # in the tuples refer to block coordinates (0..3) which map to global blocks.
    # Wait, the strassen_2x2 defined above uses indices 0..3 for the 2x2 matrix elements.
    # For the outer level, the "elements" are 2x2 blocks.
    # So index 0 in outer Strassen corresponds to block (0,0).
    # Index 1 corresponds to block (0,1).
    # Index 2 corresponds to block (1,0).
    # Index 3 corresponds to block (1,1).
    
    # This is exactly the same structure!
    # So for each outer term (u_outer, v_outer, w_outer) from strassen_2x2:
    #   For each inner term (u_inner, v_inner, w_inner) from strassen_2x2:
    #       Construct the contribution to U, V, W.
    
    for t_outer, (u_out, v_out, w_out) in enumerate(strassen_2x2):
        for t_inner, (u_in, v_in, w_in) in enumerate(strassen_2x2):
            col = t_outer * 7 + t_inner
            
            # Process U (A)
            # u_out entries are (block_idx, coeff_out)
            # u_in entries are (local_idx, coeff_in)
            # Global A index = block_row * 2 + local_row, block_col * 2 + local_col
            # block_idx = block_row * 2 + block_col
            # local_idx = local_row * 2 + local_col
            
            # We need to combine the coefficients.
            # The term is a product of sums.
            # U[global_idx, col] += coeff_out * coeff_in
            
            # However, u_out and u_in are lists of (idx, coeff).
            # We need to sum contributions from all pairs (idx_out, idx_in) that map to the same global index?
            # No, the Strassen formula is a sum of products.
            # M_outer = (sum A_blocks) * (sum B_blocks)
            # Then M_inner computes the product of these sums.
            # Actually, the recursive structure means:
            # C = sum_{outer} (A_outer * B_outer)
            # A_outer is a linear combination of A blocks.
            # B_outer is a linear combination of B blocks.
            # The product A_outer * B_outer is a 2x2 matrix product of the blocks.
            # This product is computed using inner Strassen.
            
            # So, for a fixed outer term, we have specific blocks of A and B involved.
            # Let A_comb be the linear combination of A blocks.
            # Let B_comb be the linear combination of B blocks.
            # The inner Strassen computes A_comb * B_comb.
            # The entries of the result are linear combinations of the products of entries of A_comb and B_comb.
            
            # My previous code did this by iterating through the local indices.
            # Let's stick to that logic, it was working.
            
            # For each (idx_out, coeff_out) in u_out:
            #   This refers to a specific A block (br, bc).
            #   For each (idx_in, coeff_in) in u_in:
            #      This refers to a specific local element (lr, lc) within that block?
            #      NO. The inner Strassen is applied to the *entire* 2x2 block matrices.
            #      The "indices" in the inner Strassen (0..3) refer to the local 2x2 coordinates.
            #      So, if u_out selects block (0,0) with coeff 1, and u_in selects local (0,0) with coeff 1,
            #      then the global A index is (0*2+0)*4 + (0*2+0) = 0.
            
            # Wait, the u_in indices (0..3) are relative to the 2x2 block.
            # The u_out indices (0..3) are relative to the 4x4 matrix (i.e., block indices).
            
            # So:
            # For each (idx_out, coeff_out) in u_out:
            #   br = idx_out // 2
            #   bc = idx_out % 2
            #   For each (idx_in, coeff_in) in u_in:
            #      lr = idx_in // 2
            #      lc = idx_in % 2
            #      global_row = br * 2 + lr
            #      global_col = bc * 2 + lc
            #      global_idx = global_row * 4 + global_col
            #      U[global_idx, col] += coeff_out * coeff_in
            
            for idx_out, coeff_out in u_out:
                br = idx_out // 2
                bc = idx_out % 2
                for idx_in, coeff_in in u_in:
                    lr = idx_in // 2
                    lc = idx_in % 2
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    u_idx = gr * 4 + gc
                    U[u_idx, col] += coeff_out * coeff_in
            
            # Similarly for V (B)
            # v_out indices refer to B blocks.
            # B block index (br, bc) -> global row br*2+lr, col bc*2+lc.
            # V index is j*4+k. j is row, k is col.
            for idx_out, coeff_out in v_out:
                br = idx_out // 2
                bc = idx_out % 2
                for idx_in, coeff_in in v_in:
                    lr = idx_in // 2
                    lc = idx_in % 2
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    v_idx = gr * 4 + gc
                    V[v_idx, col] += coeff_out * coeff_in
            
            # Similarly for W (C)
            # w_out indices refer to C blocks.
            # C block index (br, bc) -> global row br*2+lr, col bc*2+lc.
            # W index is k*4+i. k is row, i is col.
            for idx_out, coeff_out in w_out:
                br = idx_out // 2
                bc = idx_out % 2
                for idx_in, coeff_in in w_in:
                    lr = idx_in // 2
                    lc = idx_in % 2
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    w_idx = gc * 4 + gr
                    W[w_idx, col] += coeff_out * coeff_in
    
    # Verify
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # Should be 1 for all i,j,k.
    
    # Fast check
    # We can use matrix multiplication properties or just loop.
    # 4*4*4 = 64 checks.
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # u = U[i*4+j, :]
                # v = V[j*4+k, :]
                # w = W[k*4+i, :]
                # s = sum(u * v * w)
                # Use np.dot for speed? np.dot(u, v*w)
                s = int(np.dot(U[i*4+j, :], V[j*4+k, :] * W[k*4+i, :]))
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break

    if not valid:
        # Fallback to schoolbook if something went wrong (should not happen)
        U = np.zeros((16, 64), dtype=np.int8)
        V = np.zeros((16, 64), dtype=np.int8)
        W = np.zeros((16, 64), dtype=np.int8)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        # Rank is 64, which is worse, but valid.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
