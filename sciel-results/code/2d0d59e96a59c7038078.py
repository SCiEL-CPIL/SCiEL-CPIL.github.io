"""Matrix multiplication 5x5x5 - Block decomposition with recursive Strassen."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    Returns list of (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    with local row-major indices 0..3.
    """
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append(([0,3],[1,1],[0,3],[1,1],[0,3],[1,1]))
    # M2 = (a10+a11)b00
    terms.append(([2,3],[1,1],[0],[1],[2,3],[1,-1]))
    # M3 = a00(b01-b11)
    terms.append(([0],[1],[1,3],[1,-1],[1,3],[1,1]))
    # M4 = a11(b10-b00)
    terms.append(([3],[1],[2,0],[1,-1],[0,2],[1,1]))
    # M5 = (a00+a01)b11
    terms.append(([0,1],[1,1],[3],[1],[0,1],[-1,1]))
    # M6 = (a10-a00)(b00+b01)
    terms.append(([2,0],[1,-1],[0,1],[1,1],[3],[1]))
    # M7 = (a01-a11)(b10+b11)
    terms.append(([1,3],[1,-1],[2,3],[1,1],[0],[1]))
    return terms

def _strassen_4x4():
    """
    Return 49-term decomposition for 4x4 matrix multiplication using
    two levels of Strassen.
    Returns list of (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    with local row-major indices 0..15 for A, B, C.
    """
    s2 = _strassen_2x2()
    result = []
    for outer_idx, (ou_idx, ou_c, ov_idx, ov_c, ow_idx, ow_c) in enumerate(s2):
        # Each outer term maps 2x2 sub-blocks to scalar products
        # We need to expand each 2x2 sub-block using inner Strassen
        # ou_idx contains local 2x2 indices (0..3) for the A sub-block
        # For a 4x4 matrix, the 2x2 blocks are at positions:
        #   block (0,0): rows 0-1, cols 0-1 -> local idx 0-3
        #   block (0,1): rows 0-1, cols 2-3 -> local idx 4-7
        #   block (1,0): rows 2-3, cols 0-1 -> local idx 8-11
        #   block (1,1): rows 2-3, cols 2-3 -> local idx 12-15
        
        # Map outer 2x2 index to 4x4 block position
        def _map_2to4(local_2x2_idx, block_row, block_col):
            # local_2x2_idx is 0..3 in the 2x2 sub-block
            r = local_2x2_idx // 2
            c = local_2x2_idx % 2
            return (block_row * 2 + r) * 4 + (block_col * 2 + c)
        
        # For each outer term, the U part has indices in {0,1,2,3} representing
        # the 4 sub-blocks of the 4x4 A matrix
        # We need to expand each sub-block reference into 4 individual elements
        # using the inner Strassen decomposition
        
        # Actually, the correct approach: the outer Strassen operates on 2x2 blocks.
        # Each "element" in the outer 2x2 is itself a 2x2 matrix.
        # The outer term says: product of (sum of certain 2x2 blocks from A) 
        # and (sum of certain 2x2 blocks from B), contributing to certain 2x2 blocks of C.
        # Each such product of 2x2 matrices needs 7 inner Strassen products.
        
        # For the outer term with U indices ou_idx and coeffs ou_c:
        # This means we compute sum_{alpha in ou_idx} ou_c[alpha] * A_block[alpha]
        # where A_block[alpha] is a 2x2 matrix.
        # Similarly for V.
        # The product is a 2x2 matrix, which we then place into C blocks.
        
        # To decompose: for each pair (u_block, v_block) that appears,
        # we need the full 2x2*2x2->2x2 Strassen decomposition.
        # But the coefficients combine linearly.
        
        # Better approach: enumerate all combinations of U-blocks and V-blocks
        # and for each, apply the 2x2 Strassen decomposition with the combined coefficient.
        
        u_blocks = {}
        for idx, coeff in zip(ou_idx, ou_c):
            u_blocks[idx] = u_blocks.get(idx, 0) + coeff
        
        v_blocks = {}
        for idx, coeff in zip(ov_idx, ov_c):
            v_blocks[idx] = v_blocks.get(idx, 0) + coeff
        
        w_blocks = {}
        for idx, coeff in zip(ow_idx, ow_c):
            w_blocks[idx] = w_blocks.get(idx, 0) + coeff
        
        # For each pair (ub, vb), compute ub*A[ub] * vb*B[vb]
        # This is a 2x2 * 2x2 product with coefficient ub*vb
        # Apply inner Strassen to get 7 terms
        for ub, uc in u_blocks.items():
            for vb, vc in v_blocks.items():
                combined_coeff = uc * vc
                if combined_coeff == 0:
                    continue
                for inner_term in s2:
                    iu_idx, iu_c, iv_idx, iv_c, iw_idx, iw_c = inner_term
                    # Map inner 2x2 indices to 4x4 indices
                    # ub determines which 2x2 block of A: ub=0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    ub_row, ub_col = ub // 2, ub % 2
                    vb_row, vb_col = vb // 2, vb % 2
                    
                    u_4x4 = {}
                    for li, lc in zip(iu_idx, iu_c):
                        g4 = _map_2to4(li, ub_row, ub_col)
                        u_4x4[g4] = u_4x4.get(g4, 0) + combined_coeff * lc
                    
                    v_4x4 = {}
                    for li, lc in zip(iv_idx, iv_c):
                        g4 = _map_2to4(li, vb_row, vb_col)
                        v_4x4[g4] = v_4x4.get(g4, 0) + combined_coeff * lc
                    
                    # W: the result goes into C blocks specified by w_blocks
                    # iw_idx gives which 2x2 block of the 2x2 result
                    # w_blocks gives which 4x4 C block to place it in
                    # iw_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    for iw, iw_c in zip(iw_idx, iw_c):
                        for wb, wc in w_blocks.items():
                            if iw_c == 0 or wc == 0:
                                continue
                            # The inner result 2x2 block iw goes into C block wb
                            # But we need to check: the outer w tells us which C sub-block
                            # the outer product contributes to. The inner w tells us
                            # which element within that 2x2 sub-block.
                            # So C block = wb, and within that block, position = iw
                            wb_row, wb_col = wb // 2, wb % 2
                            g4_w = _map_2to4(iw, wb_row, wb_col)
                            w_4x4 = {}
                            w_4x4[g4_w] = w_4x4.get(g4_w, 0) + combined_coeff * iw_c * wc
                    
                    if u_4x4 and v_4x4 and w_4x4:
                        # Convert to lists
                        u_list = sorted(u_4x4.keys())
                        u_c_list = [u_4x4[k] for k in u_list]
                        v_list = sorted(v_4x4.keys())
                        v_c_list = [v_4x4[k] for k in v_list]
                        w_list = sorted(w_4x4.keys())
                        w_c_list = [w_4x4[k] for k in w_list]
                        result.append((u_list, u_c_list, v_list, v_c_list, w_list, w_c_list))
    return result


def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact using random test matrices."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U_arr[i*m+j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for j in range(m):
                v_vals += V_arr[j*p+k, :] * B[j, k]
            C_decomp[i, k] = np.dot(W_arr[k*n+i, :], u_vals * v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Block decomposition: split 5 into 4+1
    # Use 4x4x4 Strassen for the top-left, schoolbook for the rest
    row_splits = [0, 4, 5]
    col_splits = [0, 4, 5]
    
    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Get 4x4x4 decomposition
    strassen_4 = _strassen_4x4()
    
    # Get 2x2 decomposition for fallback
    strassen_2 = _strassen_2x2()
    
    for i in range(2):  # A row block
        for j in range(2):  # A col / B row block
            for k in range(2):  # B col / C col block
                ha = row_splits[i+1] - row_splits[i]
                wa = col_splits[j+1] - col_splits[j]
                hb = row_splits[j+1] - row_splits[j]
                wb = col_splits[k+1] - col_splits[k]
                
                a_row_start = row_splits[i]
                a_col_start = col_splits[j]
                b_row_start = row_splits[j]
                b_col_start = col_splits[k]
                c_row_start = row_splits[i]
                c_col_start = col_splits[k]
                
                if ha == 4 and wa == 4 and hb == 4 and wb == 4:
                    # Use 4x4x4 Strassen
                    for term in strassen_4:
                        u_local, u_coeffs, v_local, v_coeffs, w_local, w_coeffs = term
                        u_dict = {}
                        for li, coeff in zip(u_local, u_coeffs):
                            if coeff != 0:
                                li_row = li // 4
                                li_col = li % 4
                                global_u = (a_row_start + li_row) * m + (a_col_start + li_col)
                                u_dict[global_u] = u_dict.get(global_u, 0) + coeff
                        v_dict = {}
                        for li, coeff in zip(v_local, v_coeffs):
                            if coeff != 0:
                                li_row = li // 4
                                li_col = li % 4
                                global_v = (b_row_start + li_row) * p + (b_col_start + li_col)
                                v_dict[global_v] = v_dict.get(global_v, 0) + coeff
                        w_dict = {}
                        for li, coeff in zip(w_local, w_coeffs):
                            if coeff != 0:
                                li_row = li // 4
                                li_col = li % 4
                                global_w = (c_col_start + li_col) * n + (c_row_start + li_row)
                                w_dict[global_w] = w_dict.get(global_w, 0) + coeff
                        terms.append((u_dict, v_dict, w_dict))
                else:
                    # Schoolbook
                    for li in range(ha):
                        for lj in range(wa):
                            for lk in range(wb):
                                global_u = (a_row_start + li) * m + (a_col_start + lj)
                                global_v = (b_row_start + lj) * p + (b_col_start + lk)
                                global_w = (c_col_start + lk) * n + (c_row_start + li)
                                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # Handle the 4x1, 1x4, 1x1 boundary blocks
    # A[0:4, 4:5] * B[4:5, 0:4] -> C[0:4, 0:4]  (4x1 * 1x4 -> 4x4)
    # A[0:4, 4:5] * B[4:5, 4:5] -> C[0:4, 4:5]  (4x1 * 1x1 -> 4x1)
    # A[4:5, 0:4] * B[0:4, 0:4] -> C[4:5, 0:4]  (1x4 * 4x4 -> 1x4)
    # A[4:5, 0:4] * B[0:4, 4:5] -> C[4:5, 4:5]  (1x4 * 4x1 -> 1x1)
    # A[4:5, 4:5] * B[4:5, 0:4] -> C[4:5, 0:4]  (1x1 * 1x4 -> 1x4)
    # A[4:5, 4:5] * B[4:5, 4:5] -> C[4:5, 4:5]  (1x1 * 1x1 -> 1x1)
    
    # (i=0, j=1, k=0): A[0:4,4:5] * B[4:5,0:4] -> C[0:4,0:4]
    a_rs, a_cs = 0, 4
    b_rs, b_cs = 4, 0
    c_rs, c_cs = 0, 0
    for li in range(4):
        for lj in range(1):
            for lk in range(4):
                gu = (a_rs + li) * m + (a_cs + lj)
                gv = (b_rs + lj) * p + (b_cs + lk)
                gw = (c_cs + lk) * n + (c_rs + li)
                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))
    
    # (i=0, j=1, k=1): A[0:4,4:5] * B[4:5,4:5] -> C[0:4,4:5]
    a_rs, a_cs = 0, 4
    b_rs, b_cs = 4, 4
    c_rs, c_cs = 0, 4
    for li in range(4):
        for lj in range(1):
            for lk in range(1):
                gu = (a_rs + li) * m + (a_cs + lj)
                gv = (b_rs + lj) * p + (b_cs + lk)
                gw = (c_cs + lk) * n + (c_rs + li)
                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))
    
    # (i=1, j=0, k=0): A[4:5,0:4] * B[0:4,0:4] -> C[4:5,0:4]
    a_rs, a_cs = 4, 0
    b_rs, b_cs = 0, 0
    c_rs, c_cs = 4, 0
    for li in range(1):
        for lj in range(4):
            for lk in range(4):
                gu = (a_rs + li) * m + (a_cs + lj)
                gv = (b_rs + lj) * p + (b_cs + lk)
                gw = (c_cs + lk) * n + (c_rs + li)
                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))
    
    # (i=1, j=0, k=1): A[4:5,0:4] * B[0:4,4:5] -> C[4:5,4:5]
    a_rs, a_cs = 4, 0
    b_rs, b_cs = 0, 4
    c_rs, c_cs = 4, 4
    for li in range(1):
        for lj in range(4):
            for lk in range(1):
                gu = (a_rs + li) * m + (a_cs + lj)
                gv = (b_rs + lj) * p + (b_cs + lk)
                gw = (c_cs + lk) * n + (c_rs + li)
                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))
    
    # (i=1, j=1, k=0): A[4:5,4:5] * B[4:5,0:4] -> C[4:5,0:4]
    a_rs, a_cs = 4, 4
    b_rs, b_cs = 4, 0
    c_rs, c_cs = 4, 0
    for li in range(1):
        for lj in range(1):
            for lk in range(4):
                gu = (a_rs + li) * m + (a_cs + lj)
                gv = (b_rs + lj) * p + (b_cs + lk)
                gw = (c_cs + lk) * n + (c_rs + li)
                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))
    
    # (i=1, j=1, k=1): A[4:5,4:5] * B[4:5,4:5] -> C[4:5,4:5]
    a_rs, a_cs = 4, 4
    b_rs, b_cs = 4, 4
    c_rs, c_cs = 4, 4
    for li in range(1):
        for lj in range(1):
            for lk in range(1):
                gu = (a_rs + li) * m + (a_cs + lj)
                gv = (b_rs + lj) * p + (b_cs + lk)
                gw = (c_cs + lk) * n + (c_rs + li)
                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))
    
    rank = len(terms)
    
    # Build U, V, W
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < U_rows:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < V_rows:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < W_rows:
                W[idx][c] = coeff
    
    # Verify
    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
