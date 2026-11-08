import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of columns (terms)
    # Then transpose to get row-major matrices
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        if time.time() > deadline:
            return
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
        
    # 1. Strassen decomposition for 4x4x4 (first 4 columns of B)
    # We treat A as 4x4 and B_first4 as 4x4.
    # We split A and B_first4 into 2x2 blocks.
    # A blocks:
    # A00 (rows 0-1, cols 0-1)
    # A01 (rows 0-1, cols 2-3)
    # A10 (rows 2-3, cols 0-1)
    # A11 (rows 2-3, cols 2-3)
    #
    # B blocks (first 4 cols):
    # B00 (rows 0-1, cols 0-1)
    # B01 (rows 0-1, cols 2-3)
    # B10 (rows 2-3, cols 0-1)
    # B11 (rows 2-3, cols 2-3)
    
    # Strassen Steps for 2x2 blocks:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each Mi is a 2x2 matrix product. We decompose each 2x2 product using schoolbook (4 terms).
    # Total terms for 4x4x4 part: 7 * 4 = 28.
    
    # Define the 7 Strassen steps
    # Each step is defined by linear combinations of A blocks and B blocks.
    # a_coeffs: {block_id: coeff}
    # b_coeffs: {block_id: coeff}
    # block_id 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1) in block space (row_block, col_block)
    
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M0
        ({2: 1, 3: 1}, {0: 1}),       # M1
        ({0: 1}, {1: 1, 3: -1}),      # M2
        ({3: 1}, {2: 1, 0: -1}),      # M3
        ({0: 1, 1: 1}, {3: 1}),       # M4
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M5
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M6
    ]
    
    # Helper to get global row/col offset for a 2x2 block in a 4x4 matrix
    def get_block_offset(block_id):
        # block_id 0: rb=0, cb=0 -> r_off=0, c_off=0
        # block_id 1: rb=0, cb=1 -> r_off=0, c_off=2
        # block_id 2: rb=1, cb=0 -> r_off=2, c_off=0
        # block_id 3: rb=1, cb=1 -> r_off=2, c_off=2
        rb = block_id // 2
        cb = block_id % 2
        return rb * 2, cb * 2
    
    # Reconstruction coefficients for C blocks
    # c_block_id: 0: C00, 1: C01, 2: C10, 3: C11
    # step_idx: 0..6
    # Coefficients:
    # C00: M0(+1), M3(+1), M4(-1), M6(+1)
    # C01: M2(+1), M4(+1)
    # C10: M1(+1), M3(+1)
    # C11: M0(+1), M1(-1), M2(+1), M5(+1)
    
    recon = {
        0: {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        1: {2: 1, 4: 1},              # C01
        2: {1: 1, 3: 1},              # C10
        3: {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    }
    
    for step_idx, (a_coeffs, b_coeffs) in enumerate(steps):
        # For each of the 4 scalar multiplications in 2x2 schoolbook
        # C_local[li, lj] += A_local[li, lk] * B_local[lk, lj]
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    if time.time() > deadline:
                        break
                    
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    
                    # U part: A
                    # A is 4x4.
                    # A_total[li, lk] = sum_{b_a} coeff_a[b_a] * A_block[b_a][li, lk]
                    # We need to set u[global_row * m + global_col]
                    # Global row = r_off_a + li
                    # Global col = c_off_a + lk
                    for b_a, coeff_a in a_coeffs.items():
                        r_off, c_off = get_block_offset(b_a)
                        gr = r_off + li
                        gc = c_off + lk
                        u[gr * m + gc] += coeff_a
                        
                    # V part: B
                    # B is 4x5. We are using the first 4 columns.
                    # B_total[lk, lj] = sum_{b_b} coeff_b[b_b] * B_block[b_b][lk, lj]
                    # Global row = r_off_b + lk
                    # Global col = c_off_b + lj
                    for b_b, coeff_b in b_coeffs.items():
                        r_off, c_off = get_block_offset(b_b)
                        gr = r_off + lk
                        gc = c_off + lj
                        v[gr * p + gc] += coeff_b
                        
                    # W part: C
                    # C is 4x5. We are filling the first 4 columns.
                    # The result C_local[li, lj] is added to the appropriate C block.
                    # We need to determine which C block this local (li, lj) belongs to.
                    # The local (li, lj) corresponds to the position within the 2x2 block.
                    # The Strassen step produces a 2x2 matrix M_step.
                    # M_step contributes to C blocks according to `recon`.
                    # For a specific local position (li, lj) in M_step, it contributes to
                    # the same local position (li, lj) in the target C block.
                    
                    # We iterate over the C blocks that this step contributes to.
                    for c_block_id, coeff_c in recon[c_block_id if False else 0].items(): # Placeholder, logic below
                        pass # Disabled, using explicit loop below

                    # Correct logic:
                    # The step `step_idx` contributes to various C blocks.
                    # We need to add the contribution to W for each C block it affects.
                    # The local indices (li, lj) in the 2x2 result map to global indices
                    # in the C block.
                    
                    for c_block_id, coeff_c in recon.get(step_idx, {}).items():
                        # Wait, recon is defined as C_block -> {step_idx: coeff}
                        # So we need to check if step_idx is in recon[c_block_id]
                        pass

                    # Let's restructure recon for easier lookup: step_idx -> {c_block_id: coeff}
                    # C00: M0(+1), M3(+1), M4(-1), M6(+1)
                    # C01: M2(+1), M4(+1)
                    # C10: M1(+1), M3(+1)
                    # C11: M0(+1), M1(-1), M2(+1), M5(+1)
                    
                    # Invert recon:
                    # recon_inv[step_idx] = {c_block_id: coeff}
                    
                    if step_idx == 0: # M0
                        c_contribs = {0: 1, 3: 1}
                    elif step_idx == 1: # M1
                        c_contribs = {2: 1, 3: -1}
                    elif step_idx == 2: # M2
                        c_contribs = {1: 1, 3: 1}
                    elif step_idx == 3: # M3
                        c_contribs = {0: 1, 2: 1}
                    elif step_idx == 4: # M4
                        c_contribs = {0: -1, 1: 1}
                    elif step_idx == 5: # M5
                        c_contribs = {3: 1}
                    elif step_idx == 6: # M6
                        c_contribs = {0: 1}
                    else:
                        c_contribs = {}
                        
                    for c_block_id, coeff_c in c_contribs.items():
                        r_off, c_off = get_block_offset(c_block_id)
                        gr = r_off + li
                        gc = c_off + lj
                        # W index: gc * n + gr
                        # Note: gc is column index of C (0..3), gr is row index of C (0..3)
                        # W shape is (p*n, rank). Index is (col * n + row).
                        w[gc * n + gr] += coeff_c

                    if any(x != 0 for x in u) and any(x != 0 for x in v) and any(x != 0 for x in w):
                        add_term(u, v, w)

    # 2. Add terms for the 5th column (index 4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 4 terms.
    for j in range(4):
        if time.time() > deadline:
            break
        u = [0] * (n*m)
        v = [0] * (m*p)
        w = [0] * (p*n)
        
        # A[i, j] for all i
        for i in range(4):
            u[i * m + j] = 1
            
        # B[j, 4]
        v[j * p + 4] = 1
        
        # C[i, 4] for all i
        for i in range(4):
            # W index: col * n + row
            # col = 4, row = i
            w[4 * n + i] = 1
            
        add_term(u, v, w)

    # If we ran out of time or have no terms, return empty or partial
    if not U_cols:
        # Fallback: Return a valid empty structure or simple identity if needed?
        # But task requires exact decomposition. If we fail, we return what we have.
        # However, returning empty matrices might be invalid if rank > 0 is expected?
        # The evaluator likely checks if the decomposition is exact.
        # If we return empty, it's rank 0, which is wrong.
        # But we have a deadline. If we hit deadline, we return whatever we have.
        pass

    rank = len(U_cols)
    
    # Construct the output matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # U_cols is list of lists of length n*m
    U_out = np.zeros((n*m, rank))
    V_out = np.zeros((m*p, rank))
    W_out = np.zeros((p*n, rank))
    
    for k in range(rank):
        U_out[:, k] = U_cols[k]
        V_out[:, k] = V_cols[k]
        W_out[:, k] = W_cols[k]
        
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
