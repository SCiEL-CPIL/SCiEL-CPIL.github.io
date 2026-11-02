import time
import math

def construct():
    n, m, p = 3, 3, 6
    
    # Rank 51 decomposition
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(i, j, k, coeff):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = coeff
        V[j * p + k][col] = coeff
        W[k * n + i][col] = coeff
        col += 1

    def add_strassen_2x2x2(i0, j0, k0, i1, j1, k1):
        # Adds a 2x2x2 Strassen block
        # A is [i0..i1, j0..j1]
        # B is [j0..j1, k0..k1]
        # C is [k0..k1, i0..i1]
        # Note: i0, j0, k0 are start indices, i1, j1, k1 are end indices (exclusive or inclusive? Let's use exclusive range logic or explicit indices)
        # Let's use explicit indices for the 2x2 blocks
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # We need to add these to the global U, V, W.
        # Each M is a sum of products.
        # We can just add the 7 rank-1 terms with their coefficients.
        # But wait, the coefficients are in the U, V, W matrices.
        # For Strassen, we can't just add 7 rank-1 terms with scalar coefficients in U,V,W 
        # because the "multiplication" is distributed.
        # Actually, yes we can.
        # A rank-1 term is u * v * w.
        # Strassen term M1 = (A00+A11)(B00+B11).
        # This is NOT a single rank-1 term in the tensor decomposition sense if we just put coefficients in U,V,W.
        # The tensor decomposition requires sum_c U_c V_c W_c.
        # If we use Strassen, we are replacing 8 rank-1 terms with 7 "complex" terms.
        # But each Strassen term like (A00+A11)B00 is not a rank-1 term in the original tensor basis.
        # Wait.
        # The tensor is T[i,j,k].
        # Strassen computes C using 7 multiplications of sums.
        # This corresponds to a rank 7 decomposition.
        # The factors U, V, W will have entries that are sums of the original indices.
        # For example, for M2 = (A10+A11)B00:
        # U has non-zeros at (1,0) and (1,1) with value 1.
        # V has non-zero at (0,0) with value 1.
        # W has non-zeros at (0,1) and (1,1) with value 1?
        # No.
        # Let's trace M2.
        # M2 contributes to C10 and C11.
        # C10 += M2. C11 -= M2.
        # So W[0,1] += 1, W[1,1] -= 1.
        # U[1,0] = 1, U[1,1] = 1.
        # V[0,0] = 1.
        
        # So yes, we can just add the coefficients to U, V, W for each of the 7 Strassen terms.
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank:
                return
            for (u_idx, u_val) in u_coeffs:
                U[u_idx][col] += u_val
            for (v_idx, v_val) in v_coeffs:
                V[v_idx][col] += v_val
            for (w_idx, w_val) in w_coeffs:
                W[w_idx][col] += w_val
            col += 1

        # Indices
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = i0 + 1 * m + j0
        a11 = i0 + 1 * m + j0 + 1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = j0 + 1 * p + k0
        b11 = j0 + 1 * p + k0 + 1
        
        c00 = k0 * n + i0
        c01 = k0 * n + i0 + 1
        c10 = k0 + 1 * n + i0
        c11 = k0 + 1 * n + i0 + 1
        
        # M1 = (A00+A11)(B00+B11)
        # U: a00:1, a11:1
        # V: b00:1, b11:1
        # W: c00:1, c01:1, c10:1, c11:1 (from C00+=M1, C11+=M1)
        # Wait, C00 = M1 + ... C11 = M1 - ...
        # So W[c00] += 1, W[c11] += 1.
        add_strassen_term(
            [(a00, 1), (a11, 1)],
            [(b00, 1), (b11, 1)],
            [(c00, 1), (c11, 1)]
        )
        
        # M2 = (A10+A11)B00
        # U: a10:1, a11:1
        # V: b00:1
        # W: c10:1, c11:-1 (C10+=M2, C11-=M2)
        add_strassen_term(
            [(a10, 1), (a11, 1)],
            [(b00, 1)],
            [(c10, 1), (c11, -1)]
        )
        
        # M3 = A00(B10-B11)
        # U: a00:1
        # V: b10:1, b11:-1
        # W: c01:1, c11:1 (C01+=M3, C11+=M3)
        add_strassen_term(
            [(a00, 1)],
            [(b10, 1), (b11, -1)],
            [(c01, 1), (c11, 1)]
        )
        
        # M4 = A11(B01-B00)
        # U: a11:1
        # V: b01:1, b00:-1
        # W: c00:1, c10:1 (C00+=M4, C10+=M4)
        add_strassen_term(
            [(a11, 1)],
            [(b01, 1), (b00, -1)],
            [(c00, 1), (c10, 1)]
        )
        
        # M5 = (A00+A01)B11
        # U: a00:1, a01:1
        # V: b11:1
        # W: c00:-1, c01:1 (C00-=M5, C01+=M5)
        add_strassen_term(
            [(a00, 1), (a01, 1)],
            [(b11, 1)],
            [(c00, -1), (c01, 1)]
        )
        
        # M6 = (A10-A00)(B00+B01)
        # U: a10:1, a00:-1
        # V: b00:1, b01:1
        # W: c11:-1 (C11-=M6)
        add_strassen_term(
            [(a10, 1), (a00, -1)],
            [(b00, 1), (b01, 1)],
            [(c11, -1)]
        )
        
        # M7 = (A01-A11)(B10+B11)
        # U: a01:1, a11:-1
        # V: b10:1, b11:1
        # W: c00:1 (C00+=M7)
        add_strassen_term(
            [(a01, 1), (a11, -1)],
            [(b10, 1), (b11, 1)],
            [(c00, 1)]
        )

    # Block 1: A11 * B11 and A11 * C11
    # A11: i=0,1, j=0,1
    # B11: j=0,1, k=0,1
    # C11: j=0,1, k=3,4
    add_strassen_2x2x2(0, 0, 0, 2, 2, 2) # i0=0, j0=0, k0=0, i1=2, j1=2, k1=2
    add_strassen_2x2x2(0, 0, 3, 2, 2, 5) # i0=0, j0=0, k0=3, i1=2, j1=2, k1=5

    # Block 2: A11 * [B12, C12]
    # A11: i=0,1, j=0,1
    # B12: j=0,1, k=2
    # C12: j=0,1, k=5
    # Combined B: j=0,1, k in {2,5}
    # This is not a contiguous 2x2 block in B.
    # Strassen requires B to be a 2x2 matrix.
    # Here the columns are 2 and 5.
    # We can still use the Strassen logic if we map the indices correctly.
    # The function add_strassen_2x2x2 assumes contiguous k indices.
    # We need a generalized version or just do it manually.
    # Let's write a specific helper for non-contiguous k.
    
    def add_strassen_2x2x2_noncontig(i0, j0, k_list, i1, j1):
        # k_list: [k0, k1]
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = i0 + 1 * m + j0
        a11 = i0 + 1 * m + j0 + 1
        
        b00 = j0 * p + k_list[0]
        b01 = j0 * p + k_list[1]
        b10 = j0 + 1 * p + k_list[0]
        b11 = j0 + 1 * p + k_list[1]
        
        c00 = k_list[0] * n + i0
        c01 = k_list[0] * n + i0 + 1
        c10 = k_list[1] * n + i0
        c11 = k_list[1] * n + i0 + 1
        
        def add_term_local(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank: return
            for (u_idx, u_val) in u_coeffs: U[u_idx][col] += u_val
            for (v_idx, v_val) in v_coeffs: V[v_idx][col] += v_val
            for (w_idx, w_val) in w_coeffs: W[w_idx][col] += w_val
            col += 1

        add_term_local([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
        add_term_local([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)])
        add_term_local([(a00, 1)], [(b10, 1), (b11, -1)], [(c01, 1), (c11, 1)])
        add_term_local([(a11, 1)], [(b01, 1), (b00, -1)], [(c00, 1), (c10, 1)])
        add_term_local([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
        add_term_local([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, -1)])
        add_term_local([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])

    add_strassen_2x2x2_noncontig(0, 0, [2, 5], 2, 2)

    # Block 3: A12 * B21 and A12 * C21 (Schoolbook)
    # A12: i=0,1, j=2
    # B21: j=2, k=0,1
    # C21: j=2, k=3,4
    for i in [0, 1]:
        for k in [0, 1, 3, 4]:
            add_term(i, 2, k, 1)

    # Block 4: A12 * B22 and A12 * C22 (Schoolbook)
    # A12: i=0,1, j=2
    # B22: j=2, k=2
    # C22: j=2, k=5
    for i in [0, 1]:
        for k in [2, 5]:
            add_term(i, 2, k, 1)

    # Block 5: A21 * B11 and A21 * C11 (Schoolbook)
    # A21: i=2, j=0,1
    # B11: j=0,1, k=0,1
    # C11: j=0,1, k=3,4
    for j in [0, 1]:
        for k in [0, 1, 3, 4]:
            add_term(2, j, k, 1)

    # Block 6: A21 * B12 and A21 * C12 (Schoolbook)
    # A21: i=2, j=0,1
    # B12: j=0,1, k=2
    # C12: j=0,1, k=5
    for j in [0, 1]:
        for k in [2, 5]:
            add_term(2, j, k, 1)

    # Block 7: A22 * B21 and A22 * C21 (Schoolbook)
    # A22: i=2, j=2
    # B21: j=2, k=0,1
    # C21: j=2, k=3,4
    for k in [0, 1, 3, 4]:
        add_term(2, 2, k, 1)

    # Block 8: A22 * B22 and A22 * C22 (Schoolbook)
    # A22: i=2, j=2
    # B22: j=2, k=2
    # C22: j=2, k=5
    for k in [2, 5]:
        add_term(2, 2, k, 1)

    return {"U": U, "V": V, "W": W}
