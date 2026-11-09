"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Strategy: Use Strassen for the 4x4 sub-block (rank 49), schoolbook for cross terms,
    # then attempt to identify linearly dependent rank-1 components to reduce total rank.
    
    # Build all rank-1 terms as triples (U_row, V_row, W_row, coefficient)
    # Each term contributes: U[idx_a] * V[idx_b] * W[idx_c] * coeff
    # where idx_a = i*m+j, idx_b = j*p+k, idx_c = k*n+i
    
    terms = []  # list of (a_idx, b_idx, c_idx, coeff)
    
    def add_term(a_idx, b_idx, c_idx, coeff=1):
        terms.append((a_idx, b_idx, c_idx, coeff))
    
    # --- Strassen decomposition for 4x4 * 4x4 = 4x4 ---
    # Split 4x4 into four 2x2 blocks:
    # A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C11=C[0:2,0:2], C12=C[0:2,2:4], C21=C[2:4,0:2], C22=C[2:4,2:4]
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    #
    # Each M_i is a 2x2*2x2 product computed with Strassen (7 scalar mults)
    # Total: 7 * 7 = 49 terms
    
    # Helper: add Strassen 2x2 terms for X[row_base:row_base+2, col_base:col_base+2] * Y[...] = Z[...]
    def strassen_2x2(x_rb, x_cb, y_rb, y_cb, z_rb, z_cb, scale=1):
        """Add 7 Strassen rank-1 terms for a 2x2 block multiplication."""
        # For 2x2: X[0,0]*Y[0,0] etc.
        # Strassen combinations:
        # m1 = (x00+x11)(y00+y11)
        # m2 = (x10+x11)y00
        # m3 = x00(y01-y11)
        # m4 = x11(y10-y00)
        # m5 = (x00+x01)y11
        # m6 = (x10-x00)(y00+y01)
        # m7 = (x01-x11)(y10+y11)
        #
        # z00 = m1 - m4 + m6
        # z01 = m3 + m5
        # z10 = m2 + m4
        # z11 = m1 - m3 + m7
        
        # Each m_i is a product of two linear combinations.
        # m1 = (x00+x11)(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
        # This expands to 4 rank-1 scalar terms, but in the tensor decomposition,
        # each scalar multiplication corresponds to one rank-1 term in the full tensor.
        #
        # Actually, for the tensor decomposition, each "scalar multiplication" in Strassen
        # corresponds to one rank-1 term where U, V, W have specific non-zero entries.
        
        # Let me define the 7 Strassen terms explicitly.
        # Each term is defined by which A elements, B elements, and C elements participate.
        
        # m1: (x00+x11)(y00+y11) contributes to z00 and z11
        # U: x00 at (x_rb, x_cb), x11 at (x_rb+1, x_cb+1)
        # V: y00 at (y_rb, y_cb), y11 at (y_rb+1, y_cb+1)  
        # W: z00 at (z_rb, z_rb), z11 at (z_rb+1, z_cb+1)
        # But this is a product of sums, so it's actually 4 terms...
        
        # Wait, I need to think about this differently.
        # In the tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r,
        # each rank-1 term u_r ⊗ v_r ⊗ w_r has u_r, v_r, w_r as vectors.
        #
        # For the schoolbook term A[i,j]*B[j,k] -> C[i,k]:
        # u_r has 1 at position (i,j), v_r has 1 at position (j,k), w_r has 1 at position (k,i)
        #
        # For Strassen, m1 = (A11+A22)(B11+B22) where A11 is a 2x2 block.
        # This means we need a rank-1 term where:
        # u_r has entries corresponding to A11 and A22 elements
        # v_r has entries corresponding to B11 and B22 elements
        # w_r has entries corresponding to C11 and C22 elements
        #
        # But (A11+A22)(B11+B22) is a 2x2 matrix product, not a scalar.
        # So m1 is itself a 2x2*2x2=2x2 product, which needs 7 scalar multiplications.
        #
        # The full 4x4 Strassen has 7 "blocks", each of which is a 2x2*2x2 product
        # computed with 7 scalar multiplications, giving 49 total.
        
        # Let me just enumerate all 49 scalar terms directly.
        # For the 4x4 Strassen, I'll define the 7 M-blocks, and for each,
        # enumerate the 7 Strassen scalar multiplications.
        
        # Actually, let me use a cleaner approach. I'll define the 4x4 Strassen
        # by specifying for each of the 49 scalar multiplications which A, B, C
        # elements are involved and with what coefficients.
        
        # The 4x4 Strassen can be seen as:
        # Level 1: 7 combinations of 2x2 blocks
        # Level 2: Each 2x2 block multiplication uses 7 scalar Strassen terms
        
        # Let me define the 2x2 Strassen template first.
        # For X (2x2) * Y (2x2) = Z (2x2):
        # x00=X[0,0], x01=X[0,1], x10=X[1,0], x11=X[1,1]
        # y00=Y[0,0], y01=Y[0,1], y10=Y[1,0], y11=Y[1,1]
        # z00=Z[0,0], z01=Z[0,1], z10=Z[1,0], z11=Z[1,1]
        
        # The 7 Strassen scalar multiplications and their contributions:
        # s1 = (x00+x11)(y00+y11)
        #   -> z00 += s1, z11 += s1
        # s2 = (x10+x11)y00
        #   -> z10 += s2
        # s3 = x00(y01-y11)
        #   -> z01 += s3
        # s4 = x11(y10-y00)
        #   -> z00 -= s4, z10 += s4
        # s5 = (x00+x01)y11
        #   -> z01 += s5
        # s6 = (x10-x00)(y00+y01)
        #   -> z00 += s6
        # s7 = (x01-x11)(y10+y11)
        #   -> z11 += s7
        
        # Each s_i is a product of two linear combinations of X and Y elements.
        # In the tensor decomposition, each s_i corresponds to ONE rank-1 term
        # where U has the X coefficients, V has the Y coefficients, and W has
        # the Z output coefficients.
        
        # So for s1 = (x00+x11)(y00+y11):
        # U: {x00: 1, x11: 1}
        # V: {y00: 1, y11: 1}
        # W: {z00: 1, z11: 1}
        
        # This is a valid rank-1 term in the tensor!
        
        # Now for the 4x4 case, I need to map the 2x2 block elements to full matrix indices.
        
        # Let me define a helper that, given the 2x2 Strassen template and the
        # mapping from 2x2 local indices to full matrix indices, generates the terms.
        
        # The 2x2 Strassen template (7 terms):
        # Each term: (U_coeffs, V_coeffs, W_coeffs)
        # where coeffs are dicts from local (row,col) to coefficient
        
        strassen_2x2_template = [
            # s1: (x00+x11)(y00+y11) -> z00+1, z11+1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # s2: (x10+x11)y00 -> z10+1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # s3: x00(y01-y11) -> z01+1
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # s4: x11(y10-y00) -> z00-1, z10+1
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (1,0):1}),
            # s5: (x00+x01)y11 -> z01+1
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
            # s6: (x10-x00)(y00+y01) -> z00+1
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1}),
            # s7: (x01-x11)(y10+y11) -> z11+1
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}),
        ]
        
        # For the 4x4 Strassen, I have 7 M-blocks, each is a 2x2*2x2 product.
        # M1 = (A11+A22)(B11+B22): X = A11+A22, Y = B11+B22, Z contributes to C11, C22
        # M2 = (A21+A22)B11: X = A21+A22, Y = B11, Z contributes to C21
        # M3 = A11(B12-B22): X = A11, Y = B12-B22, Z contributes to C12
        # M4 = A22(B21-B11): X = A22, Y = B21-B11, Z contributes to C11(-), C21(+)
        # M5 = (A11+A12)B22: X = A11+A12, Y = B22, Z contributes to C12
        # M6 = (A21-A11)(B11+B12): X = A21-A11, Y = B11+B12, Z contributes to C11
        # M7 = (A12-A22)(B21+B22): X = A12-A22, Y = B21+B22, Z contributes to C22
        
        # For each M-block, the X is a sum/difference of two 2x2 A-blocks,
        # Y is a sum/difference of two 2x2 B-blocks, and Z contributes to specific C-blocks.
        
        # Let me define the 7 M-blocks with their X-source, Y-source, and Z-target info.
        # X-source: list of (block, sign) where block is (row_offset, col_offset) for the 2x2 block
        # Y-source: list of (block, sign)
        # Z-target: list of (block, sign)
        
        # 4x4 block structure:
        # Block (0,0) = rows 0-1, cols 0-1
        # Block (0,1) = rows 0-1, cols 2-3
        # Block (1,0) = rows 2-3, cols 0-1
        # Block (1,1) = rows 2-3, cols 2-3
        
        m_blocks = [
            # M1: X=A11+A22, Y=B11+B22, Z->C11(+), C22(+)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2: X=A21+A22, Y=B11, Z->C21(+)
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
            # M3: X=A11, Y=B12-B22, Z->C12(+)
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
            # M4: X=A22, Y=B21-B11, Z->C11(-), C21(+)
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,0,1)]),
            # M5: X=A11+A12, Y=B22, Z->C12(+)
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1)]),
            # M6: X=A21-A11, Y=B11+B12, Z->C11(+)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)]),
            # M7: X=A12-A22, Y=B21+B22, Z->C22(+)
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(1,1,1)]),
        ]
        
        for m_idx, (x_src, y_src, z_tgt) in enumerate(m_blocks):
            for s_idx, (u_local, v_local, w_local) in enumerate(strassen_2x2_template):
                # Build the full U, V, W coefficient dicts for this rank-1 term
                u_full = {}  # (i,j) -> coeff
                v_full = {}  # (j,k) -> coeff
                w_full = {}  # (k,i) -> coeff
                
                # For U: combine contributions from all X-source blocks
                for (x_r, x_c, x_sign) in x_src:
                    for ((lr, lc), coeff) in u_local.items():
                        full_i = 2 * x_r + lr
                        full_j = 2 * x_c + lc
                        key = (full_i, full_j)
                        u_full[key] = u_full.get(key, 0) + x_sign * coeff
                
                # For V: combine contributions from all Y-source blocks
                for (y_r, y_c, y_sign) in y_src:
                    for ((lr, lc), coeff) in v_local.items():
                        full_j = 2 * y_r + lr
                        full_k = 2 * y_c + lc
                        key = (full_j, full_k)
                        v_full[key] = v_full.get(key, 0) + y_sign * coeff
                
                # For W: combine contributions from all Z-target blocks
                for (z_r, z_c, z_sign) in z_tgt:
                    for ((lr, lc), coeff) in w_local.items():
                        full_k = 2 * z_r + lr
                        full_i = 2 * z_c + lc
                        key = (full_k, full_i)
                        w_full[key] = w_full.get(key, 0) + z_sign * coeff
                
                # Now add this rank-1 term
                # U indices: (i*m + j) for each (i,j) in u_full
                # V indices: (j*p + k) for each (j,k) in v_full
                # W indices: (k*n + i) for each (k,i) in w_full
                #
                # But a rank-1 term in the tensor is defined by vectors U_r, V_r, W_r.
                # The contribution to the tensor is:
                # sum_{(i,j),(j,k),(k,i)} U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                #
                # For this to equal the desired product, we need:
                # For each output (i,k): sum_j A[i,j]*B[j,k] = sum_r U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i]
                #
                # Actually, the tensor T is defined such that:
                # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
                # and we want T = sum_r u_r ⊗ v_r ⊗ w_r
                #
                # The (i,j,j',k,k',i') component of u_r ⊗ v_r ⊗ w_r is
                # u_r[i,j] * v_r[j',k] * w_r[k',i']
                #
                # For this to match T, we need:
                # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
                #
                # So each rank-1 term (u_r, v_r, w_r) contributes:
                # For each (i,j): u_r[i*m+j]
                # For each (j,k): v_r[j*p+k]
                # For each (k,i): w_r[k*n+i]
                # And the product u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] should sum to 1
                # over all r for each (i,j,k).
                
                # So I'll store the term as (u_full, v_full, w_full)
                add_term(u_full, v_full, w_full)
    
    # --- Cross terms (schoolbook) ---
    # A[0:4,4] * B[4,0:4] -> C[0:4,0:4]
    for i in range(4):
        for k in range(4):
            u_full = {(i, 4): 1}
            v_full = {(4, k): 1}
            w_full = {(k, i): 1}
            add_term(u_full, v_full, w_full)
    
    # A[0:4,0:4] * B[0:4,4] -> C[0:4,4]
    for i in range(4):
        for j in range(4):
            u_full = {(i, j): 1}
            v_full = {(j, 4): 1}
            w_full = {(4, i): 1}
            add_term(u_full, v_full, w_full)
    
    # A[0:4,4] * B[4,4] -> C[0:4,4]
    for i in range(4):
        u_full = {(i, 4): 1}
        v_full = {(4, 4): 1}
        w_full = {(4, i): 1}
        add_term(u_full, v_full, w_full)
    
    # A[4,0:4] * B[0:4,0:4] -> C[4,0:4]
    for j in range(4):
        for k in range(4):
            u_full = {(4, j): 1}
            v_full = {(j, k): 1}
            w_full = {(k, 4): 1}
            add_term(u_full, v_full, w_full)
    
    # A[4,4] * B[4,0:4] -> C[4,0:4]
    for k in range(4):
        u_full = {(4, 4): 1}
        v_full = {(4, k): 1}
        w_full = {(k, 4): 1}
        add_term(u_full, v_full, w_full)
    
    # A[4,0:4] * B[0:4,4] -> C[4,4]
    for j in range(4):
        u_full = {(4, j): 1}
        v_full = {(j, 4): 1}
        w_full = {(4, 4): 1}
        add_term(u_full, v_full, w_full)
    
    # A[4,4] * B[4,4] -> C[4,4]
    u_full = {(4, 4): 1}
    v_full = {(4, 4): 1}
    w_full = {(4, 4): 1}
    add_term(u_full, v_full, w_full)
    
    # Now try to find linear dependencies among the terms.
    # Two terms (u1,v1,w1) and (u2,v2,w2) are linearly dependent if one is a scalar multiple of the other.
    # More generally, we can check if any term is a linear combination of others.
    #
    # For now, let's just check for exact duplicates (same U, V, W up to scaling).
    
    # Convert terms to a canonical form for comparison
    def term_key(u, v, w):
        # Scale so that the first non-zero entry in U is 1
        u_items = sorted(u.items())
        v_items = sorted(v.items())
        w_items = sorted(w.items())
        
        # Find first non-zero in U
        scale = None
        for key, val in u_items:
            if val != 0:
                scale = val
                break
        if scale == 0:
            return None
        
        u_scaled = tuple((k, val / scale) for k, val in u_items)
        v_scaled = tuple((k, val / scale) for k, val in v_items)
        w_scaled = tuple((k, val / scale) for k, val in w_items)
        
        return (u_scaled, v_scaled, w_scaled)
    
    # Check for duplicates
    seen = set()
    unique_terms = []
    for term in terms:
        u, v, w = term
        key = term_key(u, v, w)
        if key is not None and key not in seen:
            seen.add(key)
            unique_terms.append(term)
        elif key is None:
            unique_terms.append(term)
    
    # Use unique_terms as our decomposition
    final_terms = unique_terms
    rank = len(final_terms)
    
    # Build the U, V, W matrices
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(final_terms):
        for (i, j), val in u.items():
            U_mat[i * m + j][r] = val
        for (j, k), val in v.items():
            V_mat[j * p + k][r] = val
        for (k, i), val in w.items():
            W_mat[k * n + i][r] = val
    
    # Verify the decomposition
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                expected = 1
                if abs(val - expected) > 1e-9:
                    error += abs(val - expected)
    
    # If there's an error, fall back to schoolbook
    if error > 1e-6:
        # Fall back to schoolbook decomposition (rank 125)
        U_mat = [[0] * (n * m * p) for _ in range(n * m)]
        V_mat = [[0] * (n * m * p) for _ in range(m * p)]
        W_mat = [[0] * (n * m * p) for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_mat[i * m + j][r] = 1
                    V_mat[j * p + k][r] = 1
                    W_mat[k * n + i][r] = 1
                    r += 1
        
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
