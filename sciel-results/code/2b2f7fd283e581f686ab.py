import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank = 76
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_222(i_off, j_off, k_off):
        """Add Strassen rank-7 decomposition for (2x2)*(2x2) at given offsets.
        i_off: row offset in A (0-indexed)
        j_off: col offset in A / row offset in B
        k_off: col offset in B
        """
        nonlocal col
        # Strassen 7 multiplications for 2x2*2x2:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Each M is a product of linear combos of A and B entries.
        # In tensor form, each M contributes a rank-1 term.
        
        # A indices: (i_off+di, j_off+dj) for di,dj in {0,1}
        # B indices: (j_off+dj, k_off+dk) for dj,dk in {0,1}
        # C indices: (i_off+di, k_off+dk) for di,dk in {0,1}
        
        strassen_terms = [
            # (coeffs for A, coeffs for B, coeffs for C)
            # M1: (a00+a11)*(b00+b11) -> C00 += 1, C11 += 1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2: (a10+a11)*b00 -> C10 += 1, C11 -= 1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3: a00*(b01-b11) -> C01 += 1, C11 += 1
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4: a11*(b10-b00) -> C00 += 1, C10 += 1
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5: (a00+a01)*b11 -> C00 -= 1, C01 += 1
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6: (a10-a00)*(b00+b01) -> C11 += 1
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7: (a01-a11)*(b10+b11) -> C00 += 1
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            for (di, dj), a_val in a_coeffs.items():
                for (dj2, dk), b_val in b_coeffs.items():
                    assert dj == dj2, "A and B must share same j index"
                    for (di2, dk2), c_val in c_coeffs.items():
                        # This term contributes a_val * b_val * c_val to T[i_off+di, j_off+dj, k_off+dk]
                        # But we need rank-1 terms, so we add one column per (M term)
                        pass
            # Actually, each M is a single rank-1 term in the tensor decomposition
            # U[i*m+j, col] = sum of A coefficients for this term
            # V[j*p+k, col] = sum of B coefficients for this term  
            # W[k*n+i, col] = sum of C coefficients for this term
            # But that's not how rank-1 works. Each rank-1 term has U[i*m+j] = u_ij, V[j*p+k] = v_jk, W[k*n+i] = w_ki
            # and the product u_ij * v_jk * w_ki should give the tensor entry.
            
            # For Strassen, each M corresponds to:
            # M = (sum a_coeffs[di,dj] * A[i_off+di, j_off+dj]) * (sum b_coeffs[dj,dk] * B[j_off+dj, k_off+dk])
            # And this M contributes to C as: sum c_coeffs[di,dk] * C[i_off+di, k_off+dk]
            #
            # In the tensor decomposition, this is ONE rank-1 term where:
            # U[i*m+j, col] = a_coeffs.get((i-i_off, j-j_off), 0) for i in [i_off, i_off+2), j in [j_off, j_off+2)
            # V[j*p+k, col] = b_coeffs.get((j-j_off, k-k_off), 0) for j in [j_off, j_off+2), k in [k_off, k_off+2)
            # W[k*n+i, col] = c_coeffs.get((i-i_off, k-k_off), 0) for k in [k_off, k_off+2), i in [i_off, i_off+2)
            
            for di in range(2):
                for dj in range(2):
                    u_val = a_coeffs.get((di, dj), 0)
                    if u_val != 0:
                        U[(i_off + di) * m + (j_off + dj), col] = u_val
            for dj in range(2):
                for dk in range(2):
                    v_val = b_coeffs.get((dj, dk), 0)
                    if v_val != 0:
                        V[(j_off + dj) * p + (k_off + dk), col] = v_val
            for di in range(2):
                for dk in range(2):
                    w_val = c_coeffs.get((di, dk), 0)
                    if w_val != 0:
                        W[(k_off + dk) * n + (i_off + di), col] = w_val
            col += 1
    
    def add_outer_product_213(i_off, j_off, k_off):
        """Add rank-1 decomposition for (2x1)*(1x3): C[i,k] = A[i,0]*B[0,k]
        This is an outer product, rank 1.
        """
        nonlocal col
        # A has 1 column (j=j_off), B has 1 row (j=j_off)
        # C[i,k] = A[i_off+i, j_off] * B[j_off, k_off+k]
        # In tensor: U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = 1 for all i,k
        # So set U[i*m+j_off, col] = 1, V[j_off*p+k, col] = 1, W[k*n+i, col] = 1
        for i in range(2):
            U[(i_off + i) * m + j_off, col] = 1.0
        for k in range(3):
            V[j_off * p + (k_off + k), col] = 1.0
        for i in range(2):
            for k in range(3):
                W[(k_off + k) * n + (i_off + i), col] = 1.0
        col += 1
    
    def add_schoolbook_223(i_off, j_off, k_off):
        """Add schoolbook for (2x2)*(2x3): rank 12. But we'll use Strassen+dot instead."""
        pass  # Not used, replaced by add_strassen_222 + add_dot_221
    
    def add_dot_221(i_off, j_off, k_off):
        """Add rank-2 for (2x2)*(2x1): C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        Two dot products, rank 2.
        """
        nonlocal col
        # Term 1: A[i,0]*B[0,0] -> C[i,0]
        for i in range(2):
            U[(i_off + i) * m + (j_off + 0), col] = 1.0
        V[(j_off + 0) * p + k_off, col] = 1.0
        for i in range(2):
            W[k_off * n + (i_off + i), col] = 1.0
        col += 1
        
        # Term 2: A[i,1]*B[1,0] -> C[i,0]
        for i in range(2):
            U[(i_off + i) * m + (j_off + 1), col] = 1.0
        V[(j_off + 1) * p + k_off, col] = 1.0
        for i in range(2):
            W[k_off * n + (i_off + i), col] = 1.0
        col += 1
    
    def add_strassen_223(i_off, j_off, k_off):
        """Add rank-9 for (2x2)*(2x3) using Strassen on 2x2x2 part + dot for 2x2x1 part.
        Split k into k=0 (single column) and k=1,2 (two columns).
        """
        # (2x2)*(2x1) part: rank 2
        add_dot_221(i_off, j_off, k_off)
        # (2x2)*(2x2) part: rank 7 (Strassen)
        add_strassen_222(i_off, j_off, k_off + 1)
    
    def add_strassen_233(i_off, j_off, k_off):
        """Add rank-10 for (2x3)*(3x3) using outer product for j=0 + strassen_223 for j=1,2.
        Split j into j=0 (single) and j=1,2 (two).
        """
        # (2x1)*(1x3) part: rank 1
        add_outer_product_213(i_off, j_off, k_off)
        # (2x2)*(2x3) part: rank 9
        add_strassen_223(i_off, j_off + 1, k_off)
    
    # Now build the full decomposition:
    # A1(4x2) * B1(2x6): split A1 into top(2x2), bot(2x2); B1 into left(2x3), right(2x3)
    # Block 1: A1_top * B1_left: (2x2)*(2x3) at i=0, j=0, k=0
    add_strassen_223(0, 0, 0)
    # Block 2: A1_top * B1_right: (2x2)*(2x3) at i=0, j=0, k=3
    add_strassen_223(0, 0, 3)
    # Block 3: A1_bot * B1_left: (2x2)*(2x3) at i=2, j=0, k=0
    add_strassen_223(2, 0, 0)
    # Block 4: A1_bot * B1_right: (2x2)*(2x3) at i=2, j=0, k=3
    add_strassen_223(2, 0, 3)
    
    # A2(4x3) * B2(3x6): split A2 into top(2x3), bot(2x3); B2 into left(3x3), right(3x3)
    # Block 5: A2_top * B2_left: (2x3)*(3x3) at i=0, j=2, k=0
    add_strassen_233(0, 2, 0)
    # Block 6: A2_top * B2_right: (2x3)*(3x3) at i=0, j=2, k=3
    add_strassen_233(0, 2, 3)
    # Block 7: A2_bot * B2_left: (2x3)*(3x3) at i=2, j=2, k=0
    add_strassen_233(2, 2, 0)
    # Block 8: A2_bot * B2_right: (2x3)*(3x3) at i=2, j=2, k=3
    add_strassen_233(2, 2, 3)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r_idx in range(r):
                        actual += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to trivial
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback), dtype=float)
        V_f = np.zeros((m * p, rank_fallback), dtype=float)
        W_f = np.zeros((p * n, rank_fallback), dtype=float)
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col_f] = 1
                    V_f[j * p + k, col_f] = 1
                    W_f[k * n + i, col_f] = 1
                    col_f += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
