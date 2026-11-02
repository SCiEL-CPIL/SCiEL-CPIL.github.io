import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 132
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 12 blocks
    # A blocks: (row_start, col_start)
    # B blocks: (row_start, col_start)
    # C blocks: (row_start, col_start)
    # A is 4x4, B is 4x9, C is 4x9
    
    # A11: 0,0; A12: 0,2; A21: 2,0; A22: 2,2
    # B11: 0,0; B12: 0,3; B13: 0,6
    # B21: 2,0; B22: 2,3; B23: 2,6
    # C11: 0,0; C12: 0,3; C13: 0,6
    # C21: 2,0; C22: 2,3; C23: 2,6
    
    blocks = [
        # (A_r, A_c, B_r, B_c, C_r, C_c)
        (0, 0, 0, 0, 0, 0), # A11 B11 -> C11
        (0, 2, 2, 0, 0, 0), # A12 B21 -> C11
        (0, 0, 0, 3, 0, 3), # A11 B12 -> C12
        (0, 2, 2, 3, 0, 3), # A12 B22 -> C12
        (0, 0, 0, 6, 0, 6), # A11 B13 -> C13
        (0, 2, 2, 6, 0, 6), # A12 B23 -> C13
        (2, 0, 0, 0, 2, 0), # A21 B11 -> C21
        (2, 2, 2, 0, 2, 0), # A22 B21 -> C21
        (2, 0, 0, 3, 2, 3), # A21 B12 -> C22
        (2, 2, 2, 3, 2, 3), # A22 B22 -> C22
        (2, 0, 0, 6, 2, 6), # A21 B13 -> C23
        (2, 2, 2, 6, 2, 6), # A22 B23 -> C23
    ]
    
    r_idx = 0
    for (ar, ac, br, bc, cr, cc) in blocks:
        # Local indices
        # A is 2x2: (0,0) to (1,1)
        # B is 2x3: (0,0) to (1,2)
        # C is 2x3: (0,0) to (1,2)
        
        # Helper to get global indices
        def a_idx(lr, lc):
            return (ar + lr) * m + (ac + lc)
            
        def b_idx(lr, lc):
            return (br + lr) * p + (bc + lc)
            
        def c_idx(lr, lc):
            return (cr + lr) * n + (cc + lc) # Wait, W index is k*n+i.
            # C index in tensor is (i, k) where i is row, k is col.
            # W index is k*n + i.
            # So for C block at (cr, cc), local (lr, lc) is global row cr+lr, col cc+lc.
            # i = cr+lr, k = cc+lc.
            # W index = (cc+lc)*n + (cr+lr).
            
        def w_idx(lr, lc):
            return (cc + lc) * n + (cr + lr)

        # Generate 11 terms
        
        # Term 1: P1 = x00 * y00 -> z00
        U[a_idx(0, 0), r_idx] = 1
        V[b_idx(0, 0), r_idx] = 1
        W[w_idx(0, 0), r_idx] = 1
        r_idx += 1
        
        # Term 2: P2 = x01 * y10 -> z00
        U[a_idx(0, 1), r_idx] = 1
        V[b_idx(1, 0), r_idx] = 1
        W[w_idx(0, 0), r_idx] = 1
        r_idx += 1
        
        # Term 3: P3 = x10 * y00 -> z10
        U[a_idx(1, 0), r_idx] = 1
        V[b_idx(0, 0), r_idx] = 1
        W[w_idx(1, 0), r_idx] = 1
        r_idx += 1
        
        # Term 4: P4 = x11 * y10 -> z10
        U[a_idx(1, 1), r_idx] = 1
        V[b_idx(1, 0), r_idx] = 1
        W[w_idx(1, 0), r_idx] = 1
        r_idx += 1
        
        # Terms 5-11: Strassen for cols 1,2 of B -> cols 1,2 of C
        # Y2 = [y01, y02; y11, y12]
        # S1 = (x00+x11)(y01+y11 + y02+y12) ? No.
        # S1 = (x00+x11) * (Y_col1 + Y_col2)
        # Y_col1 = [y01, y11]^T, Y_col2 = [y02, y12]^T.
        # S1 is 2x1.
        # S1_top = (x00+x11) * (y01+y11 + y02+y12)
        # S1_bot = (x00+x11) * (y01+y11 + y02+y12) ? No.
        # Strassen products are scalar multiplications of sums.
        # Here "sums" are vectors?
        # No, Strassen works on scalars.
        # We are doing 2x2 * 2x2.
        # The elements are scalars.
        # So S1 is a 2x1 vector?
        # No, if we apply Strassen to the matrix product, the intermediate products are matrices.
        # But here we are decomposing the tensor.
        # The "products" in the tensor decomposition are scalar multiplications.
        # So we need to expand the Strassen formulas into scalar terms.
        
        # Let's re-evaluate.
        # Z2 = X * Y2.
        # X = [[x00, x01], [x10, x11]]
        # Y2 = [[y01, y02], [y11, y12]]
        # Z2 = [[z01, z02], [z11, z12]]
        
        # Strassen for 2x2*2x2:
        # M1 = (x00+x11)(y01+y11)
        # M2 = (x10+x11)y01
        # M3 = x00(y02-y11) -- Wait, standard Strassen:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Here a=x00, b=x01, c=x10, d=x11.
        # e=y01, f=y02, g=y11, h=y12.
        
        # M1 = (x00+x11)(y01+y12) -> contributes to z01, z12
        # M2 = (x10+x11)y01 -> contributes to z11
        # M3 = x00(y02-y12) -> contributes to z02
        # M4 = x11(y11-y01) -> contributes to z11
        # M5 = (x00+x01)y12 -> contributes to z01, z02
        # M6 = (x10-x00)(y01+y02) -> contributes to z12
        # M7 = (x01-x11)(y11+y12) -> contributes to z01, z12
        
        # Reconstruction:
        # z01 = M1 + M4 - M5 + M7
        # z02 = M3 + M5
        # z11 = M2 + M4
        # z12 = M1 - M2 + M3 + M6
        
        # We need to add these to U, V, W.
        # Each M is a product of a linear combo of X and a linear combo of Y.
        # So we need to expand them into scalar terms.
        
        # M1 = (x00+x11)(y01+y12) = x00y01 + x00y12 + x11y01 + x11y12
        # This is 4 scalar products.
        # But we can keep them as combined terms in the tensor decomposition?
        # No, the tensor decomposition requires U, V, W to be matrices.
        # Each column r corresponds to a rank-1 term.
        # So we must expand M1 into 4 rank-1 terms?
        # Or can we have U, V entries that are sums?
        # Yes, U, V, W can have any values.
        # So for M1, we can have:
        # U[x00] += 1, V[y01] += 1, W[z01] += 1, W[z12] -= 1 ?
        # No, a single rank-1 term $U V W^T$ contributes to all (i,j,k) where U_i, V_j, W_k are non-zero.
        # If we set U[x00]=1, V[y01]=1, W[z01]=1, W[z12]=-1,
        # then the term is $1 \cdot 1 \cdot 1$ at (x00, y01, z01) and $1 \cdot 1 \cdot -1$ at (x00, y01, z12).
        # This exactly matches the contribution of $x00 y01$ to $z01$ and $z12$?
        # Wait, $x00 y01$ contributes to $z01$ (coeff 1) and $z12$ (coeff 1 in M1, but M1 is in z12 with coeff 1).
        # So yes, we can group terms!
        
        # Let's check if this works for all terms.
        # We just need to sum the coefficients for each (u, v, w) triple.
        # Since U, V, W are linear, we can just accumulate the coefficients.
        
        # So for each block, we can define the 11 "Strassen-like" products as linear combinations.
        # And add them to U, V, W.
        
        # Let's implement a helper to add a term:
        # add_term(u_indices, v_indices, w_indices, coeff)
        # where u_indices is a list of (idx, coeff_u), etc.
        
        # Actually, simpler:
        # For each of the 11 M's, we expand it into scalar products.
        # M1 = (x00+x11)(y01+y12)
        # Terms:
        # (x00, y01) -> coeff 1
        # (x00, y12) -> coeff 1
        # (x11, y01) -> coeff 1
        # (x11, y12) -> coeff 1
        
        # For each such pair (xu, yv), we add to W the appropriate coeffs for z.
        
        # Let's do this for all 11 M's.
        
        def add_strassen_term(m_idx, xu, xu_coeff, yv, yv_coeff, w_coeffs):
            # w_coeffs is dict {w_idx: coeff}
            pass
            
        # This is getting complicated to hardcode.
        # Let's just generate the 11 M's as 4 scalar terms each (or less if some coeffs are 0).
        # Total terms per block: 4 (for col 0) + 7*4 (for cols 1,2) = 32 terms?
        # No, M1 has 4 terms. M2 has 2 terms. M3 has 2. M4 has 2. M5 has 2. M6 has 2. M7 has 2.
        # Total 16 terms for the 2x2 part.
        # Plus 4 terms for the 2x1 part.
        # Total 20 terms per block.
        # 12 blocks * 20 = 240 terms.
        # This is worse than 132.
        
        # Can we keep the M's as single terms?
        # No, because U and V must be vectors.
        # If we set U[x00]=1, U[x11]=1, V[y01]=1, V[y12]=1,
        # then the product is $U V^T$ which is a 4x4 matrix of 1s.
        # This creates cross terms like $x00 y12$ and $x11 y01$ which are correct,
        # but also $x00 y01$ and $x11 y12$ which are correct.
        # Wait, if we use one rank-1 term for M1, we need:
        # U[x00]=1, U[x11]=1.
        # V[y01]=1, V[y12]=1.
        # Then the product is $\sum_{u \in \{x00, x11\}, v \in \{y01, y12\}} U_u V_v W_w$.
        # This gives 4 terms: $x00y01, x00y12, x11y01, x11y12$.
        # This is exactly M1.
        # So we CAN use 1 rank-1 term for M1!
        # We just set U and V to have 1s at the appropriate indices.
        # And W to have the appropriate coeffs for the output.
        
        # So rank is indeed 11 per block.
        
        # Let's implement this.
        
        # M1 = (x00+x11)(y01+y12)
        # U: x00=1, x11=1
        # V: y01=1, y12=1
        # W: z01 += 1, z12 += 1
        
        # M2 = (x10+x11)y01
        # U: x10=1, x11=1
        # V: y01=1
        # W: z11 += 1
        
        # M3 = x00(y02-y12)
        # U: x00=1
        # V: y02=1, y12=-1
        # W: z02 += 1
        
        # M4 = x11(y11-y01)
        # U: x11=1
        # V: y11=1, y01=-1
        # W: z11 += 1
        
        # M5 = (x00+x01)y12
        # U: x00=1, x01=1
        # V: y12=1
        # W: z01 += -1, z02 += 1
        
        # M6 = (x10-x00)(y01+y02)
        # U: x10=1, x00=-1
        # V: y01=1, y02=1
        # W: z12 += 1
        
        # M7 = (x01-x11)(y11+y12)
        # U: x01=1, x11=-1
        # V: y11=1, y12=1
        # W: z01 += 1, z12 += -1
        
        # Now we add these to U, V, W for the current block.
        
        def add_term(u_updates, v_updates, w_updates):
            nonlocal r_idx
            for u_idx, val in u_updates.items():
                U[a_idx(u_idx[0], u_idx[1]), r_idx] += val
            for v_idx, val in v_updates.items():
                V[b_idx(v_idx[0], v_idx[1]), r_idx] += val
            for w_idx, val in w_updates.items():
                W[w_idx, r_idx] += val
            r_idx += 1

        # M1
        add_term(
            {(0,0): 1, (1,1): 1},
            {(0,1): 1, (1,2): 1},
            {w_idx(0,1): 1, w_idx(1,2): 1}
        )
        # M2
        add_term(
            {(1,0): 1, (1,1): 1},
            {(0,1): 1},
            {w_idx(1,1): 1}
        )
        # M3
        add_term(
            {(0,0): 1},
            {(0,2): 1, (1,2): -1},
            {w_idx(0,2): 1}
        )
        # M4
        add_term(
            {(1,1): 1},
            {(1,1): 1, (0,1): -1},
            {w_idx(1,1): 1}
        )
        # M5
        add_term(
            {(0,0): 1, (0,1): 1},
            {(1,2): 1},
            {w_idx(0,1): -1, w_idx(0,2): 1}
        )
        # M6
        add_term(
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (0,2): 1},
            {w_idx(1,2): 1}
        )
        # M7
        add_term(
            {(0,1): 1, (1,1): -1},
            {(1,1): 1, (1,2): 1},
            {w_idx(0,1): 1, w_idx(1,2): -1}
        )
        
        # Col 0 terms (P1-P4)
        # P1 = x00 y00 -> z00
        add_term({(0,0): 1}, {(0,0): 1}, {w_idx(0,0): 1})
        # P2 = x01 y10 -> z00
        add_term({(0,1): 1}, {(1,0): 1}, {w_idx(0,0): 1})
        # P3 = x10 y00 -> z10
        add_term({(1,0): 1}, {(0,0): 1}, {w_idx(1,0): 1})
        # P4 = x11 y10 -> z10
        add_term({(1,1): 1}, {(1,0): 1}, {w_idx(1,0): 1})
        
    # Check if r_idx == 132
    assert r_idx == 132, f"Rank mismatch: {r_idx}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
