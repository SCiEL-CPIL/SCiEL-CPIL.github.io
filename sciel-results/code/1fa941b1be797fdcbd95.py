import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # Rank 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Define Strassen coefficients for 2x2x2
    # M1..M7
    # We need to map the 2x2 local indices to global indices.
    
    # Block 1: A_L (i=0,1; j=0,1), B_T (j=0,1; k=0..6)
    # A_R (i=0,1; j=2,3), B_B (j=2,3; k=0..6)
    
    def process_block(a_j_offset, b_j_offset):
        # a_j_offset: 0 for A_L, 2 for A_R
        # b_j_offset: 0 for B_T, 2 for B_B
        
        # 3 pairs of columns in B
        for pair_idx in range(3):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1
            
            # Strassen 7 multiplications
            # Local A indices: (0,0), (0,1), (1,0), (1,1) map to global
            # A[i, a_j_offset + local_j]
            # Local B indices: (0,0), (0,1), (1,0), (1,1) map to global
            # B[b_j_offset + local_j, k]
            
            # M1: (A00+A11)(B00+B11)
            # This is a product of sums, so it's not a single rank-1 term in the original variables?
            # No, the decomposition is a sum of rank-1 tensors in the *transformed* variables?
            # No, the rank-1 terms are of the form u * v * w where u, v, w are linear forms?
            # The task requires U, V, W to have integer/half-integer entries.
            # This implies the decomposition is of the form:
            # T = sum_r U[:, r] V[:, r] W[:, r]^T (outer product)
            # This corresponds to scalar multiplications.
            # Strassen uses products of linear combinations.
            # (A00+A11) is a linear combination of A's.
            # So U[:, r] will have entries like 1, 1, 0, 0...
            # V[:, r] will have entries like 1, 1, 0, 0...
            # W[:, r] will have entries like 1, 1, 0, 0...
            
            # We need to expand the Strassen steps into rank-1 terms.
            # Each Strassen multiplication M_k = L_A * L_B is a rank-1 term.
            # L_A is a linear form in A. L_B is a linear form in B.
            # The result contributes to C via linear combinations.
            
            # Let's define the 7 linear forms for A and B.
            # A forms:
            # LA1 = A00 + A11
            # LA2 = A10 + A11
            # LA3 = A00
            # LA4 = A11
            # LA5 = A00 + A01
            # LA6 = A10 - A00
            # LA7 = A01 - A11
            
            # B forms (for columns k1, k2):
            # LB1 = B0,k1 + B1,k2 ? No.
            # Strassen:
            # M1 = (A00+A11)(B00+B11) -> LA1 * (B0,k1 + B1,k2)
            # M2 = (A10+A11) B00 -> LA2 * B0,k1
            # M3 = A00 (B10-B11) -> LA3 * (B1,k1 - B1,k2)
            # M4 = A11 B11 -> LA4 * B1,k2
            # M5 = (A00+A01) B11 -> LA5 * B1,k2
            # M6 = (A10-A00) (B00+B10) -> LA6 * (B0,k1 + B1,k1)
            # M7 = (A01-A11) (B11+B01) -> LA7 * (B1,k2 + B0,k2)
            
            # We add 7 terms.
            
            # Term 1
            # U: A00, A11 coeff 1. Global A indices: 4*0+a_j_offset+0, 4*1+a_j_offset+1
            # V: B0,k1, B1,k2 coeff 1. Global B indices: 7*(b_j_offset+0)+k1, 7*(b_j_offset+1)+k2
            # W: C00, C11 coeff 1? No, W is for C.
            # The contribution of M1 to C is:
            # C00 += M1
            # C11 += M1
            # So W has 1 at C00 and C11?
            # Wait, the rank-1 term is U * V * W.
            # The product U*V is the scalar multiplication.
            # W distributes it to the output.
            
            # Let's trace M1 = (A00+A11)(B0,k1 + B1,k2)
            # U[:, col] has 1 at A00, A11.
            # V[:, col] has 1 at B0,k1, B1,k2.
            # W[:, col] must be such that sum W * (U*V) adds to the correct C components.
            # M1 contributes +1 to C00 and +1 to C11.
            # So W has 1 at C00 and 1 at C11.
            
            # We need to be careful with the indices.
            # C indices: i=0,1; k=k1,k2.
            # C00 -> i=0, k=k1. W index: 7*0 + k1.
            # C11 -> i=1, k=k2. W index: 7*1 + k2.
            
            # Let's implement a helper to add a term defined by linear forms.
            
            def add_strassen_term(la_indices, la_coeffs, lb_indices, lb_coeffs, lc_indices, lc_coeffs):
                nonlocal col
                for idx, val in zip(la_indices, la_coeffs):
                    U[idx][col] = val
                for idx, val in zip(lb_indices, lb_coeffs):
                    V[idx][col] = val
                for idx, val in zip(lc_indices, lc_coeffs):
                    W[idx][col] = val
                col += 1

            # A global indices
            a00 = 4*0 + a_j_offset + 0
            a01 = 4*0 + a_j_offset + 1
            a10 = 4*1 + a_j_offset + 0
            a11 = 4*1 + a_j_offset + 1
            
            # B global indices
            b0k1 = 7*(b_j_offset + 0) + k1
            b0k2 = 7*(b_j_offset + 0) + k2
            b1k1 = 7*(b_j_offset + 1) + k1
            b1k2 = 7*(b_j_offset + 1) + k2
            
            # C global indices
            c0k1 = 7*0 + k1
            c0k2 = 7*0 + k2
            c1k1 = 7*1 + k1
            c1k2 = 7*1 + k2
            
            # M1
            add_strassen_term(
                [a00, a11], [1, 1],
                [b0k1, b1k2], [1, 1],
                [c0k1, c1k2], [1, 1]
            )
            
            # M2: (A10+A11) B0,k1
            # C10 += M2
            add_strassen_term(
                [a10, a11], [1, 1],
                [b0k1], [1],
                [c1k1], [1]
            )
            
            # M3: A00 (B1,k1 - B1,k2)
            # C01 += M3 -> C0,k2 += M3
            add_strassen_term(
                [a00], [1],
                [b1k1, b1k2], [1, -1],
                [c0k2], [1]
            )
            
            # M4: A11 B1,k2
            # C00 += M4, C11 += M4
            add_strassen_term(
                [a11], [1],
                [b1k2], [1],
                [c0k1, c1k2], [1, 1]
            )
            
            # M5: (A00+A01) B1,k2
            # C00 -= M5, C01 += M5 -> C0,k2 += M5
            add_strassen_term(
                [a00, a01], [1, 1],
                [b1k2], [1],
                [c0k1, c0k2], [-1, 1]
            )
            
            # M6: (A10-A00) (B0,k1 + B1,k1)
            # C11 += M6
            add_strassen_term(
                [a10, a00], [1, -1],
                [b0k1, b1k1], [1, 1],
                [c1k2], [1]
            )
            
            # M7: (A01-A11) (B1,k2 + B0,k2)
            # C00 += M7, C11 += M7
            add_strassen_term(
                [a01, a11], [1, -1],
                [b1k2, b0k2], [1, 1],
                [c0k1, c1k2], [1, 1]
            )
            
        # Single column k=6
        k = 6
        # Schoolbook 4 terms
        # C0,k = A00 B0,k + A01 B1,k
        # C1,k = A10 B0,k + A11 B1,k
        
        b0k = 7*(b_j_offset + 0) + k
        b1k = 7*(b_j_offset + 1) + k
        c0k = 7*0 + k
        c1k = 7*1 + k
        
        # T1: A00 B0,k -> C0,k
        add_strassen_term(
            [a00], [1],
            [b0k], [1],
            [c0k], [1]
        )
        # T2: A01 B1,k -> C0,k
        add_strassen_term(
            [a01], [1],
            [b1k], [1],
            [c0k], [1]
        )
        # T3: A10 B0,k -> C1,k
        add_strassen_term(
            [a10], [1],
            [b0k], [1],
            [c1k], [1]
        )
        # T4: A11 B1,k -> C1,k
        add_strassen_term(
            [a11], [1],
            [b1k], [1],
            [c1k], [1]
        )
        
    process_block(0, 0)
    process_block(2, 2)
    
    return {"U": U, "V": V, "W": W}
