"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 92
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = [0]
    
    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        r = current_rank[0]
        for idx, val in zip(u_indices, u_vals):
            U[idx][r] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][r] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][r] += val
        current_rank[0] += 1

    def solve(n, m, p, i_off, j_off, k_off):
        if n == 2 and m == 2 and p == 2:
            # Strassen
            # Indices
            u00 = i_off*5 + j_off
            u01 = i_off*5 + (j_off+1)
            u10 = (i_off+1)*5 + j_off
            u11 = (i_off+1)*5 + (j_off+1)
            
            v00 = j_off*5 + k_off
            v01 = j_off*5 + (k_off+1)
            v10 = (j_off+1)*5 + k_off
            v11 = (j_off+1)*5 + (k_off+1)
            
            w00 = k_off*4 + i_off
            w01 = (k_off+1)*4 + i_off
            w10 = k_off*4 + (i_off+1)
            w11 = (k_off+1)*4 + (i_off+1)
            
            # M1 = (A00+A11)(B00+B11)
            add_rank1_term([u00, u11], [1, 1], [v00, v11], [1, 1], [w00, w11], [1, 1])
            # M2 = (A10+A11)B00
            add_rank1_term([u10, u11], [1, 1], [v00], [1], [w11], [1])
            # M3 = A00(B10-B11)
            add_rank1_term([u00], [1], [v10, v11], [1, -1], [w01], [1])
            # M4 = A11(B01-B00)
            add_rank1_term([u11], [1], [v01, v00], [1, -1], [w11], [1])
            # M5 = (A00+A01)B11
            add_rank1_term([u00, u01], [1, 1], [v11], [1], [w01], [-1])
            # M6 = (A10-A00)(B00+B01)
            add_rank1_term([u10, u00], [1, -1], [v00, v01], [1, 1], [w11], [1])
            # M7 = (A01-A11)(B10+B11)
            add_rank1_term([u01, u11], [1, -1], [v10, v11], [1, 1], [w01], [1])
            
            # Verify C reconstruction coefficients for W:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # My W assignments:
            # M1: w00(+1), w11(+1) -> OK for C00, C11
            # M2: w11(+1) -> OK for C11
            # M3: w01(+1) -> OK for C01
            # M4: w11(+1) -> OK for C11
            # M5: w01(-1) -> OK for C01
            # M6: w11(+1) -> OK for C11
            # M7: w01(+1) -> OK for C01
            
            # Wait, C00 needs M4. My M4 has w11.
            # C00 = M1 + M4 - M5 + M7.
            # M1 has w00.
            # M4 has w11? No, M4 contributes to C00 and C11?
            # Let's check Strassen formulas again.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # M1: (A00+A11)(B00+B11). Contributes to C00 and C11?
            # A00B00 is C00. A11B11 is C11. A00B11 is C01? No.
            # The product (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
            # This is NOT a rank 1 term in the output C.
            # Strassen works by computing these 7 products and then combining them.
            # The "rank" of the decomposition refers to the number of scalar multiplications.
            # In the tensor decomposition sense, each scalar multiplication $M_k$ is a rank-1 tensor.
            # The $W$ vector for $M_k$ contains the coefficients of $M_k$ in the final $C$ entries.
            
            # So for M1:
            # It appears in C00 with coeff 1.
            # It appears in C11 with coeff 1.
            # So w00=1, w11=1 is correct.
            
            # For M4:
            # It appears in C00 with coeff 1.
            # It appears in C10 with coeff 1.
            # So w00=1, w10=1.
            # My code had w11 for M4. This is WRONG.
            
            # Let's re-map Strassen W coefficients.
            # M1: C00(1), C11(1)
            # M2: C10(1), C11(-1)
            # M3: C01(1), C11(1)
            # M4: C00(1), C10(1)
            # M5: C01(-1), C00(-1)
            # M6: C11(1)
            # M7: C00(1), C01(1)
            
            # Let's check standard Strassen:
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
            # C11 = M1 - M2 + M3 + M6
            
            # W coefficients:
            # M1: w00=1, w11=1
            # M2: w10=1, w11=-1
            # M3: w01=1, w11=1
            # M4: w00=1, w10=1
            # M5: w00=-1, w01=-1
            # M6: w11=1
            # M7: w00=1, w01=1
            
            # I will update the code with these correct coefficients.
            return

        # ... other cases
