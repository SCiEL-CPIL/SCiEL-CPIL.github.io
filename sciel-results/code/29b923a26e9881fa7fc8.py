# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    
    # Rank calculation
    # 4x4x4 part: 4 block multiplications, each 2x2x2 using Strassen (rank 7)
    # But C_ab = sum_c A_ac B_cb.
    # C_00 = A_00 B_00 + A_01 B_10.
    # We compute A_00 B_00 (7 mults) and A_01 B_10 (7 mults).
    # Total 14 mults for C_00.
    # Similarly for C_01, C_10, C_11.
    # Total 56 mults for 4x4x4.
    # 4x4x1 part (5th col): 4 mults.
    # Total rank = 60.
    
    rank = 60
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a Strassen term for 2x2 multiplication
    # X is 2x2 block of A, Y is 2x2 block of B, Z is 2x2 block of C
    # X indices: (r, c) in block -> global (r*2+c, c*2+c')? No.
    # A is 4x4. Block A_00 is rows 0,1 cols 0,1.
    # Global index for A[i,j] is i*m + j.
    
    def add_strassen_term(block_A, block_B, block_C, term_id, col_idx):
        # term_id 0..6 corresponds to M1..M7
        # block_A, block_B, block_C are tuples (row_offset, col_offset) for 2x2 blocks
        rA, cA = block_A
        rB, cB = block_B
        rC, cC = block_C
        
        # Define U, V, W contributions for each M_k
        # M1: (X00+X11)(Y00+Y11) -> Z00, Z11
        # M2: (X10+X11)Y00 -> Z10, Z11
        # M3: X00(Y01-Y11) -> Z01, Z11
        # M4: X11(Y10-Y00) -> Z00, Z10
        # M5: (X00+X01)Y11 -> Z00, Z01
        # M6: (X10-X00)(Y00+Y01) -> Z11
        # M7: (X01-X11)(Y10+Y11) -> Z00, Z11
        
        # We need to map local 2x2 indices (0,0), (0,1), (1,0), (1,1) to global
        # A global: (rA + r, cA + c)
        # B global: (rB + r, cB + c)
        # C global: (rC + r, cC + c)
        
        def get_A_idx(r, c):
            return (rA + r) * m + (cA + c)
        def get_B_idx(r, c):
            return (rB + r) * p + (cB + c) # Wait, B is m x p. m=4, p=5.
            # B index is j*p + k.
            # Block B is 2x2.
            # Local r, c in 0..1.
            # Global j = rB + r. Global k = cB + c.
            # But cB is offset in p dimension.
            # For 4x4x4, p=4. cB is 0 or 2.
            # For 5th col, we handle separately.
            pass
        def get_C_idx(r, c):
            # C is n x p.
            # W index is k*n + i.
            # k is row of C? No, C[i,k].
            # W[k*n + i].
            # Local r, c in 0..1.
            # Global i = rC + r. Global k = cC + c.
            # W index = (cC + c) * n + (rC + r)
            pass

        # This mapping is getting complex. I will just hardcode the indices for the 56 terms.
        # It's safer.
        
    # I will generate the 56 terms programmatically using the Strassen formulas.
    
    # Strassen coefficients for M1..M7
    # U factors (for A)
    # V factors (for B)
    # W factors (for C)
    
    # M1: A(0,0)+A(1,1), B(0,0)+B(1,1), C(0,0)+C(1,1)
    # M2: A(1,0)+A(1,1), B(0,0), C(1,0)+C(1,1)
    # M3: A(0,0), B(0,1)-B(1,1), C(0,1)+C(1,1)
    # M4: A(1,1), B(1,0)-B(0,0), C(0,0)+C(1,0)
    # M5: A(0,0)+A(0,1), B(1,1), C(0,0)+C(0,1)
    # M6: A(1,0)-A(0,0), B(0,0)+B(0,1), C(1,1)
    # M7: A(0,1)-A(1,1), B(1,0)+B(1,1), C(0,0)+C(1,1)
    
    # Note: C contributions are additive.
    # We need to distribute the W coefficients.
    
    # I will create a list of (U_indices, V_indices, W_indices, coeff)
    # Actually, U, V, W are just 1 or -1.
    
    # Let's define the 7 terms for a generic 2x2 block multiplication.
    # Returns list of (u_vals, v_vals, w_vals)
    # u_vals: dict of (r,c) -> coeff
    # v_vals: dict of (r,c) -> coeff
    # w_vals: dict of (r,c) -> coeff
    
    def get_strassen_terms():
        terms = []
        # M1
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (1,1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(1,0): 1, (1,1): 1}
        v = {(0,0): 1}
        w = {(1,0): 1, (1,1): 1}
        terms.append((u, v, w))
        
        # M3
        u = {(0,0): 1}
        v = {(0,1): 1, (1,1): -1}
        w = {(0,1): 1, (1,1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(1,1): 1}
        v = {(1,0): 1, (0,0): -1}
        w = {(0,0): 1, (1,0): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(0,0): 1, (0,1): 1}
        v = {(1,1): 1}
        w = {(0,0): 1, (0,1): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(1,0): 1, (0,0): -1}
        v = {(0,0): 1, (0,1): 1}
        w = {(1,1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(0,1): 1, (1,1): -1}
        v = {(1,0): 1, (1,1): 1}
        w = {(0,0): 1, (1,1): 1}
        terms.append((u, v, w))
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # Define the 4 block multiplications for C_00, C_01, C_10, C_11
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # Blocks:
    # A_00: r=0, c=0
    # A_01: r=0, c=2
    # A_10: r=2, c=0
    # A_11: r=2, c=2
    
    # B_00: r=0, c=0
    # B_01: r=0, c=2
    # B_10: r=2, c=0
    # B_11: r=2, c=2
    
    # C_00: r=0, c=0
    # C_01: r=0, c=2
    # C_10: r=2, c=0
    # C_11: r=2, c=2
    
    # For each C block, we have two A*B products.
    # We iterate through the 7 Strassen terms for each product.
    
    def add_block_product(A_block, B_block, C_block, col_start):
        nonlocal col
        for term in strassen_terms:
            u_vals, v_vals, w_vals = term
            for (ru, cu), u_coeff in u_vals.items():
                global_i = A_block[0] + ru
                global_j = A_block[1] + cu
                U[global_i * m + global_j, col] = u_coeff
            for (rv, cv), v_coeff in v_vals.items():
                global_j = B_block[0] + rv
                global_k = B_block[1] + cv
                V[global_j * p + global_k, col] = v_coeff
            for (rw, cw), w_coeff in w_vals.items():
                global_i = C_block[0] + rw
                global_k = C_block[1] + cw
                W[global_k * n + global_i, col] = w_coeff
            col += 1
        return col

    # C_00
    col = add_block_product((0,0), (0,0), (0,0), col) # A_00 B_00
    col = add_block_product((0,2), (2,0), (0,0), col) # A_01 B_10
    
    # C_01
    col = add_block_product((0,0), (0,2), (0,2), col) # A_00 B_01
    col = add_block_product((0,2), (2,2), (0,2), col) # A_01 B_11
    
    # C_10
    col = add_block_product((2,0), (0,0), (2,0), col) # A_10 B_00
    col = add_block_product((2,2), (2,0), (2,0), col) # A_11 B_10
    
    # C_11
    col = add_block_product((2,0), (0,2), (2,2), col) # A_10 B_01
    col = add_block_product((2,2), (2,2), (2,2), col) # A_11 B_11
    
    # Add 4 terms for 5th column (k=4)
    # C_i4 = sum_j A_ij B_j4
    # Term i: U[i,j,c]=1, V[j,4,c]=1, W[4,i,c]=1
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
        W[4 * n + i, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
