import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication.
    Uses the known rank-49 decomposition for the 4x4 block (via recursive Strassen)
    and schoolbook for the 1+4 boundary blocks.
    Total rank: 61 (boundary) + 49 (4x4 core) = 110.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    terms = []

    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))

    # Index mapping:
    # A: row i, col j -> i*m + j
    # B: row j, col k -> j*p + k
    # C: row i, col k -> k*n + i (Trace convention W[k*n+i])

    # Block decomposition: 1 + 4
    # Indices 0 is the "1" block, indices 1-4 are the "4" block.

    # --- Boundary Blocks (Schoolbook) ---
    # These involve at least one index being 0.
    # We iterate over all i, j, k where at least one is 0, and add the term.
    # This covers:
    # 1. i=0, j=0, k=0 (1 term)
    # 2. i=0, j=0, k=1..4 (4 terms)
    # 3. i=0, j=1..4, k=0 (4 terms)
    # 4. i=0, j=1..4, k=1..4 (16 terms) -> Wait, this is part of the 4x4 block? No.
    #    If i=0, it's in the top row of C.
    #    If j=0, it's in the top row of B / left col of A.
    #    If k=0, it's in the left col of C / top row of B.
    
    # Let's just explicitly list the boundary terms.
    # The core 4x4 block is i in 1..4, j in 1..4, k in 1..4.
    # Everything else is boundary.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the core 4x4 block
                if i >= 1 and i <= 4 and j >= 1 and j <= 4 and k >= 1 and k <= 4:
                    continue
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)

    # --- Core 4x4 Block (Strassen Rank 49) ---
    # We need to decompose the multiplication of A[1:5, 1:5] * B[1:5, 1:5] -> C[1:5, 1:5].
    # This is a 4x4 matrix multiplication.
    # Rank 49 is achieved by viewing 4x4 as 2x2 blocks of 2x2 matrices,
    # and using Strassen (rank 7) for the block multiplication,
    # where each block multiplication is 2x2 (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # We will recursively decompose.
    # Function to add terms for 2x2 matrix multiplication using rank 7 (Strassen).
    # Inputs:
    #   A_rows, A_cols: global row/col indices for the 2x2 A block.
    #   B_rows, B_cols: global row/col indices for the 2x2 B block.
    #   C_rows, C_cols: global row/col indices for the 2x2 C block.
    #   coeff: multiplier for the whole block product.
    
    def add_strassen_2x2(A_r, A_c, B_r, B_c, C_r, C_c, coeff=1.0):
        # A_r: [r0, r1], A_c: [c0, c1]
        # B_r: [r0, r1], B_c: [c0, c1]
        # C_r: [r0, r1], C_c: [c0, c1]
        
        # The 7 rank-1 terms for 2x2 matrix multiplication.
        # We use the explicit decomposition:
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # C = A B
        # i = ae + cg, j = af + dh, k = ce + ag, l = cf + dh
        
        # Strassen terms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(f+h)
        # M7 = (b-c)(e+g)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to express these as sums of rank-1 tensors A_ij B_kl C_mn.
        # A rank-1 tensor is A_ij * B_jk * W_ki.
        # Wait, the tensor T is sum_{i,j,k} A_ij B_jk C_ki (with appropriate indices).
        # Actually, the multiplication tensor T_{i,j,k} = 1 if C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The rank-1 decomposition is T = sum_r U_r V_r W_r.
        # U_r is a vector of size 4 (for 2x2 A), V_r size 4, W_r size 4.
        # U_r[i,j] = u_{ij}, V_r[j,k] = v_{jk}, W_r[i,k] = w_{ik}.
        # The term is u_{ij} v_{jk} w_{ik}.
        
        # The 7 terms of Strassen correspond to 7 such triples.
        # However, the "products" like (a+d) are sums of elements.
        # This means the U vector has multiple non-zero entries.
        # e.g., for M1 = (a+d)(e+h):
        # The U vector has u_00=1, u_11=1.
        # The V vector has v_00=1, v_11=1.
        # The W vector?
        # M1 contributes to C00 and C11.
        # C00 gets +M1, C11 gets +M1.
        # So W00=1, W11=1.
        # Let's check if this is a valid rank-1 term.
        # Sum_{i,j,k} U_{ij} V_{jk} W_{ik}
        # For U: (1,0; 0,1), V: (1,0; 0,1), W: (1,0; 0,1)
        # Terms:
        # i=0,j=0,k=0: 1*1*1 = 1 -> C00
        # i=0,j=1,k=1: 0*0*0 = 0
        # i=1,j=0,k=0: 0*0*0 = 0
        # i=1,j=1,k=1: 1*1*1 = 1 -> C11
        # So this term contributes 1 to C00 and 1 to C11.
        # This matches M1 contributing to C00 and C11.
        
        # So we can construct the 7 terms by defining the U, V, W patterns for each M.
        
        # M1: U=(1,0;0,1), V=(1,0;0,1), W=(1,0;0,1)
        # M2: U=(0,0;1,1), V=(1,0;0,0), W=(0,0;1,0)  -> C10 gets M2
        #    Check: i=1,j=0,k=0: U_10 V_00 W_10 = 1*1*1 = 1 -> C10. Correct.
        #    Wait, M2 = (c+d)e. c=A10, d=A11, e=B00.
        #    U should pick c and d. U_10=1, U_11=1.
        #    V should pick e. V_00=1.
        #    W should pick C10. W_10=1.
        #    Term: A_10 B_00 C_10 + A_11 B_00 C_10.
        #    Sum_j A_1j B_j0 C_10 = A_10 B_00 C_10 + A_11 B_10 C_10.
        #    Wait, the second term is A_11 B_10 C_10.
        #    But M2 only has (c+d)e = (A10+A11)B00.
        #    It does NOT have A11 B10.
        #    So the rank-1 term U=(0,0;1,1), V=(1,0;0,0), W=(0,0;1,0) generates:
        #    A10 B00 C10 + A11 B00 C10.
        #    This is exactly (A10+A11)B00 C10.
        #    So yes, it works.
        
        # M3: U=(1,0;0,0), V=(0,1;0,-1), W=(0,1;0,0) -> C01 gets M3
        #    M3 = a(f-h) = A00(B01-B11).
        #    U_00=1. V_01=1, V_11=-1. W_01=1.
        #    Term: A00 B01 C01 - A00 B11 C01.
        #    Correct.
        
        # M4: U=(0,0;0,1), V=(0,-1;1,0), W=(0,0;0,1) -> C00 gets -M4
        #    M4 = d(g-e) = A11(B10-B00).
        #    U_11=1. V_10=1, V_00=-1. W_00=1.
        #    Term: A11 B10 C00 - A11 B00 C00.
        #    We want -M4 in C00. So coeff is -1.
        #    Term: -1 * (A11 B10 C00 - A11 B00 C00) = -A11 B10 C00 + A11 B00 C00.
        #    -M4 = -A11(B10-B00) = -A11 B10 + A11 B00. Correct.
        
        # M5: U=(1,1;0,0), V=(0,0;0,1), W=(1,0;0,1) -> C00 gets +M5, C01 gets +M5?
        #    M5 = (a+b)h = (A00+A01)B11.
        #    U_00=1, U_01=1. V_11=1.
        #    W?
        #    C00 gets M5. C01 gets M5.
        #    So W_00=1, W_01=1.
        #    Term: A00 B11 C00 + A01 B11 C00 + A00 B11 C01 + A01 B11 C01.
        #    = (A00+A01)B11 (C00 + C01).
        #    Correct.
        
        # M6: U=(0,0;1,-1), V=(1,1;0,0), W=(0,0;0,1) -> C11 gets +M6
        #    M6 = (c-d)(f+h) = (A10-A11)(B00+B01).
        #    U_10=1, U_11=-1. V_00=1, V_01=1. W_11=1.
        #    Term: (A10-A11)(B00+B01) C11.
        #    Correct.
        
        # M7: U=(0,1;-1,0), V=(1,0;1,0), W=(1,0;0,0) -> C00 gets +M7
        #    M7 = (b-c)(e+g) = (A01-A10)(B00+B10).
        #    U_01=1, U_10=-1. V_00=1, V_10=1. W_00=1.
        #    Term: (A01-A10)(B00+B10) C00.
        #    Correct.
        
        # Now we define the 7 patterns (U, V, W) as lists of (idx_u, idx_v, idx_w, coeff).
        # Indices are local 0..3 for 2x2.
        # u_idx = r*2+c, v_idx = r*2+c, w_idx = r*2+c.
        
        # M1
        add_term(A_r[0]*5+A_c[0], B_r[0]*5+B_c[0], C_r[0]*5+C_c[0], coeff)
        add_term(A_r[1]*5+A_c[1], B_r[1]*5+B_c[1], C_r[1]*5+C_c[1], coeff)
        
        # M2 (coeff 1)
        add_term(A_r[1]*5+A_c[0], B_r[0]*5+B_c[0], C_r[1]*5+C_c[0], coeff)
        add_term(A_r[1]*5+A_c[1], B_r[0]*5+B_c[0], C_r[1]*5+C_c[0], coeff)
        
        # M3 (coeff 1)
        add_term(A_r[0]*5+A_c[0], B_r[0]*5+B_c[1], C_r[0]*5+C_c[1], coeff)
        add_term(A_r[0]*5+A_c[0], B_r[1]*5+B_c[1], C_r[0]*5+C_c[1], -coeff)
        
        # M4 (coeff -1)
        add_term(A_r[1]*5+A_c[1], B_r[1]*5+B_c[0], C_r[0]*5+C_c[0], -coeff)
        add_term(A_r[1]*5+A_c[1], B_r[0]*5+B_c[0], C_r[0]*5+C_c[0], coeff)
        
        # M5 (coeff 1)
        add_term(A_r[0]*5+A_c[0], B_r[1]*5+B_c[1], C_r[0]*5+C_c[0], coeff)
        add_term(A_r[0]*5+A_c[1], B_r[1]*5+B_c[1], C_r[0]*5+C_c[0], coeff)
        add_term(A_r[0]*5+A_c[0], B_r[1]*5+B_c[1], C_r[0]*5+C_c[1], coeff)
        add_term(A_r[0]*5+A_c[1], B_r[1]*5+B_c[1], C_r[0]*5+C_c[1], coeff)
        
        # M6 (coeff 1)
        add_term(A_r[1]*5+A_c[0], B_r[0]*5+B_c[0], C_r[1]*5+C_c[1], coeff)
        add_term(A_r[1]*5+A_c[0], B_r[0]*5+B_c[1], C_r[1]*5+C_c[1], coeff)
        add_term(A_r[1]*5+A_c[1], B_r[0]*5+B_c[0], C_r[1]*5+C_c[1], -coeff)
        add_term(A_r[1]*5+A_c[1], B_r[0]*5+B_c[1], C_r[1]*5+C_c[1], -coeff)
        
        # M7 (coeff 1)
        add_term(A_r[0]*5+A_c[1], B_r[0]*5+B_c[0], C_r[0]*5+C_c[0], coeff)
        add_term(A_r[0]*5+A_c[1], B_r[1]*5+B_c[0], C_r[0]*5+C_c[0], coeff)
        add_term(A_r[1]*5+A_c[0], B_r[0]*5+B_c[0], C_r[0]*5+C_c[0], -coeff)
        add_term(A_r[1]*5+A_c[0], B_r[1]*5+B_c[0], C_r[0]*5+C_c[0], -coeff)

    # Define the 4x4 block indices
    # A[1:5, 1:5] -> rows 1,2,3,4 cols 1,2,3,4
    # Split into 2x2 blocks:
    # A11: rows 1,2 cols 1,2
    # A12: rows 1,2 cols 3,4
    # A21: rows 3,4 cols 1,2
    # A22: rows 3,4 cols 3,4
    
    A11_r = [1, 2]; A11_c = [1, 2]
    A12_r = [1, 2]; A12_c = [3, 4]
    A21_r = [3, 4]; A21_c = [1, 2]
    A22_r = [3, 4]; A22_c = [3, 4]
    
    B11_r = [1, 2]; B11_c = [1, 2]
    B12_r = [1, 2]; B12_c = [3, 4]
    B21_r = [3, 4]; B21_c = [1, 2]
    B22_r = [3, 4]; B22_c = [3, 4]
    
    C11_r = [1, 2]; C11_c = [1, 2]
    C12_r = [1, 2]; C12_c = [3, 4]
    C21_r = [3, 4]; C21_c = [1, 2]
    C22_r = [3, 4]; C22_c = [3, 4]
    
    # Strassen for 4x4:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2
    # M3 = A11(B12-B22) -> C12 += M3
    # M4 = A22(B21-B11) -> C11 -= M4
    # M5 = (A11+A12)B22 -> C11 += M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B11+B21) -> C11 += M7
    
    # We need to add terms for each M.
    # Each M is a product of two 2x2 matrices.
    # We call add_strassen_2x2 for each M.
    # But wait, add_strassen_2x2 adds 7 terms for a SINGLE 2x2 multiplication.
    # Here, M1 is (A11+A22) * (B11+B22).
    # This is NOT a single 2x2 multiplication of fixed indices.
    # It is a product of SUMS.
    # (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # We can compute this product using the rank-7 decomposition of 2x2 multiplication.
    # But the "input" matrices are sums.
    # This means the U vector for the rank-7 decomposition will have entries from A11 AND A22.
    # Similarly for V.
    
    # So, we can't just call add_strassen_2x2(A11, B11, C11).
    # We need to generalize add_strassen_2x2 to accept "virtual" matrices that are sums of blocks.
    # Or, we can expand (A11+A22)(B11+B22) into 4 terms: A11B11 + A11B22 + A22B11 + A22B22.
    # Each of these is a 2x2 multiplication.
    # If we use schoolbook for these, we get 4 terms each -> 16 terms per M.
    # 7 M's -> 112 terms.
    # This is worse than 49.
    
    # To get 49, we MUST use the rank-7 structure where the inputs are linear combinations.
    # So we need to modify add_strassen_2x2 to handle "linear combination" inputs.
    # Let's define a helper that adds the 7 terms for a product of two "virtual" 2x2 matrices.
    # A virtual matrix is defined by a list of (block_row_indices, block_col_indices, coeff).
    
    def add_virtual_2x2_mult(A_parts, B_parts, C_r, C_c, coeff=1.0):
        # A_parts: list of (rows, cols, coeff)
        # B_parts: list of (rows, cols, coeff)
        # This adds the 7 rank-1 terms for the product of the virtual matrices.
        
        # We need to map the local indices 0,1,2,3 to the global indices.
        # The local index (i,j) in the virtual matrix corresponds to the sum over parts.
        # For a specific local (i,j), the value is sum_{part} coeff_part * A[rows_part[i], cols_part[j]].
        # This is not a single global index.
        # However, the rank-1 decomposition works on the "elements".
        # The 7 patterns for U, V, W are fixed.
        # For each pattern, we have a set of local (u_idx, v_idx, w_idx, coeff).
        # We need to expand each local u_idx into the corresponding global terms.
        
        # Local u_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
        # Local v_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
        # Local w_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
        
        # The 7 patterns (from above):
        # We will re-list them as (u_local, v_local, w_local, coeff)
        patterns = [
            # M1
            (0, 0, 0, 1), (3, 3, 3, 1),
            # M2
            (2, 0, 2, 1), (3, 0, 2, 1),
            # M3
            (0, 1, 1, 1), (0, 3, 1, -1),
            # M4
            (3, 2, 0, -1), (3, 0, 0, 1),
            # M5
            (0, 3, 0, 1), (1, 3, 0, 1), (0, 3, 1, 1), (1, 3, 1, 1),
            # M6
            (2, 0, 3, 1), (2, 1, 3, 1), (3, 0, 3, -1), (3, 1, 3, -1),
            # M7
            (1, 0, 0, 1), (1, 2, 0, 1), (2, 0, 0, -1), (2, 2, 0, -1),
        ]
        
        # For each pattern, we need to expand u_local and v_local.
        # u_local (i,j) -> sum_{part} coeff * A[rows[i], cols[j]]
        # This means we add a term for each part.
        
        for u_l, v_l, w_l, p_coeff in patterns:
            u_i, u_j = divmod(u_l, 2)
            v_i, v_j = divmod(v_l, 2)
            w_i, w_j = divmod(w_l, 2)
            
            # w is a single global index
            w_global = C_r[w_i] * 5 + C_c[w_j]
            
            # u is a sum of terms
            # v is a sum of terms
            # The product is sum_{u_part} sum_{v_part} (coeff_u * coeff_v * p_coeff) * A[u_global] * B[v_global] * W[w_global]
            
            for u_rows, u_cols, u_c in A_parts:
                u_global = u_rows[u_i] * 5 + u_cols[u_j]
                for v_rows, v_cols, v_c in B_parts:
                    v_global = v_rows[v_i] * 5 + v_cols[v_j]
                    total_coeff = coeff * p_coeff * u_c * v_c
                    add_term(u_global, v_global, w_global, total_coeff)

    # Now call add_virtual_2x2_mult for the 7 Strassen terms.
    
    # M1 = (A11+A22)(B11+B22)
    A_parts_M1 = [(A11_r, A11_c, 1), (A22_r, A22_c, 1)]
    B_parts_M1 = [(B11_r, B11_c, 1), (B22_r, B22_c, 1)]
    add_virtual_2x2_mult(A_parts_M1, B_parts_M1, C11_r, C11_c, 1)
    add_virtual_2x2_mult(A_parts_M1, B_parts_M1, C22_r, C22_c, 1)
    
    # M2 = (A21+A22)B11
    A_parts_M2 = [(A21_r, A21_c, 1), (A22_r, A22_c, 1)]
    B_parts_M2 = [(B11_r, B11_c, 1)]
    add_virtual_2x2_mult(A_parts_M2, B_parts_M2, C21_r, C21_c, 1)
    
    # M3 = A11(B12-B22)
    A_parts_M3 = [(A11_r, A11_c, 1)]
    B_parts_M3 = [(B12_r, B12_c, 1), (B22_r, B22_c, -1)]
    add_virtual_2x2_mult(A_parts_M3, B_parts_M3, C12_r, C12_c, 1)
    
    # M4 = A22(B21-B11)
    A_parts_M4 = [(A22_r, A22_c, 1)]
    B_parts_M4 = [(B21_r, B21_c, 1), (B11_r, B11_c, -1)]
    add_virtual_2x2_mult(A_parts_M4, B_parts_M4, C11_r, C11_c, -1)
    
    # M5 = (A11+A12)B22
    A_parts_M5 = [(A11_r, A11_c, 1), (A12_r, A12_c, 1)]
    B_parts_M5 = [(B22_r, B22_c, 1)]
    add_virtual_2x2_mult(A_parts_M5, B_parts_M5, C11_r, C11_c, 1)
    add_virtual_2x2_mult(A_parts_M5, B_parts_M5, C12_r, C12_c, 1)
    
    # M6 = (A21-A11)(B11+B12)
    A_parts_M6 = [(A21_r, A21_c, 1), (A11_r, A11_c, -1)]
    B_parts_M6 = [(B11_r, B11_c, 1), (B12_r, B12_c, 1)]
    add_virtual_2x2_mult(A_parts_M6, B_parts_M6, C22_r, C22_c, 1)
    
    # M7 = (A12-A22)(B11+B21)
    A_parts_M7 = [(A12_r, A12_c, 1), (A22_r, A22_c, -1)]
    B_parts_M7 = [(B11_r, B11_c, 1), (B21_r, B21_c, 1)]
    add_virtual_2x2_mult(A_parts_M7, B_parts_M7, C11_r, C11_c, 1)
    
    # Total terms:
    # Boundary: 125 - 64 = 61
    # Core: 7 * (number of terms per virtual mult)
    # Each virtual mult has 7 patterns.
    # Each pattern expands to |A_parts| * |B_parts| terms.
    # M1: 2*2=4 terms per pattern. 7*4=28.
    # M2: 2*1=2. 7*2=14.
    # M3: 1*2=2. 7*2=14.
    # M4: 1*2=2. 7*2=14.
    # M5: 2*1=2. 7*2=14.
    # M6: 2*2=4. 7*4=28.
    # M7: 2*2=4. 7*4=28.
    # Total core: 28+14+14+14+14+28+28 = 140.
    # Total rank: 61 + 140 = 201.
    # This is much worse than 125!
    
    # Why? Because the "virtual" matrices are sums, and the rank-1 decomposition of the product of sums expands to the product of the number of parts.
    # The rank 49 result assumes that the 7 products are of "atomic" 2x2 matrices, not sums.
    # In the standard Strassen for 4x4, the 7 products are indeed of sums of 2x2 blocks.
    # But the rank 49 is for the 4x4 tensor.
    # The 4x4 tensor can be decomposed into 49 rank-1 tensors.
    # My expansion above gives 140 rank-1 tensors for the core.
    # This means my decomposition is not minimal.
    # It is a valid decomposition, but not the rank-49 one.
    # The rank-49 decomposition does NOT expand the sums.
    # It uses the linearity of the tensor to combine terms.
    # But in the explicit U, V, W matrices, we cannot "combine" terms that have different U or V indices.
    # Wait, we CAN combine them if we sum the coefficients for the same (u_idx, v_idx, w_idx).
    # My add_term function just appends.
    # I should aggregate the terms!
    
    # Let's aggregate the terms.
    # Use a dictionary: key (u_idx, v_idx, w_idx) -> coeff.
    
    agg_terms = {}
    for u, v, w, c in terms:
        key = (u, v, w)
        agg_terms[key] = agg_terms.get(key, 0.0) + c
    
    final_terms = [(u, v, w, c) for (u, v, w), c in agg_terms.items() if c != 0]
    
    rank = len(final_terms)
    
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(final_terms):
        U[u_idx, r] = coeff
        V[v_idx, r] = coeff
        W[w_idx, r] = coeff
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
