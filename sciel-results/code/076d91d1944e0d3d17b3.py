"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49  # 7 * 7
    
    # Initialize U, V, W
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 definitions
    # For 2x2 matrices A, B:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # These are coefficients for A and B elements in the 7 products.
    # A indices: 00, 01, 10, 11 -> local indices 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> local indices 0, 1, 2, 3
    
    # strassen_coeffs[prod_idx][A_idx] and strassen_coeffs[prod_idx][B_idx]
    # prod_idx 0..6 corresponds to M1..M7
    # A_idx 0..3 corresponds to 00, 01, 10, 11
    # B_idx 0..3 corresponds to 00, 01, 10, 11
    
    A_coeffs = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # Result C indices for each M:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # coeff_C[prod_idx][C_idx]
    # C_idx 0..3 corresponds to 00, 01, 10, 11
    C_coeffs = [
        [1, 0, 0, 1], # M1 contributes to C00 (+1) and C11 (+1)
        [0, 0, 1, -1], # M2 contributes to C10 (+1) and C11 (-1)
        [0, 1, 0, 1], # M3 contributes to C01 (+1) and C11 (+1)
        [1, 0, 1, 0], # M4 contributes to C00 (+1) and C10 (+1)
        [-1, 1, 0, 0], # M5 contributes to C00 (-1) and C01 (+1)
        [0, 0, 0, 1], # M6 contributes to C11 (+1)
        [1, 0, 0, 0]  # M7 contributes to C00 (+1)
    ]
    
    # We are doing 4x4 = 2x2 blocks of 2x2.
    # Top level: 7 products (T0..T6)
    # Bottom level: 7 products (S0..S6) for each top product.
    # Total rank = 49.
    
    # Global indices for 4x4 matrix:
    # i, j in 0..3
    # Block row br = i // 2, block col bc = j // 2
    # Local row lr = i % 2, local col lc = j % 2
    
    # For U (A matrix):
    # Element A[i, j] is at U[i*m + j]
    # We need to map local (lr, lc) in a specific block (br, bc) to global index.
    # Global i = br * 2 + lr
    # Global j = bc * 2 + lc
    # U index = i * m + j = (br*2 + lr) * 4 + (bc*2 + lc)
    
    # For V (B matrix):
    # Element B[j, k] is at V[j*p + k]
    # Global j = bc * 2 + lc (from A's block col? No, B's row index is A's col index)
    # Wait, A is n x m, B is m x p.
    # A[i, j] * B[j, k].
    # In 2x2 block decomposition:
    # A_{br, bc} * B_{bc, bk}
    # Local indices:
    # A element (lr, lc) in block (br, bc) -> global i = br*2+lr, j = bc*2+lc
    # B element (lc, lk) in block (bc, bk) -> global j = bc*2+lc, k = bk*2+lk
    # Note: The 'lc' from A must match the 'lc' (row of B) for the multiplication to be valid within the block product?
    # Actually, Strassen operates on the whole 2x2 blocks.
    # The 7 products are sums of terms like A_{block}(lr1, lc1) * B_{block}(lc2, lk2).
    # The linear combination is applied to the 2x2 blocks.
    # So for a specific Strassen product term (e.g. M1 = (A00+A11)(B00+B11)), we have 4 scalar products:
    # A00*B00, A00*B11, A11*B00, A11*B11.
    # These correspond to local indices (0,0) with (0,0), (0,0) with (0,1), etc.
    
    # Let's iterate over the 49 total terms.
    term_id = 0
    for top_m in range(7):
        # Top level Strassen term M_top
        # This M_top is a 2x2 matrix result from multiplying 2x2 blocks.
        # M_top = sum_{lr, lc, lk} (sum_{lr_A} A_top_coeff * A[br, bc, lr_A, lc]) * (sum_{lk_B} B_top_coeff * B[bc, bk, lc, lk_B])
        # Wait, the Strassen formula is linear.
        # M_top = (A_block_linear_combo) * (B_block_linear_combo)
        # This expands to sum of products of individual elements.
        
        # For each top-level product top_m:
        # We have 7 bottom-level products bottom_s (corresponding to the 7 Strassen terms for the 2x2 sub-blocks).
        # Actually, it's simpler:
        # The 49 terms are the scalar products forming the 7 top-level 2x2 matrices.
        # Each top-level 2x2 matrix is formed by 7 scalar products (Strassen).
        
        for bottom_s in range(7):
            # This term corresponds to one scalar product in the computation of the top-level matrix M_top.
            # Specifically, M_top is a 2x2 matrix. Its elements are linear combinations of the 7 bottom-level scalar products?
            # No. M_top is computed using Strassen on the 2x2 blocks.
            # The 7 bottom-level products are the "scalar" products of the 2x2 blocks? No, the 2x2 blocks are matrices.
            # Strassen replaces 8 matrix multiplications with 7.
            # Here, the "matrices" being multiplied are 2x2.
            # So M_top = Strassen(A_block, B_block).
            # The result M_top is a 2x2 matrix.
            # The entries of M_top are linear combinations of the products of entries of A_block and B_block.
            # There are 7 such "grouped" products.
            # Let's call the 7 intermediate values for the top-level block multiplication as S_0..S_6.
            # Each S_k is a 2x2 matrix? No, Strassen's 7 products are matrices if the inputs are matrices.
            # But here inputs to the top-level Strassen are 2x2 matrices (blocks of the 4x4).
            # So S_k are 2x2 matrices.
            # Then C_block = combination of S_k.
            
            # However, we need a rank-49 decomposition for the 4x4.
            # This means we decompose the tensor into 49 rank-1 tensors.
            # Each rank-1 tensor corresponds to a product of one element of A, one of B, and one of C.
            # The Strassen algorithm for 4x4 uses 7 "steps" at the top level.
            # Each top-level step involves multiplying two 2x2 blocks.
            # That multiplication uses 7 scalar products (if we view the 2x2 blocks as scalars? No, that's not right).
            
            # Let's re-evaluate.
            # Rank of 2x2 mult is 7.
            # Rank of 4x4 mult using Strassen is 7 * Rank(2x2) = 7 * 7 = 49.
            # The 49 terms are:
            # For each of the 7 top-level Strassen products (which are 2x2 matrix multiplications effectively, or rather, the results of the 7 Strassen formulas applied to the blocks):
            # Wait, the 7 top-level products are $M_1, \dots, M_7$.
            # $M_k = \text{LinearCombo}(A_{blocks}) \cdot \text{LinearCombo}(B_{blocks})$.
            # Since the "scalars" in this level are 2x2 matrices, $M_k$ is a 2x2 matrix.
            # To compute $M_k$ (a 2x2 matrix), we need to multiply two 2x2 matrices.
            # This requires 7 scalar products (the entries of the 2x2 matrices).
            # So for each $k \in \{1..7\}$, we have 7 scalar products.
            # Total 49.
            
            # So, for top_m (0..6):
            # We compute the 2x2 matrix $M_{top\_m}$.
            # $M_{top\_m}$ is computed by multiplying two 2x2 matrices $X$ and $Y$.
            # $X$ is a linear combination of A's 2x2 blocks.
            # $Y$ is a linear combination of B's 2x2 blocks.
            # The multiplication $X \cdot Y$ is done via Strassen (7 products).
            # Let the 7 products for this specific top_m be $P_0, \dots, P_6$.
            # Each $P_s$ is a scalar product of an element of $X$ and an element of $Y$.
            # Element of $X$ is a linear combination of A elements.
            # Element of $Y$ is a linear combination of B elements.
            
            # So, for a specific (top_m, bottom_s):
            # We identify which element of X and Y are being multiplied.
            # This determines the local indices in the 2x2 blocks.
            # Then we expand the linear combinations to global indices.
            
            # Strassen 2x2 products (bottom level):
            # S0 = (X00 + X11) * (Y00 + Y11)
            # S1 = (X10 + X11) * Y00
            # S2 = X00 * (Y01 - Y11)
            # S3 = X11 * (Y10 - Y00)
            # S4 = (X00 + X01) * Y11
            # S5 = (X10 - X00) * (Y00 + Y01)
            # S6 = (X01 - X11) * (Y10 + Y11)
            
            # We need the coefficients for X and Y for each bottom_s.
            # X coeffs (same as A_coeffs in previous def, but for X)
            # Y coeffs (same as B_coeffs in previous def, but for Y)
            
            # X element indices (local 2x2): 00, 01, 10, 11
            # Y element indices (local 2x2): 00, 01, 10, 11
            
            # Let's define the local indices for X and Y for each bottom_s.
            # bottom_s 0: X=(0,0),(1,1); Y=(0,0),(1,1)
            # bottom_s 1: X=(1,0),(1,1); Y=(0,0)
            # ...
            
            # Instead of hardcoding, we can use the A_coeffs and B_coeffs defined earlier.
            # A_coeffs[bottom_s] gives coeffs for X elements (0,1,2,3).
            # B_coeffs[bottom_s] gives coeffs for Y elements (0,1,2,3).
            
            # Now, X is a 2x2 matrix which is a linear combination of A's blocks.
            # X = sum_{br, bc} coeff_X[br, bc] * A[br, bc]
            # Where coeff_X depends on top_m.
            # From A_coeffs[top_m], we have coeffs for the 4 blocks of A.
            # A_coeffs[top_m][0] -> A[0,0] (br=0, bc=0)
            # A_coeffs[top_m][1] -> A[0,1] (br=0, bc=1)
            # A_coeffs[top_m][2] -> A[1,0] (br=1, bc=0)
            # A_coeffs[top_m][3] -> A[1,1] (br=1, bc=1)
            
            # Similarly for Y:
            # B_coeffs[top_m][0] -> B[0,0] (bc=0, bk=0)
            # B_coeffs[top_m][1] -> B[0,1] (bc=0, bk=1)
            # B_coeffs[top_m][2] -> B[1,0] (bc=1, bk=0)
            # B_coeffs[top_m][3] -> B[1,1] (bc=1, bk=1)
            
            # So, for a specific (top_m, bottom_s):
            # X is a 2x2 matrix.
            # X[lr, lc] = sum_{br, bc} A_block_coeff[top_m][br*2+bc] * A[br*2+lr, bc*2+lc]
            # Wait, A_coeffs[top_m] has 4 entries.
            # Entry 0: A[0,0] block (br=0, bc=0)
            # Entry 1: A[0,1] block (br=0, bc=1)
            # Entry 2: A[1,0] block (br=1, bc=0)
            # Entry 3: A[1,1] block (br=1, bc=1)
            
            # So X[lr, lc] = sum_{b_idx=0..3} A_coeffs[top_m][b_idx] * A[br(b_idx)*2+lr, bc(b_idx)*2+lc]
            
            # Similarly Y[lc, lk] = sum_{b_idx=0..3} B_coeffs[top_m][b_idx] * B[bc(b_idx)*2+lc, bk(b_idx)*2+lk]
            # Note: Y's row index is the block col of A (bc).
            
            # The product term for this (top_m, bottom_s) is:
            # P = sum_{lr, lc, lk} (X_local_coeff[bottom_s][lr, lc] * Y_local_coeff[bottom_s][lc, lk]) * X[lr, lc] * Y[lc, lk]
            # Wait, X[lr, lc] and Y[lc, lk] are linear combinations.
            # The product is a sum of products of A elements and B elements.
            # We need to distribute this into the U, V, W matrices.
            # Each scalar product A[i,j] * B[j,k] contributes to C[i,k].
            
            # Let's expand:
            # Term = (sum_{a_br, a_bc} cA[top_m][a_br, a_bc] * A[a_br*2+lr, a_bc*2+lc]) * 
            #        (sum_{b_bc, b_bk} cB[top_m][b_bc, b_bk] * B[b_bc*2+lc, b_bk*2+lk])
            # Note: The 'lc' must match for the multiplication to be defined in the block structure?
            # Yes, X is a 2x2 matrix, Y is a 2x2 matrix.
            # X[lr, lc] * Y[lc, lk] is the standard matrix multiplication.
            # The indices lr, lc, lk are 0 or 1.
            
            # So for fixed top_m, bottom_s:
            # We sum over lr in 0..1, lc in 0..1, lk in 0..1.
            # For each such triple:
            # x_coeff = A_coeffs[bottom_s][lr*2+lc]  # Coeff for X[lr, lc] in Strassen formula
            # y_coeff = B_coeffs[bottom_s][lc*2+lk]  # Coeff for Y[lc, lk] in Strassen formula
            # If x_coeff == 0 or y_coeff == 0, skip.
            
            # Then we expand the linear combos for A and B.
            # A part: sum_{a_br, a_bc} A_coeffs[top_m][a_br*2+a_bc] * A[a_br*2+lr, a_bc*2+lc]
            # B part: sum_{b_bc, b_bk} B_coeffs[top_m][b_bc*2+b_bk] * B[b_bc*2+lc, b_bk*2+lk]
            
            # The total contribution to the term is:
            # sum_{a_br, a_bc, b_bc, b_bk} 
            #   (x_coeff * A_coeffs[top_m][a_br*2+a_bc]) * 
            #   (y_coeff * B_coeffs[top_m][b_bc*2+b_bk]) *
            #   A[a_br*2+lr, a_bc*2+lc] * B[b_bc*2+lc, b_bk*2+lk]
            
            # Each such non-zero product corresponds to a rank-1 tensor.
            # But wait, the rank is 49, not 49 * 8 (number of lr,lc,lk and block combos).
            # This means the 49 terms are the "atomic" products.
            # My expansion above generates many products.
            # The Strassen algorithm combines them.
            # The "rank" refers to the number of multiplications.
            # Each multiplication is of a linear form of A and a linear form of B.
            # So for a fixed (top_m, bottom_s, lr, lc, lk) where coeffs are non-zero,
            # we have ONE multiplication:
            # L_A = sum_{a_br, a_bc} (x_coeff * A_coeffs[top_m][a_br*2+a_bc]) * A[a_br*2+lr, a_bc*2+lc]
            # L_B = sum_{b_bc, b_bk} (y_coeff * B_coeffs[top_m][b_bc*2+b_bk]) * B[b_bc*2+lc, b_bk*2+lk]
            # Product P = L_A * L_B
            # This P contributes to C.
            # Where does it contribute?
            # The result of the top-level Strassen product M_top_m is a 2x2 matrix.
            # The bottom-level product P is one of the 7 values used to compute M_top_m.
            # Specifically, P is a scalar.
            # M_top_m[cr, cc] = sum_{s=0..6} C_coeffs[top_m][cr*2+cc] * (Result of bottom product s at [cr, cc]?)
            # No.
            # M_top_m is a 2x2 matrix.
            # It is computed as:
            # M_top_m[0,0] = S0 + S3 - S4 + S6
            # M_top_m[0,1] = S2 + S4
            # M_top_m[1,0] = S1 + S3
            # M_top_m[1,1] = S0 - S1 + S2 + S5
            # Where S0..S6 are the 7 scalar products from the bottom-level Strassen.
            
            # So, for a fixed (top_m, bottom_s):
            # We have a scalar P = L_A * L_B.
            # This P contributes to M_top_m[cr, cc] with coefficient C_coeffs[bottom_s][cr*2+cc].
            # Note: C_coeffs here is for the bottom level (2x2 multiplication).
            # Let's call it C_bottom_coeffs[bottom_s][cr*2+cc].
            
            # Then, M_top_m is a 2x2 matrix.
            # The final C is computed from M_top_0..M_top_6.
            # C[br, bc] (2x2 block) = sum_{m=0..6} C_top_coeffs[m][br*2+bc] * M_top_m
            # Where C_top_coeffs is the C_coeffs defined earlier for the top level.
            
            # So, the scalar P contributes to C[br, bc, cr, cc] with coefficient:
            # Coeff = C_top_coeffs[top_m][br*2+bc] * C_bottom_coeffs[bottom_s][cr*2+cc]
            
            # Global C index:
            # i = br*2 + cr
            # k = bc*2 + cc
            # W index = k*n + i
            
            # So, for each (top_m, bottom_s, lr, lc, lk) where x_coeff and y_coeff are non-zero:
            # We define L_A and L_B.
            # This is ONE rank-1 term.
            # We add the coefficients to U and V.
            # And we add the product of C coefficients to W.
            
            # Wait, is this 49 terms?
            # top_m: 7
            # bottom_s: 7
            # lr, lc, lk: 2*2*2 = 8
            # Total 392 iterations?
            # But many coeffs are zero.
            # Let's check the number of non-zero terms.
            # For a fixed (top_m, bottom_s), how many (lr, lc, lk) have non-zero x_coeff and y_coeff?
            # x_coeff = A_coeffs[bottom_s][lr*2+lc]
            # y_coeff = B_coeffs[bottom_s][lc*2+lk]
            # For a fixed bottom_s, A_coeffs has 2 non-zeros, B_coeffs has 2 non-zeros.
            # So there are at most 2*2 = 4 non-zero (lr, lc, lk) combos?
            # Let's check bottom_s=0:
            # A_coeffs: [1, 0, 0, 1] -> non-zero at 0 (0,0) and 3 (1,1).
            # B_coeffs: [1, 0, 0, 1] -> non-zero at 0 (0,0) and 3 (1,1).
            # Possible (lr, lc) for A: (0,0) and (1,1).
            # Possible (lc, lk) for B: (0,0) and (1,1).
            # To match lc:
            # If A is (0,0), lc=0. B must have lc=0. B(0,0) is valid. lk=0.
            # If A is (1,1), lc=1. B must have lc=1. B(1,1) is valid. lk=1.
            # So 2 terms for bottom_s=0.
            
            # Let's check bottom_s=1:
            # A: [0, 0, 1, 1] -> (1,0) and (1,1).
            # B: [1, 0, 0, 0] -> (0,0).
            # A(1,0) -> lc=0. B(0,0) -> lc=0. Match. lk=0.
            # A(1,1) -> lc=1. B(0,0) -> lc=0. No match.
            # So 1 term for bottom_s=1.
            
            # It seems the number of terms varies.
            # But the total rank is 49.
            # This implies that each (top_m, bottom_s) pair corresponds to EXACTLY ONE rank-1 term?
            # No, that would be 49 terms.
            # But my analysis shows multiple (lr, lc, lk) for some s.
            # This means my mapping of "rank-1 term" is too fine-grained.
            # A rank-1 term is a product of a linear form of A and a linear form of B.
            # The linear form L_A can involve multiple A elements.
            # In my expansion, for a fixed (top_m, bottom_s, lr, lc, lk), L_A is a sum of A elements.
            # Is it possible that for a fixed (top_m, bottom_s), the sum over (lr, lc, lk) combines into a single product?
            # No, because the C coefficients depend on cr, cc which depend on lr, lc, lk?
            # No, cr, cc are determined by the bottom_s and the position in M_top_m.
            # The product P is a scalar.
            # It contributes to multiple C elements.
            # So P is a single scalar value.
            # It is computed as L_A * L_B.
            # So for a fixed (top_m, bottom_s), is there only one (lr, lc, lk)?
            # Let's re-read Strassen.
            # S0 = (A00 + A11) * (B00 + B11).
            # This is ONE product of two linear forms.
            # The linear form for A is A00 + A11.
            # The linear form for B is B00 + B11.
            # This corresponds to lr,lc for A being (0,0) AND (1,1).
            # So it's not a single (lr, lc).
            # My previous loop structure was iterating over lr, lc, lk as if they were atomic.
            # But Strassen groups them.
            
            # Correct approach:
            # For each (top_m, bottom_s):
            # Define L_A = sum_{lr, lc} A_coeffs[bottom_s][lr*2+lc] * X[lr, lc]
            # Define L_B = sum_{lc, lk} B_coeffs[bottom_s][lc*2+lk] * Y[lc, lk]
            # Note: X[lr, lc] and Y[lc, lk] are themselves linear forms of A and B blocks.
            # So L_A is a linear form of A.
            # L_B is a linear form of B.
            # The product P = L_A * L_B is one rank-1 term.
            # We add the coefficients of L_A to U.
            # We add the coefficients of L_B to V.
            # We add the coefficients of P to W.
            
            # How many such (top_m, bottom_s) pairs are there?
            # 7 * 7 = 49.
            # So this matches the rank.
            
            # Let's implement this.
            
            # 1. Construct L_A and L_B for this (top_m, bottom_s).
            # L_A = sum_{lr, lc} cA_local[bottom_s][lr*2+lc] * X[lr, lc]
            # X[lr, lc] = sum_{a_br, a_bc} cA_top[top_m][a_br*2+a_bc] * A[a_br*2+lr, a_bc*2+lc]
            
            # So L_A = sum_{lr, lc, a_br, a_bc} cA_local[bottom_s][lr*2+lc] * cA_top[top_m][a_br*2+a_bc] * A[a_br*2+lr, a_bc*2+lc]
            
            # Similarly for L_B.
            
            # We accumulate coefficients for U and V.
            
            # For W:
            # P contributes to C[i, k] where i = br*2+cr, k = bc*2+cc.
            # Coeff in W[k*n+i] += cC_top[top_m][br*2+bc] * cC_bottom[bottom_s][cr*2+cc]
            # Wait, the indices br, bc, cr, cc are fixed by top_m and bottom_s?
            # No.
            # M_top_m is a 2x2 matrix.
            # Its elements M_top_m[cr, cc] are linear combinations of S_0..S_6.
            # S_bottom_s is one of the S's.
            # So S_bottom_s contributes to M_top_m[cr, cc] with coeff cC_bottom[bottom_s][cr*2+cc].
            # Then M_top_m contributes to C_block[br, bc] with coeff cC_top[top_m][br*2+bc].
            # So the total coeff for C[br*2+cr, bc*2+cc] is cC_top[top_m][br*2+bc] * cC_bottom[bottom_s][cr*2+cc].
            
            # We need to add this to W for all br, bc, cr, cc where the coeff is non-zero.
            
            # Let's code this.
            
            # Get coefficients
            cA_top = A_coeffs[top_m]
            cB_top = B_coeffs[top_m]
            cA_bot = A_coeffs[bottom_s]
            cB_bot = B_coeffs[bottom_s]
            cC_top = C_coeffs[top_m]
            cC_bot = C_coeffs[bottom_s]
            
            # Construct L_A coeffs
            # U index: i*m + j = (a_br*2+lr)*4 + (a_bc*2+lc)
            # Coeff: cA_bot[lr*2+lc] * cA_top[a_br*2+a_bc]
            
            # Construct L_B coeffs
            # V index: j*p + k = (b_bc*2+lc)*4 + (b_bk*2+lk)
            # Coeff: cB_bot[lc*2+lk] * cB_top[b_bc*2+b_bk]
            
            # Accumulate to U and V
            for lr in range(2):
                for lc in range(2):
                    if cA_bot[lr*2+lc] == 0:
                        continue
                    for a_br in range(2):
                        for a_bc in range(2):
                            coeff_A = cA_bot[lr*2+lc] * cA_top[a_br*2+a_bc]
                            if coeff_A == 0:
                                continue
                            i = a_br * 2 + lr
                            j = a_bc * 2 + lc
                            u_idx = i * m + j
                            U[u_idx][term_id] += coeff_A
            
            for lc in range(2):
                for lk in range(2):
                    if cB_bot[lc*2+lk] == 0:
                        continue
                    for b_bc in range(2):
                        for b_bk in range(2):
                            coeff_B = cB_bot[lc*2+lk] * cB_top[b_bc*2+b_bk]
                            if coeff_B == 0:
                                continue
                            j = b_bc * 2 + lc
                            k = b_bk * 2 + lk
                            v_idx = j * p + k
                            V[v_idx][term_id] += coeff_B
            
            # Accumulate to W
            for br in range(2):
                for bc in range(2):
                    coeff_C_top = cC_top[br*2+bc]
                    if coeff_C_top == 0:
                        continue
                    for cr in range(2):
                        for cc in range(2):
                            coeff_C_bot = cC_bot[cr*2+cc]
                            if coeff_C_bot == 0:
                                continue
                            total_coeff = coeff_C_top * coeff_C_bot
                            i = br * 2 + cr
                            k = bc * 2 + cc
                            w_idx = k * n + i
                            W[w_idx][term_id] += total_coeff
            
            term_id += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
