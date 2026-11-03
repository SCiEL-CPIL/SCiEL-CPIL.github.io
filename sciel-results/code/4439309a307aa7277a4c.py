import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 81: Strassen 49 (4x4x4) + Schoolbook 32 (4x4x2)
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # We split 4x4 into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen Steps for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix product.
    # We use the rank-7 decomposition for each 2x2 product.
    
    # Define the 7 scalar multiplications for a generic 2x2 product X*Y
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # t1 = (x00+x11)(y00+y11)
    # t2 = (x10+x11)y00
    # t3 = x00(y01-y11)
    # t4 = x11(y10-y00)
    # t5 = (x00+x01)y11
    # t6 = (x10-x00)(y00+y01)
    # t7 = (x01-x11)(y10+y11)
    
    # Coefficients for X and Y for each t
    # Format: { (row, col): coeff }
    T_X = [
        {(0,0): 1, (1,1): 1},  # t1
        {(1,0): 1, (1,1): 1},  # t2
        {(0,0): 1},            # t3
        {(1,1): 1},            # t4
        {(0,0): 1, (0,1): 1},  # t5
        {(1,0): 1, (0,0): -1}, # t6
        {(0,1): 1, (1,1): -1}  # t7
    ]
    T_Y = [
        {(0,0): 1, (1,0): 1},  # t1
        {(0,0): 1},            # t2
        {(0,1): 1, (1,1): -1}, # t3
        {(1,0): 1, (0,0): -1}, # t4
        {(1,1): 1},            # t5
        {(0,0): 1, (0,1): 1},  # t6
        {(1,0): 1, (1,1): 1}   # t7
    ]
    
    # Coefficients for the output 2x2 matrix Z = X*Y for each t
    # Z00 = t1 + t4 - t5 + t7
    # Z01 = t3 + t5
    # Z10 = t2 + t4
    # Z11 = t1 - t2 + t3 + t6
    T_Z = [
        {(0,0): 1, (1,1): 1},  # t1 contributes to Z00 and Z11
        {(1,0): 1},            # t2 contributes to Z10
        {(0,1): 1, (1,1): 1},  # t3 contributes to Z01 and Z11
        {(0,0): 1, (1,0): 1},  # t4 contributes to Z00 and Z10
        {(0,0): -1, (0,1): 1}, # t5 contributes to Z00 (-1) and Z01 (1)
        {(1,1): 1},            # t6 contributes to Z11
        {(0,0): 1, (1,1): 1}   # t7 contributes to Z00 and Z11
    ]
    # Correction on T_Z based on standard Strassen reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify T_Z mapping:
    # t1 (M1): Z00 += 1, Z11 += 1
    # t2 (M2): Z10 += 1, Z11 += -1
    # t3 (M3): Z01 += 1, Z11 += 1
    # t4 (M4): Z00 += 1, Z10 += 1
    # t5 (M5): Z00 += -1, Z01 += 1
    # t6 (M6): Z11 += 1
    # t7 (M7): Z00 += 1, Z11 += 1
    
    T_Z_corrected = [
        {(0,0): 1, (1,1): 1},  # t1
        {(1,0): 1, (1,1): -1}, # t2
        {(0,1): 1, (1,1): 1},  # t3
        {(0,0): 1, (1,0): 1},  # t4
        {(0,0): -1, (0,1): 1}, # t5
        {(1,1): 1},            # t6
        {(0,0): 1, (1,1): 1}   # t7
    ]

    # Strassen Steps Definitions
    # Each step defines how the 2x2 blocks of A and B are combined to form X and Y for the scalar multiplications.
    # X and Y are 2x2 matrices.
    # We need to map the elements of X and Y back to the global A and B indices.
    
    # A blocks:
    # A00: indices (0,0),(0,1),(1,0),(1,1)
    # A01: indices (0,2),(0,3),(1,2),(1,3)
    # A10: indices (2,0),(2,1),(3,0),(3,1)
    # A11: indices (2,2),(2,3),(3,2),(3,3)
    
    # B blocks:
    # B00: indices (0,0),(0,1),(1,0),(1,1)
    # B01: indices (0,2),(0,3),(1,2),(1,3)
    # B10: indices (2,0),(2,1),(3,0),(3,1)
    # B11: indices (2,2),(2,3),(3,2),(3,3)
    
    # We will define for each Strassen step s (0..6) and scalar mult t (0..6):
    # The coefficients for A[i,j] and B[j,k] and C[i,k].
    
    # This is complex. Let's use a helper to get global indices.
    # A[i,j] where i,j in 0..3.
    # Block row br = i//2, Block col bc = j//2.
    # Local row lr = i%2, Local col lc = j%2.
    
    # Strassen Step Definitions:
    # s=0: X = A00+A11, Y = B00+B11
    # s=1: X = A10+A11, Y = B00
    # s=2: X = A00, Y = B01-B11
    # s=3: X = A11, Y = B10-B00
    # s=4: X = A00+A01, Y = B11
    # s=5: X = A10-A00, Y = B00+B01
    # s=6: X = A01-A11, Y = B10+B11
    
    # For each step, we define the mapping from (local_row, local_col) in X to (block_row, block_col, local_row, local_col) in A with a sign.
    
    def get_A_coeffs(step, local_r, local_c):
        # Returns list of (i, j, coeff)
        coeffs = []
        if step == 0:
            # A00 + A11
            coeffs.append((local_r, local_c, 1))       # A00
            coeffs.append((local_r+2, local_c+2, 1))   # A11
        elif step == 1:
            # A10 + A11
            coeffs.append((local_r+2, local_c, 1))     # A10
            coeffs.append((local_r+2, local_c+2, 1))   # A11
        elif step == 2:
            # A00
            coeffs.append((local_r, local_c, 1))
        elif step == 3:
            # A11
            coeffs.append((local_r+2, local_c+2, 1))
        elif step == 4:
            # A00 + A01
            coeffs.append((local_r, local_c, 1))       # A00
            coeffs.append((local_r, local_c+2, 1))     # A01
        elif step == 5:
            # A10 - A00
            coeffs.append((local_r+2, local_c, 1))     # A10
            coeffs.append((local_r, local_c, -1))      # A00
        elif step == 6:
            # A01 - A11
            coeffs.append((local_r, local_c+2, 1))     # A01
            coeffs.append((local_r+2, local_c+2, -1))  # A11
        return coeffs

    def get_B_coeffs(step, local_r, local_c):
        # Returns list of (j, k, coeff) where j is row in B (col in A), k is col in B
        # Note: In matrix mult C = A*B, B is m x p.
        # Here m=4, p=4.
        # B[j,k].
        # Blocks of B:
        # B00: rows 0,1 cols 0,1
        # B01: rows 0,1 cols 2,3
        # B10: rows 2,3 cols 0,1
        # B11: rows 2,3 cols 2,3
        
        coeffs = []
        if step == 0:
            # B00 + B11
            coeffs.append((local_r, local_c, 1))       # B00
            coeffs.append((local_r+2, local_c+2, 1))   # B11
        elif step == 1:
            # B00
            coeffs.append((local_r, local_c, 1))
        elif step == 2:
            # B01 - B11
            coeffs.append((local_r, local_c+2, 1))     # B01
            coeffs.append((local_r+2, local_c+2, -1))  # B11
        elif step == 3:
            # B10 - B00
            coeffs.append((local_r+2, local_c, 1))     # B10
            coeffs.append((local_r, local_c, -1))      # B00
        elif step == 4:
            # B11
            coeffs.append((local_r+2, local_c+2, 1))
        elif step == 5:
            # B00 + B01
            coeffs.append((local_r, local_c, 1))       # B00
            coeffs.append((local_r, local_c+2, 1))     # B01
        elif step == 6:
            # B10 + B11
            coeffs.append((local_r+2, local_c, 1))     # B10
            coeffs.append((local_r+2, local_c+2, 1))   # B11
        return coeffs

    def get_C_coeffs(step, local_r, local_c):
        # Returns list of (i, k, coeff)
        # C blocks:
        # C00: rows 0,1 cols 0,1
        # C01: rows 0,1 cols 2,3
        # C10: rows 2,3 cols 0,1
        # C11: rows 2,3 cols 2,3
        
        # Reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # local_r, local_c are indices in the 2x2 result of the scalar mult t.
        # We need to map this to the global C block.
        # The "step" s determines which M_s this is.
        # The "local_r, local_c" determines which element of M_s (2x2) this is.
        
        # Wait, the T_Z mapping is for the 2x2 matrix M_s.
        # M_s is a 2x2 matrix.
        # M_s[0,0] contributes to C00, C01, C10, C11 depending on s.
        # Actually, M_s is just a 2x2 matrix.
        # The reconstruction formula sums the *entire* 2x2 matrices M_s into the 2x2 blocks C00, etc.
        # So M_s[0,0] adds to C00[0,0], C00[0,1]? No.
        # M_s is a 2x2 matrix.
        # C00 (2x2) = M1 (2x2) + M4 (2x2) - M5 (2x2) + M7 (2x2).
        # So M_s[i,j] adds to C_block[i,j] with the coefficient from the reconstruction formula.
        
        # Which C block does M_s contribute to?
        # All M_s contribute to ALL C blocks?
        # No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for a given step s, M_s contributes to specific C blocks.
        # And within that C block, M_s[local_r, local_c] contributes to C_block[local_r, local_c].
        
        coeffs = []
        if step == 0: # M1
            coeffs.append((local_r, local_c, 1))       # C00
            coeffs.append((local_r, local_c+2, 0))     # C01 (0)
            coeffs.append((local_r+2, local_c, 0))     # C10 (0)
            coeffs.append((local_r+2, local_c+2, 1))   # C11
            # Wait, M1 is a 2x2 matrix.
            # M1[0,0] -> C00[0,0] and C11[0,0]?
            # Yes, the block M1 is added to block C00 and block C11.
            # So M1[local_r, local_c] adds to C[local_r, local_c] (in C00) and C[local_r+2, local_c+2] (in C11).
            
        elif step == 1: # M2
            coeffs.append((local_r+2, local_c, 1))     # C10
            coeffs.append((local_r+2, local_c+2, -1))  # C11
            
        elif step == 2: # M3
            coeffs.append((local_r, local_c+2, 1))     # C01
            coeffs.append((local_r+2, local_c+2, 1))   # C11
            
        elif step == 3: # M4
            coeffs.append((local_r, local_c, 1))       # C00
            coeffs.append((local_r+2, local_c, 1))     # C10
            
        elif step == 4: # M5
            coeffs.append((local_r, local_c, -1))      # C00
            coeffs.append((local_r, local_c+2, 1))     # C01
            
        elif step == 5: # M6
            coeffs.append((local_r+2, local_c+2, 1))   # C11
            
        elif step == 6: # M7
            coeffs.append((local_r, local_c, 1))       # C00
            coeffs.append((local_r+2, local_c+2, 1))   # C11
            
        return coeffs

    # Iterate over all 49 terms
    for s in range(7):
        for t in range(7):
            # Get coeffs for X and Y from T_X, T_Y
            x_coeffs = T_X[t]
            y_coeffs = T_Y[t]
            
            # For each non-zero coeff in X, get A coeffs
            # For each non-zero coeff in Y, get B coeffs
            # The term is sum_{x_idx, y_idx} x_coeff * y_coeff * A_part * B_part
            # We need to distribute this into U, V, W.
            # U[i*m+j] += coeff_A
            # V[j*p+k] += coeff_B
            # W[k*n+i] += coeff_C
            
            # This is a rank-1 update for each combination?
            # No, the scalar multiplication is (Sum A_parts) * (Sum B_parts).
            # This is NOT a rank-1 term in the global tensor if the sums are non-trivial.
            # Wait.
            # The rank-7 decomposition of 2x2x2 says:
            # C = Sum_t (X_t * Y_t) where X_t, Y_t are linear combos of inputs.
            # But X_t * Y_t is a matrix product? No, X_t and Y_t are scalars in the rank-7 decomp?
            # No.
            # In Strassen, M_k = P_k * Q_k where P_k, Q_k are 2x2 matrices.
            # And M_k is computed by rank-7 decomp of 2x2x2.
            # So M_k = Sum_{t=1}^7 u_{k,t} v_{k,t} w_{k,t} ?
            # No.
            # The rank-7 decomp of 2x2x2 is:
            # C_{ij} = Sum_{t=1}^7 U_{i,j,t} V_{j,k,t} W_{k,i,t} ?
            # No, the indices are mixed.
            
            # Let's stick to the scalar expansion.
            # The term is (Sum_{a} c_a A_a) * (Sum_{b} d_b B_b).
            # This equals Sum_{a,b} c_a d_b A_a B_b.
            # Each A_a B_b is a rank-1 term in the tensor?
            # A_a is A[i_a, j_a]. B_b is B[j_b, k_b].
            # For this to be a valid term in the decomposition of C = AB, we must have j_a == j_b.
            # In the Strassen construction, the linear combinations P_k and Q_k are such that
            # the "j" indices match up for the multiplication to be valid?
            # Actually, the tensor decomposition allows any linear combination.
            # The term is $U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
            # If we have a term $c A_{i_1, j_1} d B_{j_2, k_1}$, it only contributes if $j_1 = j_2$.
            # In Strassen, the sums are over blocks.
            # e.g. (A00 + A11). A00 has j in {0,1}. A11 has j in {2,3}.
            # (B00 + B11). B00 has j in {0,1}. B11 has j in {2,3}.
            # The product (A00+A11)(B00+B11) involves terms like A00*B00 (j=0,1 match), A00*B11 (j=0,1 vs 2,3 NO MATCH).
            # Wait.
            # Matrix multiplication requires the inner dimension to match.
            # A is 4x4. B is 4x4.
            # A[i,j] B[j,k].
            # If we take A00 (cols 0,1) and B11 (rows 2,3), the indices j don't match.
            # So A00 * B11 is NOT part of the matrix product C.
            # Therefore, the linear combination (A00+A11) * (B00+B11) is NOT simply the sum of all products.
            # It is the sum of products where the indices match?
            # No, Strassen works because the "mismatched" terms cancel out in the final reconstruction?
            # No, that's not right.
            # In Strassen, the inputs to the recursive calls are 2x2 matrices.
            # The multiplication of these 2x2 matrices is exact.
            # The "mismatched" terms don't exist because the 2x2 multiplication is defined on the blocks.
            # When we expand to scalars, we must be careful.
            
            # Actually, the rank-7 decomp of 2x2x2 is:
            # C_{ij} = Sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} is NOT the form.
            # The form is:
            # C_{i,k} = Sum_{j} A_{i,j} B_{j,k}.
            # The tensor T_{i,j,k}.
            # The rank-7 decomp provides 7 triples (u, v, w) such that
            # T_{i,j,k} = Sum_t u_{i,j,t} v_{j,k,t} w_{k,i,t}? No.
            # T_{i,j,k} is the identity tensor? No, T is the multiplication tensor.
            # T_{i,j,k} = 1 if ...?
            # The multiplication tensor is defined by C_{i,k} = Sum_j A_{i,j} B_{j,k}.
            # The tensor is $\\mathcal{T}_{i,j,k}$.
            # The decomposition is $\\mathcal{T}_{i,j,k} = \\sum_{r} U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
            # No, the standard decomposition is:
            # $C_{i,k} = \\sum_{r} W_{k,i,r} (\\sum_{j} U_{i,j,r} V_{j,k,r})$?
            # No.
            # The factors are U (n x m x R), V (m x p x R), W (p x n x R).
            # The reconstruction is:
            # $C_{i,k} = \\sum_{r} W_{k,i,r} \\cdot (U_r \\cdot V_r)$?
            # No.
            # $C_{i,k} = \\sum_{r} W_{k,i,r} \\cdot M_r$ where $M_r$ is a scalar?
            # No.
            # The standard tensor decomposition for matrix multiplication is:
            # $\\mathcal{T} = \\sum_{r=1}^R u_r \\otimes v_r \\otimes w_r$.
            # Where $u_r$ is a vector of size $n \\times m$, $v_r$ of size $m \\times p$, $w_r$ of size $p \\times n$.
            # And the multiplication is:
            # $C_{i,k} = \\sum_{r} w_{k,i,r} \\cdot (\\sum_{j} u_{i,j,r} v_{j,k,r})$?
            # No.
            # $C_{i,k} = \\sum_{r} w_{k,i,r} \\cdot (u_r \\cdot v_r)$ is not dimensionally correct.
            
            # Correct interpretation:
            # $U$ is $R \\times (n \\times m)$.
            # $V$ is $R \\times (m \\times p)$.
            # $W$ is $R \\times (p \\times n)$.
            # $C_{i,k} = \\sum_{r} W_{r, k \\cdot n + i} \\cdot (\\sum_{j} U_{r, i \\cdot m + j} V_{r, j \\cdot p + k})$?
            # No.
            # $C_{i,k} = \\sum_{r} W_{r, k \\cdot n + i} \\cdot M_r$ where $M_r$ is not a scalar.
            
            # Let's look at the interface:
            # U: (n*m, rank)
            # V: (m*p, rank)
            # W: (p*n, rank)
            # This implies:
            # $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot (\\text{something})$
            # The "something" must be derived from U and V.
            # Usually: $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot (U_{:, r} \\cdot V_{:, r})$?
            # No, U and V are vectors. Their dot product is a scalar.
            # So $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot (\\sum_{x} U_{x, r} V_{x, r})$?
            # This would mean U and V are multiplied element-wise and summed?
            # If so, the term is $\\sum_{x} U_{x,r} V_{x,r}$.
            # And $x$ indexes the $n \\times m$ and $m \\times p$ spaces?
            # But U has size $n \\times m$ and V has size $m \\times p$. They can't be multiplied element-wise unless $n \\times m = m \\times p$.
            # Here $4 \\times 4 = 16$ and $4 \\times 6 = 24$. Sizes differ.
            
            # Therefore, the decomposition must be:
            # $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot (\\sum_{j} U_{i \\cdot m + j, r} V_{j \\cdot p + k, r})$?
            # No, V is indexed by $j \\cdot p + k$.
            # So for a fixed r, we have vectors $u_r$ and $v_r$.
            # But $u_r$ has length 16, $v_r$ has length 24.
            # The term must be a contraction.
            # The only way this works is if the "rank" term is a single scalar multiplication $A_{i,j} B_{j,k}$.
            # In that case, $U_{i \\cdot m + j, r} = 1$, $V_{j \\cdot p + k, r} = 1$, $W_{k \\cdot n + i, r} = 1$.
            # And the sum is over r.
            # $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot U_{i \\cdot m + j_r, r} \\cdot V_{j_r \\cdot p + k, r}$?
            # No, j is summed over in the matrix multiplication.
            # The tensor decomposition is:
            # $\\mathcal{T}_{i,j,k} = \\sum_{r} U_{i \\cdot m + j, r} V_{j \\cdot p + k, r} W_{k \\cdot n + i, r}$?
            # If this is the definition, then for a fixed r, U, V, W are vectors.
            # And the product is element-wise?
            # If so, $U$ and $V$ must have the same length.
            # But $n \\cdot m = 16$ and $m \\cdot p = 24$.
            # So they cannot be element-wise multiplied.
            
            # UNLESS: The "rank" decomposition is for the tensor $\\mathcal{T}_{i,j,k}$.
            # And the factors are:
            # $U$ of size $R \\times (n \\times m)$
            # $V$ of size $R \\times (m \\times p)$
            # $W$ of size $R \\times (p \\times n)$
            # And the reconstruction is:
            # $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot (\\sum_{j} U_{i \\cdot m + j, r} V_{j \\cdot p + k, r})$?
            # No, that sums j.
            
            # Let's assume the standard:
            # $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot M_r$
            # Where $M_r$ is the result of the "multiplication" represented by $U_r$ and $V_r$.
            # But $U_r$ and $V_r$ are just coefficients.
            # The only logical interpretation for differing sizes is:
            # $C_{i,k} = \\sum_{r} W_{k \\cdot n + i, r} \\cdot (\\sum_{j} U_{i \\cdot m + j, r} V_{j \\cdot p + k, r})$ is WRONG because V depends on j and k.
            
            # Actually, looking at the interface:
            # U: (n*m, rank)
            # V: (m*p, rank)
            # W: (p*n, rank)
            # This matches the dimensions of the tensor factors.
            # The decomposition is:
            # $\\mathcal{T}_{i,j,k} = \\sum_{r} U_{i \\cdot m + j, r} V_{j \\cdot p + k, r} W_{k \\cdot n + i, r}$
            # This requires that for a fixed r, the term is non-zero only if the indices match?
            # No, it's a sum of rank-1 tensors.
            # A rank-1 tensor is $u \\otimes v \\otimes w$.
            # $u$ has size $n \\times m$.
            # $v$ has size $m \\times p$.
            # $w$ has size $p \\times n$.
            # The tensor $\\mathcal{T}$ has indices $i, j, k$.
            # $\\mathcal{T}_{i,j,k} = \\sum_{r} u_{r, i, j} v_{r, j, k} w_{r, k, i}$?
            # Note the index j in u and v.
            # $u_{r, i, j}$ and $v_{r, j, k}$.
            # So for a fixed r, we sum over j?
            # No, the tensor component is $\\mathcal{T}_{i,j,k}$.
            # If the decomposition is $\\sum u_{r, i, j} v_{r, j, k} w_{r, k, i}$, then for a fixed i, j, k, there is no sum over j.
            # It is just the product of the components.
            # So $U_{i \\cdot m + j, r}$ is $u_{r, i, j}$.
            # $V_{j \\cdot p + k, r}$ is $v_{r, j, k}$.
            # $W_{k \\cdot n + i, r}$ is $w_{r, k, i}$.
            # And the term is $u_{r, i, j} v_{r, j, k} w_{r, k, i}$.
            # This is valid!
            
            # So, for each rank r, we specify:
            # - Which (i, j) in U has a non-zero value.
            # - Which (j, k) in V has a non-zero value.
            # - Which (k, i) in W has a non-zero value.
            # And the values.
            # The sum over r gives the tensor.
            
            # For the schoolbook term $A_{i,j} B_{j,k}$:
            # r corresponds to (i, j, k).
            # $U_{i, j, r} = 1$.
            # $V_{j, k, r} = 1$.
            # $W_{k, i, r} = 1$.
            # All other entries for this r are 0.
            
            # For Strassen:
            # We have terms like $(A_{i_1, j_1} + A_{i_2, j_2}) (B_{j_1, k_1} + B_{j_2, k_2})$.
            # This expands to $A_{i_1, j_1} B_{j_1, k_1} + A_{i_1, j_1} B_{j_2, k_2} + A_{i_2, j_2} B_{j_1, k_1} + A_{i_2, j_2} B_{j_2, k_2}$.
            # The cross terms $A_{i_1, j_1} B_{j_2, k_2}$ have $j_1 \\neq j_2$.
            # In the tensor decomposition, the term for r would have:
            # $U_{i_1, j_1, r} = 1$.
            # $V_{j_2, k_2, r} = 1$.
            # $W_{k_2, i_1, r} = 1$.
            # This contributes to $\\mathcal{T}_{i_1, j_2, k_2}$?
            # No, the tensor component is $\\mathcal{T}_{i, j, k}$.
            # The product is $u_{i, j} v_{j, k} w_{k, i}$.
            # If we set $u_{i_1, j_1} = 1$ and $v_{j_2, k_2} = 1$, then for the component $\\mathcal{T}_{i_1, j_2, k_2}$, we need $u_{i_1, j_2}$ and $v_{j_2, k_2}$.
            # But $u_{i_1, j_2}$ is 0 (since we set $u_{i_1, j_1}$).
            # So the cross term does NOT appear in the tensor decomposition if we use this form.
            # This implies that Strassen's "cross terms" must cancel out or be zero in the final sum?
            # Or my understanding of the tensor decomposition form is slightly off.
            
            # Actually, the standard tensor decomposition for matrix multiplication DOES allow cross terms.
            # The rank-7 decomp of 2x2x2 has terms like $(a+d)(e+g)$.
            # This expands to $ae + ag + de + dg$.
            # The term $ag$ corresponds to $A_{0,0} B_{1,1}$? (if a=A00, g=B10? No, g is B10 in 2x2?).
            # In 2x2, $A = [[a,b],[c,d]]$, $B = [[e,f],[g,h]]$.
            # $ag$ is $A_{0,0} B_{1,0}$.
            # The indices are $i=0, j=0$ for A, and $j=1, k=0$ for B.
            # $j$ does not match ($0 \\neq 1$).
            # So $A_{0,0} B_{1,0}$ is not part of the matrix product $C_{0,0} = ae + bg$.
            # So why does Strassen work?
            # Because the "cross terms" appear in OTHER components of C?
            # No, they don't.
            # The magic of Strassen is that the linear combinations are chosen such that when you combine the results, the cross terms cancel out in the final C blocks.
            # But in the tensor decomposition, each rank-1 term is a specific $u \\otimes v \\otimes w$.
            # If we use the rank-7 decomp, we are adding 7 rank-1 terms.
            # Each term corresponds to a scalar multiplication $x \\cdot y$.
            # The "cross terms" are inherent in the linear combinations.
            # The tensor decomposition sums these 7 terms.
            # The sum of the 7 rank-1 tensors equals the multiplication tensor.
            # This is a mathematical fact.
            
            # So, to implement Strassen, I just need to generate the 7 rank-1 terms for each 2x2 block multiplication, and map them to the global indices.
            
            # For the term $(A_{i_1, j_1} + A_{i_2, j_2}) (B_{j_1, k_1} + B_{j_2, k_2})$:
            # This is NOT a rank-1 term in the global tensor if $j_1 \\neq j_2$.
            # Wait.
            # The rank-7 decomp of 2x2x2 says:
            # $C = \\sum_{t=1}^7 M_t$ where $M_t$ are rank-1 tensors?
            # No, $M_t$ are scalar multiplications.
            # The tensor is $\\sum_{t} u_t \\otimes v_t \\otimes w_t$.
            # For the term $t_1 = (a+d)(e+g)$:
            # $u_{t1}$ has $u_{0,0}=1, u_{1,1}=1$.
            # $v_{t1}$ has $v_{0,0}=1, v_{1,0}=1$.
            # $w_{t1}$ has $w_{0,0}=1, w_{1,1}=1$?
            # No, $w$ depends on the reconstruction.
            
            # I will use the `add_term` logic which assumes a single $A_{i,j} B_{j,k}$ structure.
            # This is ONLY valid for Schoolbook.
            # For Strassen, I MUST handle the linear combinations.
            # This means for one "Strassen term", I might need to add multiple entries to U, V, W?
            # No, a rank-1 term in the tensor decomposition has ONE non-zero pattern in U, V, W.
            # The "cross terms" are handled by the fact that the sum of the 7 rank-1 tensors equals the full tensor.
            # So for each of the 7 scalar multiplications, I define ONE rank-1 tensor.
            # The U vector for this tensor has non-zeros at the positions corresponding to the linear combination of A.
            # The V vector has non-zeros at the positions corresponding to the linear combination of B.
            # The W vector has non-zeros at the positions corresponding to the linear combination of C.
            
            # So, for $t_1 = (a+d)(e+g)$:
            # $U_{0,0, r} = 1$
            # $U_{1,1, r} = 1$
            # $V_{0,0, r} = 1$
            # $V_{1,0, r} = 1$
            # $W_{0,0, r} = 1$ (contributes to C00)
            # $W_{1,1, r} = 1$ (contributes to C11)
            # And so on.
            
            # This is exactly what my `get_A_coeffs`, `get_B_coeffs`, `get_C_coeffs` do!
            # They return the non-zero entries.
            # I just need to loop over them and set the values in U, V, W.
            
            # I will implement this.

    for s in range(7):
        for t in range(7):
            x_coeffs = T_X[t]
            y_coeffs = T_Y[t]
            
            # Get global A coeffs
            a_coeffs = []
            for (lr, lc), val in x_coeffs.items():
                for (i, j, coeff) in get_A_coeffs(s, lr, lc):
                    a_coeffs.append((i, j, val * coeff))
            
            # Get global B coeffs
            b_coeffs = []
            for (lr, lc), val in y_coeffs.items():
                for (j, k, coeff) in get_B_coeffs(s, lr, lc):
                    b_coeffs.append((j, k, val * coeff))
            
            # Get global C coeffs
            c_coeffs = []
            for (lr, lc), val in T_Z_corrected[t].items():
                for (i, k, coeff) in get_C_coeffs(s, lr, lc):
                    c_coeffs.append((i, k, val * coeff))
            
            # Now, add the rank-1 term
            # U[i*m+j] += a_val
            # V[j*p+k] += b_val
            # W[k*n+i] += c_val
            
            # Wait, the rank-1 term is $u \\otimes v \\otimes w$.
            # The value in U, V, W is the coefficient.
            # So I just set them.
            
            for (i, j, val) in a_coeffs:
                U[i * m + j][col] = val
            for (j, k, val) in b_coeffs:
                V[j * p + k][col] = val
            for (i, k, val) in c_coeffs:
                W[k * n + i][col] = val
            
            col += 1

    # --- Part 2: 4x4x2 Schoolbook (Rank 32) ---
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                add_term(i, j, k, 1, 1, 1)

    return {"U": U, "V": V, "W": W}

