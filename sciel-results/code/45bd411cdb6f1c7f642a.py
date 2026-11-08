import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 multiplication (k=0..3)
    # Part 2: 4x4x1 multiplication (k=4)
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # i, j are indices for A (0-based)
    # j, k are indices for B (0-based)
    # k, i are indices for C (0-based)
    # coeff_A, coeff_B, coeff_C are multipliers
    def add_term(i, j, k, coeff_A, coeff_B, coeff_C):
        nonlocal col
        if coeff_A == 0 or coeff_B == 0 or coeff_C == 0:
            return
        U[i * m + j][col] += coeff_A
        V[j * p + k][col] += coeff_B
        W[k * n + i][col] += coeff_C
        col += 1
        
    # --- Part 1: 4x4x4 ---
    # We decompose 4x4x4 into 7 block-products of 2x2x2.
    # Each 2x2x2 block product is decomposed into 7 scalar products.
    
    # Define the 7 Strassen linear combinations for 2x2 blocks
    # X and Y are 2x2 matrices of variables.
    # We need to map global indices (i,j) to block indices (u,v) and local indices (a,b).
    
    # Block (u,v) of A contains A[2u+a, 2v+b] for a,b in {0,1}
    # Block (u,v) of B contains B[2v+a, 2w+b] for a,b in {0,1}
    # Block (u,w) of C contains C[2u+a, 2w+b] for a,b in {0,1}
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each M_s, X_s and Y_s are 2x2 matrices.
    # Then we apply 2x2 Strassen to X_s Y_s = Z_s.
    
    # Let's define the linear forms for X_s and Y_s in terms of block elements.
    # Block element A[u,v,a,b] corresponds to global A[2u+a, 2v+b].
    
    # We will iterate s from 0 to 6 (for M1..M7)
    # And t from 0 to 6 (for inner Strassen terms)
    
    # Inner Strassen terms for 2x2 product X Y = Z:
    # m1 = (x00 + x11)(y00 + y11)
    # m2 = (x10 + x11)y00
    # m3 = x00(y01 - y11)
    # m4 = x11(y10 - y00)
    # m5 = (x00 + x01)y11
    # m6 = (x10 - x00)(y00 + y01)
    # m7 = (x01 - x11)(y10 + y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # We need to express x_ab and y_cd as linear combinations of global A and B elements.
    
    # For a given s, X_s is a 2x2 matrix.
    # X_s[u, v] is a linear form.
    # Let's precompute the coefficients for each X_s[u, v] and Y_s[u, v].
    
    # X_s coefficients for A blocks:
    # s=0 (M1): X0 = A11 + A22. X0[0,0] = A[11][0,0] + A[22][0,0], etc.
    # This is tedious. Let's use a dictionary or list of lists.
    
    # A_block[u, v, a, b] -> global i = 2u+a, j = 2v+b
    # B_block[v, w, a, b] -> global j = 2v+a, k = 2w+b (for 4x4x4, k is 0..3)
    # C_block[u, w, a, b] -> global i = 2u+a, k = 2w+b
    
    # Let's define the structure of X_s and Y_s.
    # X_s is 2x2. X_s[u, v] is a sum of A_block[?, ?, ?, ?].
    # Actually, X_s is formed from A blocks.
    # M1: X = A11 + A22. So X[0,0] = A11[0,0] + A22[0,0]. X[0,1] = A11[0,1] + A22[0,1]. etc.
    # M2: X = A21 + A22.
    # M3: X = A11.
    # M4: X = A22.
    # M5: X = A11 + A12.
    # M6: X = A21 - A11.
    # M7: X = A12 - A22.
    
    # Similarly for Y_s using B blocks.
    # M1: Y = B11 + B22.
    # M2: Y = B11.
    # M3: Y = B12 - B22.
    # M4: Y = B21 - B11.
    # M5: Y = B22.
    # M6: Y = B11 + B12.
    # M7: Y = B21 + B22.
    
    # Note: B block indices are (row_block, col_block).
    # B11 is block (0,0). B12 is (0,1). B21 is (1,0). B22 is (1,1).
    
    # Let's create a function to get the coefficients of a linear form on A or B.
    # Return a dict: {(i,j): coeff} for A, {(j,k): coeff} for B.
    
    def get_A_coeffs(s, u, v):
        # Returns dict of (i,j) -> coeff for X_s[u,v]
        coeffs = {}
        # u,v are 0 or 1 (local block indices in X_s)
        # Wait, X_s is a 2x2 matrix. The indices u,v here are the local 2x2 indices.
        # The "block" content comes from A.
        
        # Define which A blocks are involved and with what sign.
        # s=0: A11 + A22. Blocks (0,0) and (1,1).
        # s=1: A21 + A22. Blocks (1,0) and (1,1).
        # s=2: A11. Block (0,0).
        # s=3: A22. Block (1,1).
        # s=4: A11 + A12. Blocks (0,0) and (0,1).
        # s=5: A21 - A11. Blocks (1,0) and (0,0).
        # s=6: A12 - A22. Blocks (0,1) and (1,1).
        
        blocks = [
            [(0,0, 1), (1,1, 1)], # s=0
            [(1,0, 1), (1,1, 1)], # s=1
            [(0,0, 1)],           # s=2
            [(1,1, 1)],           # s=3
            [(0,0, 1), (0,1, 1)], # s=4
            [(1,0, 1), (0,0, -1)],# s=5
            [(0,1, 1), (1,1, -1)] # s=6
        ]
        
        for bu, bv, sign in blocks[s]:
            # Add block (bu, bv) to X_s[u, v]
            # The element at local (u,v) of the block corresponds to global:
            # i = 2*bu + u
            # j = 2*bv + v
            i = 2 * bu + u
            j = 2 * bv + v
            coeffs[(i, j)] = coeffs.get((i, j), 0) + sign
        return coeffs

    def get_B_coeffs(s, u, v):
        # Returns dict of (j,k) -> coeff for Y_s[u,v]
        coeffs = {}
        # Y_s is 2x2. Local indices u,v.
        # Blocks of B.
        # s=0: B11 + B22. Blocks (0,0) and (1,1).
        # s=1: B11. Block (0,0).
        # s=2: B12 - B22. Blocks (0,1) and (1,1).
        # s=3: B21 - B11. Blocks (1,0) and (0,0).
        # s=4: B22. Block (1,1).
        # s=5: B11 + B12. Blocks (0,0) and (0,1).
        # s=6: B21 + B22. Blocks (1,0) and (1,1).
        
        blocks = [
            [(0,0, 1), (1,1, 1)], # s=0
            [(0,0, 1)],           # s=1
            [(0,1, 1), (1,1, -1)],# s=2
            [(1,0, 1), (0,0, -1)],# s=3
            [(1,1, 1)],           # s=4
            [(0,0, 1), (0,1, 1)], # s=5
            [(1,0, 1), (1,1, 1)]  # s=6
        ]
        
        for bu, bv, sign in blocks[s]:
            # Add block (bu, bv) to Y_s[u, v]
            # j = 2*bu + u
            # k = 2*bv + v
            j = 2 * bu + u
            k = 2 * bv + v
            coeffs[(j, k)] = coeffs.get((j, k), 0) + sign
        return coeffs

    # Inner Strassen terms for 2x2 product X Y = Z
    # We need to expand m1..m7 into linear forms for X and Y.
    # Then for each term, we combine A coeffs and B coeffs.
    # And determine which C elements are affected.
    
    # m1 = (x00 + x11)(y00 + y11)
    # m2 = (x10 + x11)y00
    # m3 = x00(y01 - y11)
    # m4 = x11(y10 - y00)
    # m5 = (x00 + x01)y11
    # m6 = (x10 - x00)(y00 + y01)
    # m7 = (x01 - x11)(y10 + y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # We will iterate s=0..6, t=0..6.
    # For each (s,t), we have a rank-1 term.
    # The A-part is a linear form LA = sum c_ij A_ij.
    # The B-part is a linear form LB = sum d_jk B_jk.
    # The C-part is a linear form LC = sum e_ki C_ki.
    
    # We need to find LA, LB, LC for each t.
    
    # Let's precompute the linear forms for X and Y for each t.
    # X_forms[t] = list of (u, v, sign) for x_uv
    # Y_forms[t] = list of (u, v, sign) for y_uv
    
    X_forms = [
        [(0,0, 1), (1,1, 1)], # m1
        [(1,0, 1), (1,1, 1)], # m2
        [(0,0, 1)],           # m3
        [(1,1, 1)],           # m4
        [(0,0, 1), (0,1, 1)], # m5
        [(1,0, 1), (0,0, -1)],# m6
        [(0,1, 1), (1,1, -1)] # m7
    ]
    
    Y_forms = [
        [(0,0, 1), (1,1, 1)], # m1
        [(0,0, 1)],           # m2
        [(0,1, 1), (1,1, -1)],# m3
        [(1,0, 1), (0,0, -1)],# m4
        [(1,1, 1)],           # m5
        [(0,0, 1), (0,1, 1)], # m6
        [(1,0, 1), (1,1, 1)]  # m7
    ]
    
    # C contributions for each m_t
    # Z00: m1(+1), m4(+1), m5(-1), m7(+1)
    # Z01: m3(+1), m5(+1)
    # Z10: m2(+1), m4(+1)
    # Z11: m1(+1), m2(-1), m3(+1), m6(+1)
    
    C_contribs = [
        [(0,0, 1), (0,1, 1), (1,0, 1), (1,1, 1)], # m1: Z00, Z01, Z10, Z11 all get +1? No.
        # Let's check formulas:
        # Z00 = m1 + m4 - m5 + m7
        # Z01 = m3 + m5
        # Z10 = m2 + m4
        # Z11 = m1 - m2 + m3 + m6
        
        # m1: Z00(+1), Z11(+1)
        # m2: Z10(+1), Z11(-1)
        # m3: Z01(+1), Z11(+1)
        # m4: Z00(+1), Z10(+1)
        # m5: Z00(-1), Z01(+1)
        # m6: Z11(+1)
        # m7: Z00(+1)
        
        [(0,0, 1), (1,1, 1)], # m1
        [(1,0, 1), (1,1, -1)],# m2
        [(0,1, 1), (1,1, 1)], # m3
        [(0,0, 1), (1,0, 1)], # m4
        [(0,0, -1), (0,1, 1)],# m5
        [(1,1, 1)],           # m6
        [(0,0, 1)]            # m7
    ]
    
    for s in range(7):
        for t in range(7):
            # Build A coeffs
            A_coeffs = {}
            for u, v, sign in X_forms[t]:
                # Get coeffs for X_s[u, v]
                c_dict = get_A_coeffs(s, u, v)
                for (i, j), c in c_dict.items():
                    A_coeffs[(i, j)] = A_coeffs.get((i, j), 0) + sign * c
            
            # Build B coeffs
            B_coeffs = {}
            for u, v, sign in Y_forms[t]:
                c_dict = get_B_coeffs(s, u, v)
                for (j, k), c in c_dict.items():
                    B_coeffs[(j, k)] = B_coeffs.get((j, k), 0) + sign * c
            
            # Build C coeffs
            C_coeffs = {}
            for u, v, sign in C_contribs[t]:
                # Z_s[u, v] corresponds to C block (u, v)
                # Global C indices:
                # i = 2*u + a
                # k = 2*v + b
                # But wait, the C_contribs give the local Z indices (u,v).
                # We need to expand Z[u,v] into global C[i,k].
                # Z[u,v] is a 2x2 block? No, Z is 2x2.
                # Z[u,v] is a scalar in the 2x2 block product context?
                # No, X and Y are 2x2 matrices of linear forms.
                # Z = XY is 2x2 matrix of linear forms.
                # Z[u,v] is a linear form in C.
                # Specifically, Z[u,v] = sum_{a,b} C[2u+a, 2v+b] * delta?
                # In the 2x2x2 decomposition, Z is the result matrix.
                # The term m_t contributes to Z[u,v] with coefficient sign.
                # So Z[u,v] gets added to C_block[u,v].
                # C_block[u,v] consists of C[2u+a, 2v+b] for a,b in 0,1.
                # But wait, in the 2x2x2 tensor, the output is a 2x2 matrix.
                # Each element Z[a,b] is a scalar.
                # So for each (a,b) in 0,1, C[2u+a, 2v+b] gets the coefficient.
                
                for a in range(2):
                    for b in range(2):
                        i = 2 * u + a
                        k = 2 * v + b
                        C_coeffs[(k, i)] = C_coeffs.get((k, i), 0) + sign
            
            # Now we have A_coeffs, B_coeffs, C_coeffs.
            # We need to decompose this rank-1 term into scalar multiplications?
            # No, A_coeffs and B_coeffs are linear forms.
            # The rank-1 term is (sum A_coeffs * A) * (sum B_coeffs * B) -> sum C_coeffs * C.
            # This is NOT a rank-1 term in the original tensor unless the linear forms are single variables.
            # Wait. The Strassen decomposition for 2x2x2 uses linear combinations.
            # The rank-1 terms in the 2x2x2 decomposition are of the form:
            # (linear form in X) * (linear form in Y) -> (linear form in Z).
            # This is a valid rank-1 term for the *block* tensor.
            # But for the *global* tensor, we need to expand these.
            
            # Actually, a term like (x00 + x11)(y00 + y11) is a product of two linear forms.
            # This corresponds to a rank-1 term in the tensor space if we consider the variables to be the elements of X, Y.
            # But our variables are the elements of A, B.
            # Since x00, x11 are linear in A, the product is a quadratic form in A.
            # This is NOT a rank-1 term in the A, B, C tensor.
            
            # I made a mistake. The Strassen algorithm reduces the number of *multiplications* of *matrices*.
            # But the tensor rank is about the number of *scalar* multiplications.
            # When we say rank 7 for 2x2x2, we mean 7 scalar multiplications.
            # The 7 terms are:
            # m1 = (x00 + x11)(y00 + y11)
            # Here x00, x11, y00, y11 are variables (elements of the 2x2 matrices).
            # If x00 and x11 are distinct variables, then (x00+x11) is a linear form.
            # The product (x00+x11)(y00+y11) is a rank-1 term in the tensor of variables x, y, z.
            # So, if we treat the 2x2 blocks as "variables", the rank is 7.
            # But in the 4x4x4 case, the "variables" are the 16 elements of A, 16 of B.
            # The block elements are NOT independent variables; they ARE the variables.
            # So x00 is A[0,0]. x11 is A[0,1]? No.
            # In the block decomposition, X is a 2x2 matrix of *blocks*.
            # X[0,0] is a 2x2 matrix of A elements.
            # So X[0,0] is NOT a variable. It is a matrix.
            # The product X[0,0] Y[0,0] is a matrix product.
            # So we cannot just treat block products as scalar products.
            
            # Correct approach:
            # The rank-49 decomposition of 4x4x4 is obtained by applying the rank-7 decomposition of 2x2x2 to the 7 block products.
            # Each block product M_s = X_s Y_s is a 2x2 matrix product.
            # The rank of a 2x2 matrix product is 7.
            # So M_s can be computed with 7 scalar multiplications.
            # Total 7 * 7 = 49 scalar multiplications.
            
            # So, for each s (1..7), we need to generate the 7 rank-1 terms for the 2x2x2 multiplication of the blocks X_s and Y_s.
            # The "variables" for this sub-problem are the 4 elements of X_s and 4 of Y_s.
            # But X_s elements are linear forms in A.
            # So a term like (x00 + x11)(y00 + y11) becomes:
            # (sum c1 A + sum c2 A) * (sum d1 B + sum d2 B)
            # This is a product of two linear forms in A and B.
            # This is a valid rank-1 term in the global tensor?
            # Yes. A rank-1 term is $u_i v_j w_k$.
            # Here $u$ is a vector of coefficients for A, $v$ for B, $w$ for C.
            # The term is $(\sum a_{ij} A_{ij}) (\sum b_{jk} B_{jk}) = \sum c_{ki} C_{ki}$.
            # This is exactly what we have.
            
            # So we just need to output the coefficients.
            
            # However, we must be careful. The term $(x00+x11)(y00+y11)$ is ONE rank-1 term.
            # It is NOT decomposed into 4 terms.
            # So for each s, t, we add ONE term to the global decomposition.
            
            # We have A_coeffs (dict (i,j)->c), B_coeffs (dict (j,k)->c), C_coeffs (dict (k,i)->c).
            # We add this single term to U, V, W.
            
            # But wait, if A_coeffs has multiple non-zero entries, it's still one term.
            # U[i*m+j, col] = A_coeffs[(i,j)]
            # V[j*p+k, col] = B_coeffs[(j,k)]
            # W[k*n+i, col] = C_coeffs[(k,i)]
            
            # We need to iterate over all non-zero coefficients.
            # But they must be consistent.
            # Actually, we just fill the column `col` with these coefficients.
            
            # Find max indices to ensure we cover all.
            # Just iterate over all possible i,j,k and check if coeff is non-zero.
            
            for i in range(n):
                for j in range(m):
                    cA = A_coeffs.get((i, j), 0)
                    if cA != 0:
                        U[i * m + j][col] = cA
            
            for j in range(m):
                for k in range(p):
                    cB = B_coeffs.get((j, k), 0)
                    if cB != 0:
                        V[j * p + k][col] = cB
            
            for k in range(p):
                for i in range(n):
                    cC = C_coeffs.get((k, i), 0)
                    if cC != 0:
                        W[k * n + i][col] = cC
            
            col += 1

    # --- Part 2: 4x4x1 (k=4) ---
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # 16 terms.
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
