import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for M_4,4,4 (rank 49) and M_4,4,1 (rank 16)
    # Total rank 65
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    # u: dict or array of coeffs for A (n x m)
    # v: dict or array of coeffs for B (m x p)
    # w: dict or array of coeffs for C (p x n) -> W[k*n+i]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (i, j, val)
        # v_coeffs: list of (j, k, val)
        # w_coeffs: list of (k, i, val)
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for i, j, val in u_coeffs:
            u_vec[i * m + j] = val
        for j, k, val in v_coeffs:
            v_vec[j * p + k] = val
        for k, i, val in w_coeffs:
            w_vec[k * n + i] = val
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # --- Part 1: 4x4x4 Strassen ---
    
    # Strassen for 2x2x2
    # Returns list of (u_coeffs, v_coeffs, w_coeffs) for the 7 products
    # and the linear combinations for C
    
    # Actually, we can just generate the 49 terms directly.
    
    # Let's define the Strassen terms for 2x2 blocks.
    # Blocks are indexed by (block_i, block_j)
    
    # We need to map the 49 scalar multiplications to the global indices.
    
    # Strassen Step 1: 7 block products M1..M7
    # Each M is a 2x2 matrix product.
    # Strassen Step 2: Each M is decomposed into 7 scalar products.
    
    # We will iterate through the 7 block products, and for each, iterate through the 7 scalar products.
    
    # Define the linear forms for A and B for each of the 7 M's in 2x2x2 Strassen.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M_s, we have a linear form on A_blocks and B_blocks.
    # A_blocks are 2x2 matrices.
    # We need to expand A_blocks into global A indices.
    
    # Global A is 4x4.
    # A_block[bi, bj] corresponds to A[2*bi:2*bi+2, 2*bj:2*bj+2].
    
    # Let's generate the terms.
    
    strassen_terms = []
    
    # M1
    # A_form: A00 + A11 -> [[1,0],[0,1]] in block space?
    # A00 is block (0,0), A11 is block (1,1).
    # So A coeffs: Block(0,0): [[1,0],[0,1]], Block(1,1): [[1,0],[0,1]]
    # B_form: B00 + B11 -> Block(0,0): [[1,0],[0,1]], Block(1,1): [[1,0],[0,1]]
    # C contribution: C00 += 1, C11 += 1
    
    # M2
    # A_form: A10 + A11 -> Block(1,0): [[1,0],[0,1]], Block(1,1): [[1,0],[0,1]]
    # B_form: B00 -> Block(0,0): [[1,0],[0,1]]
    # C contribution: C10 += 1
    
    # M3
    # A_form: A00 -> Block(0,0): [[1,0],[0,1]]
    # B_form: B01 - B11 -> Block(0,1): [[1,0],[0,1]], Block(1,1): [[-1,0],[0,-1]]
    # C contribution: C01 += 1
    
    # M4
    # A_form: A11 -> Block(1,1): [[1,0],[0,1]]
    # B_form: B10 - B00 -> Block(1,0): [[1,0],[0,1]], Block(0,0): [[-1,0],[0,-1]]
    # C contribution: C00 += 1, C10 += 1
    
    # M5
    # A_form: A00 + A01 -> Block(0,0): [[1,0],[0,1]], Block(0,1): [[1,0],[0,1]]
    # B_form: B11 -> Block(1,1): [[1,0],[0,1]]
    # C contribution: C00 -= 1, C01 += 1
    
    # M6
    # A_form: A10 - A00 -> Block(1,0): [[1,0],[0,1]], Block(0,0): [[-1,0],[0,-1]]
    # B_form: B00 + B01 -> Block(0,0): [[1,0],[0,1]], Block(0,1): [[1,0],[0,1]]
    # C contribution: C11 += 1
    
    # M7
    # A_form: A01 - A11 -> Block(0,1): [[1,0],[0,1]], Block(1,1): [[-1,0],[0,-1]]
    # B_form: B10 + B11 -> Block(1,0): [[1,0],[0,1]], Block(1,1): [[1,0],[0,1]]
    # C contribution: C00 += 1
    
    # This is getting complicated to track manually.
    # I will write a small function to expand block forms to global forms.
    
    def expand_block_form(block_coeffs):
        # block_coeffs: dict {(bi, bj): [[c00, c01], [c10, c11]]}
        # Returns list of (i, j, val)
        terms = []
        for (bi, bj), mat in block_coeffs.items():
            for ii in range(2):
                for jj in range(2):
                    if mat[ii][jj] != 0:
                        global_i = 2 * bi + ii
                        global_j = 2 * bj + jj
                        terms.append((global_i, global_j, mat[ii][jj]))
        return terms
    
    # Define the 7 M's block forms
    I2 = [[1, 0], [0, 1]]
    N2 = [[-1, 0], [0, -1]]
    
    M_forms = [
        # (A_blocks, B_blocks, C_contribs)
        # C_contribs: dict {(ci, cj): coeff}
        ({(0,0): I2, (1,1): I2}, {(0,0): I2, (1,1): I2}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): I2, (1,1): I2}, {(0,0): I2}, {(1,0): 1}), # M2
        ({(0,0): I2}, {(0,1): I2, (1,1): N2}, {(0,1): 1}), # M3
        ({(1,1): I2}, {(1,0): I2, (0,0): N2}, {(0,0): 1, (1,0): 1}), # M4
        ({(0,0): I2, (0,1): I2}, {(1,1): I2}, {(0,0): -1, (0,1): 1}), # M5
        ({(1,0): I2, (0,0): N2}, {(0,0): I2, (0,1): I2}, {(1,1): 1}), # M6
        ({(0,1): I2, (1,1): N2}, {(1,0): I2, (1,1): I2}, {(0,0): 1}), # M7
    ]
    
    # Now, for each M_s, we decompose the 2x2 block product into 7 scalar products using Strassen.
    # The 2x2 block product is C_block = A_block * B_block.
    # We use the same Strassen formulas for the 2x2 case.
    
    # The 2x2 Strassen terms (for scalar blocks)
    # M1 = (A00+A11)(B00+B11)
    # ...
    
    # We need to map the "local" 2x2 indices to the "global" 4x4 indices?
    # No, the "blocks" are 2x2 matrices.
    # The product of two 2x2 matrices is computed with 7 scalar multiplications of the *entries* of the linear forms.
    # Wait. The linear forms $L_A$ and $L_B$ produce 2x2 matrices.
    # Let $X = L_A(A)$ and $Y = L_B(B)$.
    # We need to compute $Z = X Y$.
    # $Z$ is a 2x2 matrix.
    # We use Strassen to compute $Z$ from $X$ and $Y$ with 7 multiplications.
    # Each multiplication is of the form $x_{linear} \cdot y_{linear}$?
    # No, Strassen uses linear combinations of the *entries* of $X$ and $Y$.
    # So we need to expand the Strassen terms for $X$ and $Y$ in terms of $X_{ij}$ and $Y_{ij}$.
    # Then substitute $X_{ij}$ and $Y_{ij}$ with their linear forms in $A$ and $B$.
    
    # Strassen for 2x2:
    # m1 = (X00+X11)(Y00+Y11)
    # m2 = (X10+X11)Y00
    # m3 = X00(Y01-Y11)
    # m4 = X11(Y10-Y00)
    # m5 = (X00+X01)Y11
    # m6 = (X10-X00)(Y00+Y01)
    # m7 = (X01-X11)(Y10+Y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # For each of the 7 block products M_s (from the 4x4 level), we have X and Y.
    # We generate 7 scalar multiplications for each M_s.
    # Total 49.
    
    # We need to track the contribution of each scalar multiplication to the global C.
    # The scalar multiplication $m_t$ contributes to $Z_{ij}$ with a coefficient.
    # $Z_{ij}$ is the $(i,j)$ entry of the block $M_s$.
    # $M_s$ contributes to the global $C$ with the coefficients defined in `M_forms`.
    
    # Let's iterate.
    
    for s in range(7):
        A_blocks, B_blocks, C_contribs = M_forms[s]
        
        # Expand A_blocks to global A terms
        # But we need the *local* X entries for Strassen.
        # X is a 2x2 matrix. X_{ij} is the linear form for the (i,j) entry of the block product.
        # Wait, X is $L_A(A)$.
        # $L_A(A)$ is a 2x2 matrix.
        # The entries of $L_A(A)$ are linear forms in $A$.
        
        # Let's get the linear forms for X_{00}, X_{01}, X_{10}, X_{11}.
        # X_{ij} = sum of A_terms where block (bi,bj) has coeff at (ii,jj) such that it maps to (i,j)?
        # No, $L_A(A)$ is defined by the block coefficients.
        # $X = \sum_{bi,bj} c_{bi,bj} A_{bi,bj}$.
        # $X_{ij} = \sum_{bi,bj} c_{bi,bj,ij} A_{bi,bj,ij}$?
        # No, $A_{bi,bj}$ is a 2x2 block.
        # $X_{ij} = \sum_{bi,bj} (A_{bi,bj})_{ij} \cdot c_{bi,bj}$?
        # No, the coefficients are matrices.
        # $X = \sum_{bi,bj} A_{bi,bj} \cdot C_{bi,bj}$?
        # In my `M_forms`, the coefficients are matrices like I2.
        # So $X_{ij} = \sum_{bi,bj} (A_{bi,bj})_{ij} \cdot (C_{bi,bj})_{ij}$?
        # No, $C_{bi,bj}$ is a scalar? No, it's a matrix.
        # If $C_{bi,bj} = I2$, then $X_{ij} = (A_{bi,bj})_{ij}$.
        
        # So, for a given $s$, $X_{ij}$ is the sum of $(A_{bi,bj})_{ij}$ for all $(bi,bj)$ where $C_{bi,bj}$ has a 1 at $(ij,ij)$?
        # Actually, if $C_{bi,bj}$ is a diagonal matrix, it scales the diagonal.
        # If it's I2, it's the identity.
        
        # Let's just extract the linear forms for X_{00}, X_{01}, X_{10}, X_{11}.
        
        X_forms = [[None for _ in range(2)] for _ in range(2)]
        Y_forms = [[None for _ in range(2)] for _ in range(2)]
        
        for i in range(2):
            for j in range(2):
                X_forms[i][j] = []
                Y_forms[i][j] = []
                
        for (bi, bj), C_mat in A_blocks.items():
            for ii in range(2):
                for jj in range(2):
                    if C_mat[ii][jj] != 0:
                        # This coefficient applies to A_{bi,bj}[ii,jj]
                        # Which is A[2*bi+ii, 2*bj+jj]
                        # And it contributes to X[ii,jj]?
                        # No, X is the sum of blocks.
                        # X[ii,jj] gets contribution from A[bi,bj][ii,jj] * C_mat[ii,jj]?
                        # Yes, if C_mat is diagonal.
                        # If C_mat is not diagonal, it mixes.
                        # But in Strassen, the forms are like (A00+A11).
                        # A00 and A11 are blocks.
                        # So X00 = A00[0,0] + A11[0,0].
                        # So yes, it's entry-wise.
                        
                        global_i = 2 * bi + ii
                        global_j = 2 * bj + jj
                        X_forms[i][j].append((global_i, global_j, C_mat[ii][jj]))
                        
        for (bi, bj), C_mat in B_blocks.items():
            for ii in range(2):
                for jj in range(2):
                    if C_mat[ii][jj] != 0:
                        global_i = 2 * bi + ii
                        global_j = 2 * bj + jj
                        Y_forms[i][j].append((global_i, global_j, C_mat[ii][jj]))
                        
        # Now apply Strassen to X and Y
        # m1 = (X00+X11)(Y00+Y11)
        # ...
        
        strassen_local = [
            # (X_coeffs, Y_coeffs, Z_contribs)
            # Z_contribs: {(zi, zj): coeff}
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # m1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # m2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}), # m3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # m4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # m5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # m6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}), # m7
        ]
        
        for t in range(7):
            X_local_coeffs, Y_local_coeffs, Z_contribs = strassen_local[t]
            
            # Build global X and Y linear forms
            u_coeffs = []
            v_coeffs = []
            
            for (zi, zj), coeff in X_local_coeffs.items():
                if coeff != 0:
                    # X[zi,zj] is a linear form
                    for (gi, gj, val) in X_forms[zi][zj]:
                        u_coeffs.append((gi, gj, val * coeff))
                        
            for (zi, zj), coeff in Y_local_coeffs.items():
                if coeff != 0:
                    for (gi, gj, val) in Y_forms[zi][zj]:
                        v_coeffs.append((gi, gj, val * coeff))
                        
            # Build W coeffs
            # Z_contribs gives contribution to Z[zi,zj]
            # Z is the block M_s.
            # M_s contributes to global C with C_contribs.
            # Z[zi,zj] is the (zi,zj) entry of the block.
            # The block M_s is located at (bi, bj) in the 4x4 C?
            # No, M_s is one of the 7 intermediate matrices.
            # M_s contributes to C blocks.
            # C_contribs maps (ci, cj) to coeff.
            # Z[zi,zj] is a scalar.
            # It contributes to C_block[ci,cj][zi,zj]?
            # Yes.
            # Global C index: (2*ci+zi, 2*cj+zj).
            # W index: (k, i) -> (2*cj+zj, 2*ci+zi).
            
            w_coeffs = []
            for (zi, zj), z_coeff in Z_contribs.items():
                if z_coeff == 0: continue
                for (ci, cj), c_coeff in C_contribs.items():
                    if c_coeff == 0: continue
                    global_k = 2 * cj + zj
                    global_i = 2 * ci + zi
                    w_coeffs.append((global_k, global_i, z_coeff * c_coeff))
                    
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # --- Part 2: 4x4x1 Schoolbook ---
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            u_coeffs = [(i, j, 1)]
            v_coeffs = [(j, 4, 1)]
            w_coeffs = [(4, i, 1)] # k=4, i=i
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # Convert to matrices
    rank = len(U_list)
    U = [row[:] for row in zip(*U_list)]
    V = [row[:] for row in zip(*V_list)]
    W = [row[:] for row in zip(*W_list)]
    
    return {"U": U, "V": V, "W": W}
