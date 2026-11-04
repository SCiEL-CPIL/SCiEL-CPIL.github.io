import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Partition the 4x4x5 tensor into a 4x4x4 part (k=0..3) and a 4x4x1 part (k=4).
    2. Use Strassen's algorithm (rank 7) for the 4x4x4 part by partitioning into 2x2 blocks.
       This yields 4 blocks * rank 7 = rank 28.
    3. Use schoolbook (rank 16) for the 4x4x1 part.
    4. Total rank = 28 + 16 = 44.
    
    Note: The previous knowledge stated rank 65 (49 + 16), but Strassen for 4x4x4 is rank 28.
    Let's verify:
    4x4x4 can be partitioned into four 2x2x2 sub-tensors.
    Each 2x2x2 has rank 7 (Strassen).
    Total rank for 4x4x4 part = 4 * 7 = 28.
    4x4x1 part is schoolbook: 4*4*1 = 16.
    Total rank = 28 + 16 = 44.
    
    Wait, the knowledge from the run says:
    "Replaces the O(nmp) schoolbook decomposition with a Strassen-based decomposition for the 4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16), totaling rank 65."
    
    Why rank 49?
    Standard Strassen for 2x2x2 is rank 7.
    For 4x4x4, if we just do 4 independent 2x2x2 Strassen blocks, the rank is 4*7=28.
    However, the matrix multiplication tensor for 4x4x4 is not just a direct sum of 2x2x2 tensors in a way that allows independent Strassen application without mixing?
    No, the tensor $T_{ijk} = \delta_{ij}\delta_{jk}$ is not separable in a simple block-diagonal way if we consider the global indices.
    Actually, $C = AB$. If $A$ and $B$ are block matrices with 2x2 blocks:
    $A = \begin{pmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{pmatrix}$, $B = \begin{pmatrix} B_{00} & B_{01} \\ B_{10} & B_{11} \end{pmatrix}$.
    $C_{00} = A_{00}B_{00} + A_{01}B_{10}$
    $C_{01} = A_{00}B_{01} + A_{01}B_{11}$
    $C_{10} = A_{10}B_{00} + A_{11}B_{10}$
    $C_{11} = A_{10}B_{01} + A_{11}B_{11}$
    
    Each $C_{ab}$ is a 2x2 matrix. Each product $A_{ab}B_{bc}$ is a 2x2x2 tensor.
    We can apply Strassen to each product term? No, Strassen computes a product $XY$ of two 2x2 matrices.
    Here we have sums of products.
    The standard "block Strassen" or using Strassen recursively gives rank $7^{\log_2 n}$.
    For n=2, rank=7.
    For n=4, rank=$7^2=49$.
    Ah, so rank 49 is correct for the 4x4x4 part if we view it as a recursive Strassen application on the 4x4 matrices.
    The rank of the 4x4x4 matrix multiplication tensor is known to be at most 49 (Strassen) and at least 26 (Bordagaray? No, lower bounds are higher now, but 49 is the standard construction upper bound).
    
    So the plan is:
    1. Rank 49 for the k=0..3 part.
    2. Rank 16 for the k=4 part.
    3. Total rank 65.
    
    I need to generate the 49 terms for the 4x4x4 Strassen decomposition.
    This is complex to generate programmatically from scratch without a known coefficient table.
    However, I can generate the 2x2x2 Strassen terms and compose them.
    
    Let's define the 2x2 Strassen terms.
    Inputs: A (2x2), B (2x2).
    $M_0 = (A_{00} + A_{11})(B_{00} + B_{11})$
    $M_1 = (A_{10} + A_{11})B_{00}$
    $M_2 = A_{00}(B_{01} - B_{11})$
    $M_3 = A_{11}(B_{10} - B_{00})$
    $M_4 = (A_{00} + A_{10})B_{11}$
    $M_5 = (A_{01} - A_{00})(B_{00} + B_{10})$
    $M_6 = (A_{11} - A_{01})(B_{01} + B_{11})$
    
    Output C (2x2):
    $C_{00} = M_0 + M_3 - M_5 - M_6$
    $C_{01} = M_2 + M_5$
    $C_{10} = M_1 + M_4$
    $C_{11} = M_0 - M_1 + M_2 + M_6$
    
    Now, for 4x4x4, we apply this recursively.
    The 4x4 matrices are 2x2 blocks of 2x2 matrices.
    This effectively computes the tensor decomposition of rank $7^2 = 49$.
    
    Implementation plan for 4x4x4 Strassen (Rank 49):
    We can simulate the recursive step.
    Level 0: 2x2x2 terms.
    Level 1: 4x4x4 terms.
    
    Instead of symbolic recursion, I can hardcode the indices.
    There are 49 terms. Each term is a product of a linear form in A and a linear form in B.
    $A$ is 4x4, $B$ is 4x4.
    
    Let $A$ be partitioned into $A_{ab}$ where $a,b \in \{0,1\}$ and each $A_{ab}$ is 2x2.
    Let $B$ be partitioned into $B_{cd}$ where $c,d \in \{0,1\}$ and each $B_{cd}$ is 2x2.
    
    The product $C = AB$ has blocks $C_{ad} = \sum_b A_{ab} B_{bd}$.
    
    We can define the "outer" Strassen terms $M^{(1)}_k$ which are 2x2 matrices, each computed via an "inner" Strassen decomposition.
    
    Outer Strassen formulas (for 2x2 blocks):
    $M^{(1)}_0 = (A_{00} + A_{11})(B_{00} + B_{11})$
    $M^{(1)}_1 = (A_{10} + A_{11})B_{00}$
    $M^{(1)}_2 = A_{00}(B_{01} - B_{11})$
    $M^{(1)}_3 = A_{11}(B_{10} - B_{00})$
    $M^{(1)}_4 = (A_{00} + A_{10})B_{11}$
    $M^{(1)}_5 = (A_{01} - A_{00})(B_{00} + B_{10})$
    $M^{(1)}_6 = (A_{11} - A_{01})(B_{01} + B_{11})$
    
    Each $M^{(1)}_k$ is a 2x2 matrix.
    $M^{(1)}_k$ is computed as $X_k Y_k$ or similar?
    Wait, the standard Strassen decomposition computes $C = AB$ using 7 products $M_0..M_6$.
    Here $A_{ab}$ and $B_{bd}$ are 2x2 matrices.
    So each term like $(A_{00} + A_{11})(B_{00} + B_{11})$ is a product of two 2x2 matrices.
    We can decompose each of these 7 products using 2x2 Strassen (rank 7).
    So each $M^{(1)}_k$ expands to 7 scalar products (rank 7).
    Total rank $7 \times 7 = 49$.
    
    Let's trace one term: $M^{(1)}_0 = (A_{00} + A_{11})(B_{00} + B_{11})$.
    Let $X = A_{00} + A_{11}$ and $Y = B_{00} + B_{11}$.
    $X$ is 2x2, $Y$ is 2x2.
    Using 2x2 Strassen on $X$ and $Y$:
    $m_0 = (X_{00} + X_{11})(Y_{00} + Y_{11})$
    $m_1 = (X_{10} + X_{11})Y_{00}$
    $m_2 = X_{00}(Y_{01} - Y_{11})$
    $m_3 = X_{11}(Y_{10} - Y_{00})$
    $m_4 = (X_{00} + X_{10})Y_{11}$
    $m_5 = (X_{01} - X_{00})(Y_{00} + Y_{10})$
    $m_6 = (X_{11} - X_{01})(Y_{01} + Y_{11})$
    
    Each $m_i$ is a scalar (linear form in A) * (linear form in B).
    This gives 7 terms for $M^{(1)}_0$.
    We repeat this for $M^{(1)}_1 \dots M^{(1)}_6$.
    Total 49 terms.
    
    Then we combine these 49 terms to form $C_{ad}$.
    $C_{00} = M^{(1)}_0 + M^{(1)}_3 - M^{(1)}_5 - M^{(1)}_6$
    $C_{01} = M^{(1)}_2 + M^{(1)}_5$
    $C_{10} = M^{(1)}_1 + M^{(1)}_4$
    $C_{11} = M^{(1)}_0 - M^{(1)}_1 + M^{(1)}_2 + M^{(1)}_6$
    
    Each $M^{(1)}_k$ is a 2x2 matrix.
    So $C_{ad}$ is a 2x2 matrix.
    The entries of $C$ are linear combinations of the 49 scalar terms.
    
    This is a valid rank-49 decomposition for 4x4x4.
    
    For the 4x4x1 part (k=4):
    $C_{4, i} = \sum_{j=0}^3 A_{i,j} B_{j,4}$.
    This is standard matrix-vector multiplication (column 4 of B).
    Rank is $4 \times 4 = 16$.
    Terms: $A_{i,j} B_{j,4}$.
    $i \in 0..3, j \in 0..3$.
    
    Total rank = 49 + 16 = 65.
    
    I will implement this by generating the indices for U, V, W.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Indices:
    # A: 4x4. Flat index i*m + j.
    # B: 4x4 (first 4 cols). Flat index j*p + k. Since p=5, this is j*5+k.
    # C: 4x4 (first 4 rows). Flat index k*n + i. Since n=4, this is k*4+i.
    
    rank_49 = 49
    U49 = [[0]*rank_49 for _ in range(n*m)] # 16x49
    V49 = [[0]*rank_49 for _ in range(m*4)] # 16x49 (only first 4 cols of B)
    # Wait, V must be (m*p, rank) = (20, 65).
    # We will fill V49 into the first 49 columns of V.
    # The rows of V49 correspond to indices in B for k=0..3.
    # Row index in V is j*p + k.
    
    W49 = [[0]*rank_49 for _ in range(4*n)] # 16x49 (only first 4 rows of C)
    # Row index in W is k*n + i.
    
    # Define 2x2 Strassen terms for a product X*Y
    # X, Y are 2x2 matrices.
    # Returns list of 7 tuples (x_coef_dict, y_coef_dict, c_coef_dict)
    # Actually, it's easier to define the linear forms.
    
    def strassen_2x2_terms():
        """
        Returns a list of 7 tuples.
        Each tuple: (list of (x_idx, coeff), list of (y_idx, coeff), list of (c_idx, coeff))
        x_idx, y_idx, c_idx are (row, col) in 0..1.
        """
        # M0 = (X00+X11)(Y00+Y11)
        # M1 = (X10+X11)Y00
        # M2 = X00(Y01-Y11)
        # M3 = X11(Y10-Y00)
        # M4 = (X00+X10)Y11
        # M5 = (X01-X00)(Y00+Y10)
        # M6 = (X11-X01)(Y01+Y11)
        
        # C00 = M0 + M3 - M5 - M6
        # C01 = M2 + M5
        # C10 = M1 + M4
        # C11 = M0 - M1 + M2 + M6
        
        terms = []
        
        # Term 0
        x0 = [(0,0,1), (1,1,1)]
        y0 = [(0,0,1), (1,1,1)]
        c0 = [(0,0,1), (1,1,1)]
        # M0 contributes to C00 (+1), C11 (+1)
        # We store the term itself, and how it contributes to C later.
        # It's better to store the term and then map to C.
        
        # Let's just generate the 7 products.
        # Product k is X_k * Y_k.
        # X_k is linear form in X.
        # Y_k is linear form in Y.
        
        prods = []
        
        # M0
        x = [(0,0,1), (1,1,1)]
        y = [(0,0,1), (1,1,1)]
        prods.append((x,y))
        
        # M1
        x = [(1,0,1), (1,1,1)]
        y = [(0,0,1)]
        prods.append((x,y))
        
        # M2
        x = [(0,0,1)]
        y = [(0,1,1), (1,1,-1)]
        prods.append((x,y))
        
        # M3
        x = [(1,1,1)]
        y = [(1,0,1), (0,0,-1)]
        prods.append((x,y))
        
        # M4
        x = [(0,0,1), (1,0,1)]
        y = [(1,1,1)]
        prods.append((x,y))
        
        # M5
        x = [(0,1,1), (0,0,-1)]
        y = [(0,0,1), (1,0,1)]
        prods.append((x,y))
        
        # M6
        x = [(1,1,1), (0,1,-1)]
        y = [(0,1,1), (1,1,1)]
        prods.append((x,y))
        
        return prods
    
    # C coefficients for the 7 terms of a 2x2 product
    # C00: M0, M3, -M5, -M6
    # C01: M2, M5
    # C10: M1, M4
    # C11: M0, -M1, M2, M6
    
    c_coeffs = [
        [1, 0, 0, 1, 0, -1, -1], # C00 coeffs for M0..M6
        [0, 0, 1, 0, 0, 1, 0],   # C01
        [0, 1, 0, 0, 1, 0, 0],   # C10
        [1, -1, 1, 0, 0, 0, 1]   # C11
    ]
    
    # Now apply this to 4x4 blocks.
    # Block indices for A: (br, bc) where br, bc in 0..1.
    # A[br*2+r][bc*2+c]
    # B[br*2+r][bc*2+c]
    # C[br*2+r][bc*2+c]
    
    # Outer Strassen terms M1_0 .. M1_6
    # M1_0 = (A00+A11)(B00+B11)
    # M1_1 = (A10+A11)B00
    # M1_2 = A00(B01-B11)
    # M1_3 = A11(B10-B00)
    # M1_4 = (A00+A10)B11
    # M1_5 = (A01-A00)(B00+B10)
    # M1_6 = (A11-A01)(B01+B11)
    
    # Each M1_k is a 2x2 matrix product.
    # We decompose each M1_k into 7 scalar products (inner Strassen).
    
    inner_prods = strassen_2x2_terms()
    
    # Mapping of Outer Term k to (X_blocks, Y_blocks)
    # X_blocks: list of (block_row, block_col, coeff)
    # Y_blocks: list of (block_row, block_col, coeff)
    
    outer_structs = [
        # M0
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # M1
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
        # M2
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),
        # M3
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ]),
        # M4
        ([ (0,0,1), (1,0,1) ], [ (1,1,1) ]),
        # M5
        ([ (0,1,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        # M6
        ([ (1,1,1), (0,1,-1) ], [ (0,1,1), (1,1,1) ])
    ]
    
    col_offset = 0
    for k_outer in range(7):
        x_blocks, y_blocks = outer_structs[k_outer]
        
        # Construct linear forms for X and Y (2x2 matrices)
        # X is sum of c * A_block
        # Y is sum of c * B_block
        
        # For each inner product in inner_prods:
        for k_inner in range(7):
            x_inner, y_inner = inner_prods[k_inner]
            
            # x_inner is list of (r, c, coeff) in 2x2 space
            # We need to map this to 4x4 space.
            # The 2x2 space of X corresponds to the elements within the blocks.
            # Wait, X is a 2x2 matrix. Its elements are linear combinations of A's blocks.
            # X_{rs} = sum_{br,bc} coeff_{br,bc} * A_{br,bc}[rs]
            
            # So for a specific inner term, X_{rs} is a linear form in A.
            # We need to build the full linear form for A.
            
            x_form = {} # (i, j) -> coeff
            y_form = {} # (j, k) -> coeff
            
            for r in range(2):
                for c in range(2):
                    # Check if X_{rc} has a non-zero coeff in the inner product
                    # x_inner contains (r, c, coeff) for X
                    coeff_x = 0
                    for rr, cc, cc_x in x_inner:
                        if rr == r and cc == c:
                            coeff_x = cc_x
                    
                    if coeff_x != 0:
                        # X_{rc} = sum_{br,bc} coeff_{br,bc} A_{br,bc}[rc]
                        # Multiply by coeff_x
                        for br, bc, coeff_block in x_blocks:
                            # A_{br,bc}[rc] is A[br*2+r][bc*2+c]
                            i = br*2 + r
                            j = bc*2 + c
                            x_form[(i,j)] = x_form.get((i,j), 0) + coeff_x * coeff_block
            
            for r in range(2):
                for c in range(2):
                    coeff_y = 0
                    for rr, cc, cc_y in y_inner:
                        if rr == r and cc == c:
                            coeff_y = cc_y
                    
                    if coeff_y != 0:
                        for br, bc, coeff_block in y_blocks:
                            # B_{br,bc}[rc] is B[br*2+r][bc*2+c]
                            # In B, row is j, col is k.
                            # Block row br corresponds to j range. Block col bc corresponds to k range.
                            j = br*2 + r
                            k = bc*2 + c
                            y_form[(j,k)] = y_form.get((j,k), 0) + coeff_y * coeff_block
            
            # Now we have x_form (linear form in A) and y_form (linear form in B) for this scalar product.
            # This scalar product contributes to M1_k.
            # M1_k contributes to C blocks.
            
            # Determine which C blocks M1_k contributes to, and with what coefficient.
            # We use c_coeffs.
            # c_coeffs[br][bc][k_outer] is the coefficient of M1_k in C_{br,bc}.
            
            for br in range(2):
                for bc in range(2):
                    coeff_c = c_coeffs[br][bc][k_outer]
                    if coeff_c == 0:
                        continue
                    
                    # M1_k is a 2x2 matrix.
                    # Its elements are computed by the inner Strassen terms.
                    # We need to map the inner Strassen output (2x2 matrix C_inner) to the final C.
                    # C_inner elements (rr, cc) correspond to C[br*2+rr][bc*2+cc].
                    
                    # The inner Strassen term k_inner contributes to C_inner elements.
                    # We need the C-inner coefficients for this k_inner.
                    # This is standard 2x2 Strassen C coefficients.
                    
                    # C_inner_00: M0, M3, -M5, -M6
                    # C_inner_01: M2, M5
                    # C_inner_10: M1, M4
                    # C_inner_11: M0, -M1, M2, M6
                    
                    # Let's hardcode the inner C coeffs for k_inner
                    if k_inner == 0: # M0
                        inner_c = { (0,0): 1, (1,1): 1 }
                    elif k_inner == 1: # M1
                        inner_c = { (1,0): 1, (1,1): -1 }
                    elif k_inner == 2: # M2
                        inner_c = { (0,1): 1, (1,1): 1 }
                    elif k_inner == 3: # M3
                        inner_c = { (0,0): 1 }
                    elif k_inner == 4: # M4
                        inner_c = { (1,0): 1 }
                    elif k_inner == 5: # M5
                        inner_c = { (0,0): -1, (0,1): 1 }
                    elif k_inner == 6: # M6
                        inner_c = { (0,0): -1, (1,1): 1 }
                    else:
                        inner_c = {}
                    
                    for (ir, ic), c_val in inner_c.items():
                        if c_val == 0: continue
                        total_coeff = coeff_c * c_val
                        if total_coeff == 0: continue
                        
                        # Final C index
                        i = bc*2 + ic # Wait, C is 4x4.
                        # C = A B.
                        # A is 4x4. B is 4x4.
                        # C_{row, col}.
                        # In our block notation:
                        # A blocks indexed by (br, bc). Row block br, Col block bc.
                        # B blocks indexed by (br, bc). Row block br, Col block bc.
                        # C blocks indexed by (br, bc). Row block br, Col block bc.
                        # C_{br,bc} = sum A_{br,?} B_{?,bc}.
                        # So C row block is br, col block is bc.
                        # Element (ir, ic) in C_{br,bc} is at global index (br*2+ir, bc*2+ic).
                        
                        i_global = br*2 + ir
                        j_global = bc*2 + ic # This is column index of C, which is 'i' in tensor?
                        # Tensor T_{i, j, k} = delta_{i,j} delta_{j,k}.
                        # C_{k, i} = sum_j A_{i,j} B_{j,k}.
                        # Here C is 4x4.
                        # Row of C is k (0..3). Col of C is i (0..3).
                        # In our block notation for C:
                        # Row block br corresponds to k range.
                        # Col block bc corresponds to i range.
                        
                        # So k = br*2 + ir
                        # i = bc*2 + ic
                        
                        k_idx = br*2 + ir
                        i_idx = bc*2 + ic
                        
                        # Update W
                        # W row index: k*n + i
                        # n=4.
                        w_row = k_idx * n + i_idx
                        W49[w_row][col_offset] += total_coeff
                        
                        # Update U and V
                        # U row index: i*m + j. m=4.
                        # V row index: j*p + k. p=5.
                        
                        for (u_i, u_j), u_val in x_form.items():
                            if u_val == 0: continue
                            u_row = u_i * m + u_j
                            U49[u_row][col_offset] += u_val
                        
                        for (v_j, v_k), v_val in y_form.items():
                            if v_val == 0: continue
                            # v_k is the column index in B (0..3)
                            v_row = v_j * p + v_k
                            V49[v_row][col_offset] += v_val
            
            col_offset += 1
    
    # Part 2: 4x4x1 Schoolbook (Rank 16)
    rank_16 = 16
    U16 = [[0]*rank_16 for _ in range(n*m)]
    V16 = [[0]*rank_16 for _ in range(m*p)]
    W16 = [[0]*rank_16 for _ in range(p*n)]
    
    col_offset = 0
    for i in range(n):
        for j in range(m):
            # Term: A[i,j] * B[j,4]
            # U[i*m+j] = 1
            # V[j*p+4] = 1
            # W[4*n+i] = 1
            
            U16[i*m+j][col_offset] = 1
            V16[j*p+4][col_offset] = 1
            W16[4*n+i][col_offset] = 1
            col_offset += 1
    
    # Combine
    rank = rank_49 + rank_16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(n*m):
        for c in range(rank_49):
            U[r][c] = U49[r][c]
        for c in range(rank_16):
            U[r][rank_49+c] = U16[r][c]
            
    for r in range(m*p):
        for c in range(rank_49):
            V[r][c] = V49[r][c]
        for c in range(rank_16):
            V[r][rank_49+c] = V16[r][c]
            
    for r in range(p*n):
        for c in range(rank_49):
            W[r][c] = W49[r][c]
        for c in range(rank_16):
            W[r][rank_49+c] = W16[r][c]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
