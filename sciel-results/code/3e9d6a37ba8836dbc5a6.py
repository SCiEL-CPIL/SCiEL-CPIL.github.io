import numpy as np

def construct():
    n = 5
    m = 5
    p = 5
    
    # Schoolbook rank is n*m*p = 125? No.
    # The rank of the matrix multiplication tensor $M_{n,p,q}$ is the number of terms in the bilinear algorithm.
    # Schoolbook: $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Number of terms is $n \times p \times m$? No.
    # The summation is over $j$. For each pair $(i,k)$, we sum over $j$.
    # Total terms = $n \times p \times m$?
    # Wait. The tensor is $T_{(i,k), (i,j), (j,k)}$.
    # The indices are:
    # U: (i, k) -> flattened $i*p + k$? No, U is for A. A is $n \times m$.
    # Let's check the interface:
    # U: (n*m, rank) -> Rows correspond to A's elements.
    # V: (m*p, rank) -> Rows correspond to B's elements.
    # W: (p*n, rank) -> Rows correspond to C's elements.
    #
    # Wait, the standard matrix multiplication is $C = A B$.
    # $A$ is $n \times m$. $B$ is $m \times p$. $C$ is $n \times p$.
    # $C_{i,k} = \sum_{j=0}^{m-1} A_{i,j} B_{j,k}$.
    #
    # U rows: index into A. $a = i \times m + j$.
    # V rows: index into B. $b = j \times p + k$.
    # W rows: index into C. $c = i \times p + k$.
    #
    # The tensor $T$ is defined by $T_{a,b,c} = 1$ if $a,b,c$ are consistent with $A_{i,j} B_{j,k} \to C_{i,k}$, else 0.
    #
    # Schoolbook:
    # For each $i \in 0..n-1$, $j \in 0..m-1$, $k \in 0..p-1$:
    #   There is a term $A_{i,j} B_{j,k}$ contributing to $C_{i,k}$.
    #   This requires rank 1 for each such triplet.
    #   Total rank = $n \times m \times p$.
    #
    # For 5x5x5: Rank = 125.
    #
    # Wait, is the "rank" of the tensor the bilinear complexity? Yes.
    # Schoolbook complexity is $nmp$.
    #
    # So for 5x5, schoolbook rank is 125.
    #
    # The directive mentioned "rank 30". This is much lower than 125.
    # Strassen for 2x2 is rank 7 (vs 8 schoolbook).
    # Strassen for 3x3 is rank 23 (vs 27 schoolbook).
    #
    # So the "rank" in the task refers to the bilinear complexity.
    #
    # My previous calculation of "Schoolbook Rank 25" was wrong. It's 125.
    #
    # So I need a decomposition with rank < 125.
    #
    # The directive suggests rank 30.
    # If I can implement the 3x3 Strassen (23) and 2x2 Strassen (7) and combine them correctly for the 5x5 case, I might get a valid low rank.
    #
    # But how to combine them for the full 5x5?
    #
    # If I just use the direct sum of the 3x3 and 2x2 tensors, I only cover the diagonal blocks.
    # I MUST cover the cross blocks.
    #
    # If I use Schoolbook for the cross blocks:
    # Cross blocks:
    # 1. $A_{3 \times 2} B_{2 \times 2} \to C_{3 \times 2}$. Rank $3 \times 2 \times 2 = 12$.
    # 2. $A_{2 \times 3} B_{3 \times 2} \to C_{2 \times 2}$. Rank $2 \times 3 \times 2 = 12$.
    # 3. $A_{3 \times 2} B_{2 \times 3} \to C_{3 \times 3}$. Rank $3 \times 2 \times 3 = 18$.
    # 4. $A_{2 \times 3} B_{3 \times 2} \to C_{2 \times 3}$. Rank $2 \times 3 \times 2 = 12$.
    #
    # Plus diagonal:
    # 5. $A_{3 \times 3} B_{3 \times 3} \to C_{3 \times 3}$. Rank 23 (Strassen-3).
    # 6. $A_{2 \times 2} B_{2 \times 2} \to C_{2 \times 2}$. Rank 7 (Strassen-2).
    #
    # Total Rank = $23 + 7 + 12 + 12 + 18 + 12 = 84$.
    #
    # This is valid (exact) and rank 84 < 125.
    # Score would be $125/84 \approx 1.48$.
    #
    # Can I do better?
    # The directive says "achieve a total rank of 30".
    # This implies a much smarter construction than block-wise schoolbook for cross terms.
    #
    # However, without the explicit formulas for the "Direct Sum" construction that achieves 30, I cannot implement it.
    #
    # I will implement the Rank 84 solution (Strassen for diagonal, Schoolbook for cross).
    # This is a valid, improved solution over Schoolbook (125).
    #
    # Wait, I need the actual Strassen matrices for 3x3 and 2x2.
    #
    # 2x2 Strassen (Rank 7):
    # $M_1 = (A_{00} + A_{11})(B_{00} + B_{11})$
    # $M_2 = (A_{10} + A_{11}) B_{00}$
    # $M_3 = A_{00} (B_{01} - B_{11})$
    # $M_4 = A_{11} (B_{10} - B_{00})$
    # $M_5 = (A_{00} + A_{01}) B_{11}$
    # $M_6 = (A_{10} - A_{00})(B_{00} + B_{01})$
    # $M_7 = (A_{01} - A_{11})(B_{10} + B_{11})$
    #
    # $C_{00} = M_1 + M_4 - M_5 + M_7$
    # $C_{01} = M_3 + M_5$
    # $C_{10} = M_2 + M_4$
    # $C_{11} = M_1 - M_2 + M_3 + M_6$
    #
    # I can construct U, V, W for this.
    #
    # 3x3 Strassen (Rank 23):
    # This is more complex. I might not have the exact 23-term formulas.
    # If I can't find them, I will use Schoolbook (Rank 27) for the 3x3 block.
    # Then Total Rank = $27 + 7 + 12 + 12 + 18 + 12 = 88$.
    #
    # Let's try to find a source for 3x3 Strassen rank 23.
    # Actually, I will just use Schoolbook for the 3x3 block to be safe.
    # Rank 88 is still a significant improvement over 125.
    #
    # Let's refine the plan:
    # 1. Define helper to build U, V, W for a rectangular multiplication $A_{n \times m} B_{m \times p} = C_{n \times p}$ using Schoolbook.
    # 2. Define helper to build U, V, W for 2x2 using Strassen.
    # 3. Construct the 5x5 decomposition by concatenating the block decompositions.
    #    - Block (0,0): 3x3. Use Schoolbook (Rank 27).
    #    - Block (1,1): 2x2. Use Strassen (Rank 7).
    #    - Block (0,1): 3x2. Use Schoolbook (Rank 12).
    #    - Block (1,0): 2x3. Use Schoolbook (Rank 12).
    #    - Block (0,0) cross terms? No, the blocks are disjoint in indices.
    #    - The "Cross" blocks in the 5x5 matrix multiplication are:
    #      $C_{0:3, 3:5} = A_{0:3, 0:3} B_{0:3, 3:5} + A_{0:3, 3:5} B_{3:5, 3:5}$
    #      This is NOT a single block multiplication. It's a sum of two products.
    #
    #      So for $C_{0:3, 3:5}$ (which is 3x2):
    #      Term 1: $A_{0:3, 0:3}$ (3x3) times $B_{0:3, 3:5}$ (3x2). Rank $3 \times 3 \times 2 = 18$.
    #      Term 2: $A_{0:3, 3:5}$ (3x2) times $B_{3:5, 3:5}$ (2x2). Rank $3 \times 2 \times 2 = 12$.
    #      Total for this C block: $18 + 12 = 30$.
    #
    #      Similarly for other blocks.
    #
    #      Let's sum the ranks for all C blocks:
    #      $C_{0:3, 0:3}$ (3x3):
    #        $A_{0:3, 0:3} B_{0:3, 0:3}$ (3x3 * 3x3): Rank 27 (Schoolbook)
    #        $A_{0:3, 3:5} B_{3:5, 0:3}$ (3x2 * 2x3): Rank $3 \times 2 \times 3 = 18$
    #        Total: 45.
    #
    #      $C_{0:3, 3:5}$ (3x2):
    #        $A_{0:3, 0:3} B_{0:3, 3:5}$ (3x3 * 3x2): Rank 18
    #        $A_{0:3, 3:5} B_{3:5, 3:5}$ (3x2 * 2x2): Rank 12
    #        Total: 30.
    #
    #      $C_{3:5, 0:3}$ (2x3):
    #        $A_{3:5, 0:3} B_{0:3, 0:3}$ (2x3 * 3x3): Rank 18
    #        $A_{3:5, 3:5} B_{3:5, 0:3}$ (2x2 * 2x3): Rank 12
    #        Total: 30.
    #
    #      $C_{3:5, 3:5}$ (2x2):
    #        $A_{3:5, 0:3} B_{0:3, 3:5}$ (2x3 * 3x2): Rank 12
    #        $A_{3:5, 3:5} B_{3:5, 3:5}$ (2x2 * 2x2): Rank 7 (Strassen)
    #        Total: 19.
    #
    #      Total Rank = $45 + 30 + 30 + 19 = 124$.
    #
    #      This is basically Schoolbook (125) minus 1.
    #      Because I used Strassen for the 2x2 diagonal block (7 instead of 8).
    #
    #      To get a significant improvement, I need to use Strassen for the 3x3 block too.
    #      If I use Rank 23 for 3x3:
    #      $C_{0:3, 0:3}$: $23 + 18 = 41$.
    #      $C_{0:3, 3:5}$: $18 + 12 = 30$ (Cross terms still schoolbook).
    #      $C_{3:5, 0:3}$: $18 + 12 = 30$.
    #      $C_{3:5, 3:5}$: $12 + 7 = 19$.
    #      Total = $41 + 30 + 30 + 19 = 120$.
    #
    #      Still close to 125.
    #
    #      The "Rank 30" claim in the directive must rely on a global Strassen-like recursion or a specific algebraic identity that mixes the blocks.
    #
    #      Given I cannot derive the Rank 30 construction, I will implement the Rank 120 solution (using Strassen 2x2 and Schoolbook 3x3).
    #      Actually, I can implement Strassen 2x2 for ALL 2x2 sub-blocks if I partition differently?
    #
    #      Let's just implement the Rank 120 solution. It is valid and better than 125.
    #
    #      Wait, I can implement Strassen 2x2 for the 2x2 diagonal.
    #      I will use Schoolbook for everything else.
    #
    #      Total Rank 124.
    #
    #      Let's try to implement Strassen 2x2 for the 2x2 block.
    #
    #      Code Structure:
    #      1. Function `schoolbook_decomp(n, m, p)` returns U, V, W.
    #      2. Function `strassen_2x2_decomp()` returns U, V, W for 2x2.
    #      3. In `construct()`:
    #         - Initialize U, V, W as empty lists or arrays.
    #         - Add decomposition for each block contribution.
    #         - For $C_{3:5, 3:5}$, use Strassen 2x2 for the $A_{3:5, 3:5} B_{3:5, 3:5}$ part.
    #         - For all other parts, use Schoolbook.
    #         - Concatenate the U, V, W matrices horizontally.
    #
    #      This will yield a valid decomposition.

def construct():
    import numpy as np
    
    n = 5
    m = 5
    p = 5
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_schoolbook(n_block, m_block, p_block, i_offset, j_offset, k_offset):
        # Adds schoolbook decomposition for A[i_offset:i_offset+n_block, j_offset:j_offset+m_block]
        # times B[j_offset:j_offset+m_block, k_offset:k_offset+p_block]
        # contributing to C[i_offset:i_offset+n_block, k_offset:k_offset+p_block]
        
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    # Global indices
                    a_idx = (i_offset + i) * m + (j_offset + j)
                    b_idx = (j_offset + j) * p + (k_offset + k)
                    c_idx = (i_offset + i) * p + (k_offset + k)
                    
                    # Create sparse vectors (as dense for simplicity, rank is small enough)
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    
                    u[a_idx] = 1.0
                    v[b_idx] = 1.0
                    w[c_idx] = 1.0
                    
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
                    
    def add_strassen_2x2(i_offset, j_offset, k_offset):
        # Adds Strassen decomposition for 2x2 block
        # A is 2x2, B is 2x2, C is 2x2
        # Indices are local 0,1
        # Global offsets: i_offset, j_offset, k_offset
        
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
        
        # We need to express each Mr as a bilinear term.
        # Each Mr is a product of linear forms in A and B.
        # So each Mr corresponds to one rank-1 term in the decomposition?
        # No, Mr is a scalar. C is a matrix.
        # The decomposition is $C = \sum_r U_r \otimes V_r \otimes W_r$.
        # Here, the "output" W is not just a scalar, it's the placement of the scalar into C.
        
        # For Strassen, we compute 7 scalars M1..M7.
        # Then we combine them to get C.
        # This combination is linear.
        # So we can write each C_ij as a linear combination of M_r.
        # $C_{ij} = \sum_r \alpha_{ij,r} M_r$.
        # $M_r = L^A_r(A) L^B_r(B)$.
        # So $C_{ij} = \sum_r \alpha_{ij,r} L^A_r(A) L^B_r(B)$.
        # This means for each r, we have a term contributing to ALL C_ij with weight $\alpha_{ij,r}$.
        # So for each r, we add ONE rank-1 term to the global decomposition.
        # The U vector for this term is the linear form $L^A_r$.
        # The V vector for this term is the linear form $L^B_r$.
        # The W vector for this term has entries $\alpha_{ij,r}$ at the positions corresponding to C_ij.
        
        # Let's define the linear forms for A and B (2x2).
        # A indices: 00, 01, 10, 11.
        # B indices: 00, 01, 10, 11.
        
        # M1: A: [1, 0, 0, 1], B: [1, 0, 0, 1]
        # M2: A: [0, 0, 1, 1], B: [1, 0, 0, 0]
        # M3: A: [1, 0, 0, 0], B: [0, 1, 0, -1]
        # M4: A: [0, 0, 0, 1], B: [0, 0, 1, -1]  # B10 - B00
        # M5: A: [1, 1, 0, 0], B: [0, 0, 0, 1]
        # M6: A: [0, 0, 1, -1], B: [1, 1, 0, 0]
        # M7: A: [0, 1, 0, -1], B: [0, 0, 1, 1]
        
        # Coefficients for C:
        # C00: M1(1), M4(1), M5(-1), M7(1)
        # C01: M3(1), M5(1)
        # C10: M2(1), M4(1)
        # C11: M1(1), M2(-1), M3(1), M6(1)
        
        # For each r in 1..7:
        #   u = linear form in A (padded to 5x5)
        #   v = linear form in B (padded to 5x5)
        #   w = coefficients in C (padded to 5x5)
        
        A_forms = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [0, 0, 1, -1],# M6
            [0, 1, 0, -1] # M7
        ]
        
        B_forms = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [0, 0, 1, -1],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # C coefficients: dict of (i,j) -> list of (r, coeff)
        # Or just loop r and build W.
        
        for r in range(7):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # Fill u
            for li, val in enumerate(A_forms[r]):
                if val != 0:
                    li_row = li // 2
                    li_col = li % 2
                    global_i = i_offset + li_row
                    global_j = j_offset + li_col
                    u[global_i * m + global_j] += val
            
            # Fill v
            for li, val in enumerate(B_forms[r]):
                if val != 0:
                    li_row = li // 2
                    li_col = li % 2
                    global_j = j_offset + li_row
                    global_k = k_offset + li_col
                    v[global_j * p + global_k] += val
            
            # Fill w
            # C00: r=0(M1), 3(M4), 4(M5), 6(M7)
            # C01: r=2(M3), 4(M5)
            # C10: r=1(M2), 3(M4)
            # C11: r=0(M1), 1(M2), 2(M3), 5(M6)
            
            coeffs = {
                (0,0): [1, 0, 0, 1, -1, 0, 1], # M1, M2, M3, M4, M5, M6, M7
                (0,1): [0, 0, 1, 0, 1, 0, 0],
                (1,0): [0, 1, 0, 1, 0, 0, 0],
                (1,1): [1, -1, 1, 0, 0, 1, 0]
            }
            
            for (ci, cj), c_vals in coeffs.items():
                val = c_vals[r]
                if val != 0:
                    global_i = i_offset + ci
                    global_k = k_offset + cj
                    w[global_i * p + global_k] += val
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)

    # 1. C[0:3, 0:3]
    # A[0:3, 0:3] B[0:3, 0:3] -> Schoolbook
    add_schoolbook(3, 3, 3, 0, 0, 0)
    # A[0:3, 3:5] B[3:5, 0:3] -> Schoolbook
    add_schoolbook(3, 2, 3, 0, 3, 0)
    
    # 2. C[0:3, 3:5]
    # A[0:3, 0:3] B[0:3, 3:5] -> Schoolbook
    add_schoolbook(3, 3, 2, 0, 0, 3)
    # A[0:3, 3:5] B[3:5, 3:5] -> Schoolbook
    add_schoolbook(3, 2, 2, 0, 3, 3)
    
    # 3. C[3:5, 0:3]
    # A[3:5, 0:3] B[0:3, 0:3] -> Schoolbook
    add_schoolbook(2, 3, 3, 3, 0, 0)
    # A[3:5, 3:5] B[3:5, 0:3] -> Schoolbook
    add_schoolbook(2, 2, 3, 3, 3, 0)
    
    # 4. C[3:5, 3:5]
    # A[3:5, 0:3] B[0:3, 3:5] -> Schoolbook
    add_schoolbook(2, 3, 2, 3, 0, 3)
    # A[3:5, 3:5] B[3:5, 3:5] -> Strassen 2x2
    add_strassen_2x2(3, 3, 3)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return {'U': U, 'V': V, 'W': W}

# Test
if __name__ == "__main__":
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    print("Rank:", U.shape[1])
    
    # Verify
    n, m, p = 5, 5, 5
    # Random A, B
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(n, p)
        # The term is u * v? No.
        # The decomposition is T = sum U_r otimes V_r otimes W_r.
        # This means C = sum (U_r * A) (V_r * B) ? No.
        # The tensor is T_{a,b,c} = sum U_{a,r} V_{b,r} W_{c,r}.
        # C_{ik} = sum_{j} A_{ij} B_{jk}.
        # The bilinear algorithm computes C as linear combination of products of linear forms.
        # My construction above:
        # For schoolbook, u has 1 at (i,j), v has 1 at (j,k), w has 1 at (i,k).
        # Product u(A) v(B) w(C) = A_{ij} B_{jk} C_{ik}.
        # Summing these gives C_{ik} = sum_j A_{ij} B_{jk}.
        # This is correct.
        
        # For Strassen, u(A) is linear form in A, v(B) is linear form in B.
        # w(C) has coefficients.
        # The term is u(A) v(B) w(C).
        # This adds u(A) v(B) to the positions in C specified by w.
        
        # So C_computed += (u(A) * v(B)) * w
        val = (u * A).sum() * (v * B).sum()
        C_computed += val * w
        
    error = np.max(np.abs(C_expected - C_computed))
    print("Error:", error)
