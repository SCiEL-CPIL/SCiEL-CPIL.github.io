import numpy as np

def construct():
    n, m, p = 5, 5, 5
    # Total rank will be 124
    # U: (25, R), V: (25, R), W: (25, R)
    
    # We will build the columns one by one
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        """
        Adds a rank-1 term.
        u_idx: list of indices in U (size 25) with non-zero values
        v_idx: list of indices in V (size 25) with non-zero values
        w_idx: list of indices in W (size 25) with non-zero values
        """
        U = [0.0] * 25
        V = [0.0] * 25
        W = [0.0] * 25
        for idx in u_idx:
            U[idx] = u_val
        for idx in v_idx:
            V[idx] = v_val
        for idx in w_idx:
            W[idx] = w_val
        U_cols.append(U)
        V_cols.append(V)
        W_cols.append(W)
        
    def add_standard_block(a_rows, b_cols, c_rows, c_cols):
        """
        Adds the standard decomposition for A[a_rows, b_cols] * B[b_cols, c_cols] -> C[c_rows, c_cols]
        a_rows: range of i
        b_cols: range of j
        c_rows: range of i (same as a_rows)
        c_cols: range of k
        """
        for i in a_rows:
            for j in b_cols:
                for k in c_cols:
                    # Term: A[i,j] * B[j,k]
                    # U has 1 at i
                    # V has 1 at j
                    # W has 1 at k
                    add_term([i], [j], [k])
                    
    # Block definitions
    # Rows/Cols 0,1 are "2" block
    # Rows/Cols 2,3,4 are "3" block
    
    # 1. C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:5] * B[2:5, 0:2]
    
    # Part 1a: A[0:2, 0:2] * B[0:2, 0:2] using Strassen
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices in global space:
    # A00: i=0, j=0
    # A11: i=1, j=1
    # B00: j=0, k=0
    # B11: j=1, k=1
    # etc.
    
    # M1
    add_term([0, 1], [0, 1], [0, 1]) # (A00+A11) is not a single index, it's a sum of terms?
    # Wait, the Strassen algorithm computes linear combinations of A and B first, then multiplies.
    # The decomposition is a sum of rank-1 tensors.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This is 4 rank-1 terms. But Strassen combines them into 1 "multiplication" step.
    # In the tensor decomposition sense, a "multiplication" step $X \cdot Y$ where $X$ is a linear combo of A's and $Y$ is a linear combo of B's corresponds to a set of rank-1 terms?
    # No. The tensor decomposition rank counts the number of rank-1 tensors $u \otimes v \otimes w$.
    # If we use Strassen, we replace 8 rank-1 terms with 7 "groups".
    # But each "group" $M = (\sum c_i A_i)(\sum d_j B_j)$ expands to $\sum_{i,j} c_i d_j A_i B_j$.
    # This is NOT a single rank-1 tensor in the original space unless the coefficients are 1 and indices are single.
    # 
    # CORRECTION:
    # The rank of a tensor is the minimum number of rank-1 tensors to sum to it.
    # Strassen's algorithm shows that the tensor $M_{2,2,2}$ can be written as a sum of 7 rank-1 tensors in a *different* basis?
    # No, Strassen's algorithm is a circuit. It uses 7 multiplications.
    # Each multiplication $M_k = (\sum_i \alpha_{ki} A_i) (\sum_j \beta_{kj} B_j)$
    # The result $C$ is $\sum_k \gamma_{ik} M_k$.
    # This corresponds to a decomposition of the tensor into 7 terms of the form:
    # $T_k = (\sum_i \alpha_{ki} e_i) \otimes (\sum_j \beta_{kj} e_j) \otimes (\sum_l \gamma_{lk} e_l)$
    # This is a rank-1 tensor in the *extended* space?
    # No, the tensor $T$ is in $V_A \otimes V_B \otimes V_C$.
    # A rank-1 tensor is $u \otimes v \otimes w$ where $u \in V_A, v \in V_B, w \in V_C$.
    # Strassen shows that $M_{2,2,2}$ has rank 7.
    # The 7 vectors are:
    # $u_1 = e_{00} + e_{11}$
    # $v_1 = e_{00} + e_{11}$
    # $w_1 = e_{00} + e_{11} - e_{01} + e_{10}$ (Wait, $C_{00} = M_1 + M_4 - M_5 + M_7$)
    
    # So we need to add these 7 specific rank-1 terms for the $2 \times 2$ block.
    
    # Let's define the Strassen terms for the top-left 2x2 block (indices 0,1)
    
    # Term 1
    add_term([0, 1], [0, 1], [0, 1, 0, 1]) # This is wrong syntax for add_term
    # My add_term expects single values or lists.
    # $u_1 = e_0 + e_1$ (in A space, indices 0,1)
    # $v_1 = e_0 + e_1$ (in B space, indices 0,1)
    # $w_1 = e_0 + e_1 - e_0 + e_1$? No.
    # $C_{00} = M_1 + M_4 - M_5 + M_7$
    # $C_{01} = M_3 + M_5$
    # $C_{10} = M_2 + M_4$
    # $C_{11} = M_1 - M_2 + M_3 + M_6$
    
    # The $w$ vector for $M_1$ has components:
    # $w_{1,00} = 1$
    # $w_{1,01} = 0$
    # $w_{1,10} = 0$
    # $w_{1,11} = 1$
    # So $w_1 = e_{00} + e_{11}$ in C space (indices 0, 3 for 2x2? No, global indices 0, 1, 2, 3? No, C is 5x5. Block is 0:2, 0:2. Indices 0, 1.)
    
    # Let's refine add_term to accept coefficients.
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    def add_rank1(u_dict, v_dict, w_dict):
        U = [0.0] * 25
        V = [0.0] * 25
        W = [0.0] * 25
        for idx, val in u_dict.items():
            U[idx] = val
        for idx, val in v_dict.items():
            V[idx] = val
        for idx, val in w_dict.items():
            W[idx] = val
        U_cols.append(U)
        V_cols.append(V)
        W_cols.append(W)

    # Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Indices: A(i,j) -> i*5+j. B(j,k) -> j*5+k. C(i,k) -> i*5+k.
    
    # M1 = (A00+A11)(B00+B11)
    add_rank1({0: 1, 6: 1}, {0: 1, 6: 1}, {0: 1, 1: 0, 5: 0, 6: 1}) # C00=1, C11=1
    # Wait, C indices: 0 (0,0), 1 (0,1), 5 (1,0), 6 (1,1).
    # w1: C00=1, C11=1.
    add_rank1({0: 1, 6: 1}, {0: 1, 6: 1}, {0: 1, 6: 1})
    
    # M2 = (A10+A11)B00
    # A10 is idx 5, A11 is idx 6. B00 is idx 0.
    # w2: C10=1 (idx 5).
    add_rank1({5: 1, 6: 1}, {0: 1}, {5: 1})
    
    # M3 = A00(B10-B11)
    # A00 is idx 0. B10 is idx 5, B11 is idx 6.
    # w3: C01=1 (idx 1).
    add_rank1({0: 1}, {5: 1, 6: -1}, {1: 1})
    
    # M4 = A11(B01-B00)
    # A11 is idx 6. B01 is idx 1, B00 is idx 0.
    # w4: C10=1 (idx 5), C00=1 (idx 0).
    add_rank1({6: 1}, {1: 1, 0: -1}, {5: 1, 0: 1})
    
    # M5 = (A00+A01)B11
    # A00 is 0, A01 is 1. B11 is 6.
    # w5: C01=1 (idx 1), C00=-1 (idx 0).
    add_rank1({0: 1, 1: 1}, {6: 1}, {1: 1, 0: -1})
    
    # M6 = (A10-A00)(B00+B01)
    # A10 is 5, A00 is 0. B00 is 0, B01 is 1.
    # w6: C11=1 (idx 6).
    add_rank1({5: 1, 0: -1}, {0: 1, 1: 1}, {6: 1})
    
    # M7 = (A01-A11)(B10+B11)
    # A01 is 1, A11 is 6. B10 is 5, B11 is 6.
    # w7: C00=1 (idx 0), C11=1 (idx 6).
    add_rank1({1: 1, 6: -1}, {5: 1, 6: 1}, {0: 1, 6: 1})
    
    # Part 1b: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # i in 0,1. j in 2,3,4. k in 0,1.
    add_standard_block([0, 1], [2, 3, 4], [0, 1], [0, 1])
    
    # 2. C[0:2, 2:5] = A[0:2, 0:2] * B[0:2, 2:5] + A[0:2, 2:5] * B[2:5, 2:5]
    # A[0:2, 0:2] (2x2) * B[0:2, 2:5] (2x3) -> 2x3
    # Standard: 2*2*3 = 12
    add_standard_block([0, 1], [0, 1], [0, 1], [2, 3, 4])
    
    # A[0:2, 2:5] (2x3) * B[2:5, 2:5] (3x3) -> 2x3
    # Standard: 2*3*3 = 18
    add_standard_block([0, 1], [2, 3, 4], [0, 1], [2, 3, 4])
    
    # 3. C[2:5, 0:2] = A[2:5, 0:2] * B[0:2, 0:2] + A[2:5, 2:5] * B[2:5, 0:2]
    # A[2:5, 0:2] (3x2) * B[0:2, 0:2] (2x2) -> 3x2
    # Standard: 3*2*2 = 12
    add_standard_block([2, 3, 4], [0, 1], [2, 3, 4], [0, 1])
    
    # A[2:5, 2:5] (3x3) * B[2:5, 0:2] (3x2) -> 3x2
    # Standard: 3*3*2 = 18
    add_standard_block([2, 3, 4], [2, 3, 4], [2, 3, 4], [0, 1])
    
    # 4. C[2:5, 2:5] = A[2:5, 0:2] * B[0:2, 2:5] + A[2:5, 2:5] * B[2:5, 2:5]
    # A[2:5, 0:2] (3x2) * B[0:2, 2:5] (2x3) -> 3x3
    # Standard: 3*2*3 = 18
    add_standard_block([2, 3, 4], [0, 1], [2, 3, 4], [2, 3, 4])
    
    # A[2:5, 2:5] (3x3) * B[2:5, 2:5] (3x3) -> 3x3
    # Standard: 3*3*3 = 27
    add_standard_block([2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4])
    
    U = np.array(U_cols)
    V = np.array(V_cols)
    W = np.array(W_cols)
    
    # Verify shape
    # U: (25, R)
    # V: (25, R)
    # W: (25, R)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
