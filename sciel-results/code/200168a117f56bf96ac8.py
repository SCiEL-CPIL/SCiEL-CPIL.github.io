import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 4x4x4 using nested Strassen, 
    # then add 4 terms for the 5th column.
    
    # Helper to create a linear form dictionary: { (i, j): coeff }
    # But we need to output U, V, W matrices.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # Note: The problem statement says W[k*n+i, r].
    
    rank = 49 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 multiplication
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Outer Strassen terms (Block level)
    # A blocks: 00, 01, 10, 11 corresponding to I,J in {0,1}
    # B blocks: 00, 01, 10, 11 corresponding to J,K in {0,1}
    
    # We define the 7 outer terms as linear combinations of blocks.
    # Each block is a 2x2 matrix of variables.
    # So an outer term is a 2x2 matrix of linear forms in A and a 2x2 matrix of linear forms in B.
    
    # Let's define the 7 outer linear form matrices for A and B.
    # A_terms[t] is a 2x2 matrix where entry (I,J) is a linear form in A.
    # Actually, Strassen combines blocks.
    # P1: A part is A00 + A11. B part is B00 + B11.
    # P2: A part is A10 + A11. B part is B00.
    # P3: A part is A00. B part is B01 - B11.
    # P4: A part is A11. B part is B10 - B00.
    # P5: A part is A00 + A01. B part is B11.
    # P6: A part is A10 - A00. B part is B00 + B01.
    # P7: A part is A01 - A11. B part is B10 + B11.
    
    # Note: Indices (I,J) for A blocks:
    # A00 is I=0, J=0. A01 is I=0, J=1. A10 is I=1, J=0. A11 is I=1, J=1.
    # Indices (J,K) for B blocks:
    # B00 is J=0, K=0. B01 is J=0, K=1. B10 is J=1, K=0. B11 is J=1, K=1.
    
    # We will iterate through the 7 outer terms.
    # For each outer term, we have a 2x2 matrix of linear forms for A and a 2x2 matrix of linear forms for B.
    # We need to decompose the product of these two 2x2 matrices of linear forms into 7 scalar products.
    
    # Let's define the 7 inner Strassen terms for a 2x2 matrix of linear forms.
    # If X and Y are 2x2 matrices of linear forms, we compute 7 products X_inner * Y_inner.
    
    # To make it easier, let's just generate the 49 terms directly.
    # For each of the 7 outer terms (t_out), we have:
    # A_mat = 2x2 matrix of linear forms
    # B_mat = 2x2 matrix of linear forms
    # We apply inner Strassen to A_mat and B_mat.
    
    # Inner Strassen terms for matrices X, Y:
    # Q1 = (X00+X11)(Y00+Y11)
    # Q2 = (X10+X11)Y00
    # Q3 = X00(Y01-Y11)
    # Q4 = X11(Y10-Y00)
    # Q5 = (X00+X01)Y11
    # Q6 = (X10-X00)(Y00+Y01)
    # Q7 = (X01-X11)(Y10+Y11)
    
    # Each Q is a product of two linear forms.
    # So each outer term generates 7 scalar products.
    
    # We need to accumulate the contribution of each scalar product to the final C_ik.
    # C is a 4x4 matrix (for the first 4 columns).
    # C_ik = sum over outer terms t_out of (Inner Strassen Result for t_out)[K, I]
    # Wait, Strassen output formulas:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So for each outer term t_out, the result is a 2x2 matrix of bilinear forms (sums of scalar products).
    # Let R_out[t_out] be a 2x2 matrix where entry (K, I) is the sum of scalar products contributing to C_{K, I} for that outer term.
    # Then C_{K, I} = sum_{t_out} R_out[t_out][K, I].
    
    # Let's build the U, V, W matrices.
    # Each scalar product corresponds to a column in U, V, W.
    
    # Define the outer terms' A and B block linear forms.
    # We represent a linear form in A as a dict {(i,j): coeff}
    # We represent a linear form in B as a dict {(j,k): coeff}
    
    # Outer Term Definitions
    # t_out 0: P1
    # A: A00 + A11. 
    # A00 is block (I=0, J=0). A11 is block (I=1, J=1).
    # So A_mat[0,0] = A00 (linear forms), A_mat[1,1] = A11. Others 0.
    # Wait, Strassen adds the *blocks*.
    # So A_mat for P1 is:
    # [ A00, 0 ]
    # [ 0, A11 ]
    # No, P1 = (A00+A11)(B00+B11).
    # This means the linear form for A is the sum of the blocks A00 and A11?
    # No, Strassen's P1 is $(A_{11}+A_{22})(B_{11}+B_{22})$.
    # Here $A_{11}$ is the top-left block. $A_{22}$ is the bottom-right block.
    # So the "matrix" of linear forms for A in P1 is:
    # [ A00, 0 ]
    # [ 0, A11 ]
    # This is not a sum. The product is of the *sums* of the blocks?
    # No, Strassen defines $P_1 = (A_{11} + A_{22}) (B_{11} + B_{22})$.
    # This is a product of two $2 \times 2$ matrices: $M_A = A_{11} + A_{22}$ and $M_B = B_{11} + B_{22}$.
    # $M_A$ is a $2 \times 2$ matrix where the top-left entry is the sum of the top-left entries of $A_{11}$ and $A_{22}$?
    # No. $A_{11}$ is a $2 \times 2$ matrix. $A_{22}$ is a $2 \times 2$ matrix.
    # $A_{11} + A_{22}$ is a $2 \times 2$ matrix.
    # So $M_A$ is a $2 \times 2$ matrix of linear forms.
    # $M_A[0,0] = A_{00} + A_{20}$? No.
    # $A_{11}$ is the block at $I=0, J=0$. Its entries are $A_{00}, A_{01}, A_{10}, A_{11}$.
    # $A_{22}$ is the block at $I=1, J=1$. Its entries are $A_{20}, A_{21}, A_{30}, A_{31}$.
    # So $M_A[0,0] = A_{00} + A_{20}$? No, the indices are global.
    # $A_{11}$ block entries: $A_{00}, A_{01}, A_{10}, A_{11}$.
    # $A_{22}$ block entries: $A_{20}, A_{21}, A_{30}, A_{31}$.
    # Sum: $A_{00}+A_{20}$? No, $A_{00}$ and $A_{20}$ are different rows.
    # $A_{11}$ is rows 0,1 cols 0,1. $A_{22}$ is rows 2,3 cols 2,3.
    # So $M_A$ is a $2 \times 2$ matrix of $2 \times 2$ matrices? No.
    # Strassen operates on the *blocks* as if they were scalars.
    # So $P_1$ is the product of the block-matrix $M_A$ and $M_B$.
    # $M_A = \begin{pmatrix} A_{11} & 0 \\ 0 & A_{22} \end{pmatrix}$? No.
    # $M_A = A_{11} + A_{22}$ is not a block matrix sum.
    # It is the sum of the two blocks, placed in the same position?
    # No, $A_{11}$ and $A_{22}$ are disjoint blocks.
    # $M_A$ is a $2 \times 2$ matrix where the (0,0) entry is the block $A_{11}$?
    # No. The formula is $P_1 = (A_{11} + A_{22})(B_{11} + B_{22})$.
    # This implies we form a new $2 \times 2$ matrix $S_A$ where $S_A = A_{11} + A_{22}$.
    # But $A_{11}$ and $A_{22}$ are $2 \times 2$ matrices. Their sum is a $2 \times 2$ matrix.
    # So $S_A$ is a $2 \times 2$ matrix of linear forms.
    # $S_A[0,0] = A_{00} + A_{20}$? No.
    # $A_{11}$ is the top-left block. $A_{22}$ is the bottom-right block.
    # $A_{11} + A_{22}$ is not defined as a single block sum unless we shift them.
    # Ah, Strassen's algorithm is for $2 \times 2$ matrices $A, B$.
    # $A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$.
    # $P_1 = (A_{11} + A_{22})(B_{11} + B_{22})$.
    # Here $A_{11}$ is the top-left element of the $2 \times 2$ block matrix.
    # If the "elements" are themselves $2 \times 2$ matrices, then $A_{11}$ is the top-left $2 \times 2$ block of $A$.
    # $A_{22}$ is the bottom-right $2 \times 2$ block of $A$.
    # $A_{11} + A_{22}$ is the sum of these two $2 \times 2$ matrices.
    # So $S_A$ is a $2 \times 2$ matrix where each entry is the sum of the corresponding entries of the top-left and bottom-right blocks.
    # $S_A[0,0] = A_{00} + A_{20}$? No.
    # $A_{11}$ entries: $A_{00}, A_{01}, A_{10}, A_{11}$.
    # $A_{22}$ entries: $A_{20}, A_{21}, A_{30}, A_{31}$.
    # $S_A[0,0] = A_{00} + A_{20}$? No, $A_{20}$ is row 2, col 0. $A_{11}$ is row 0, col 0.
    # The sum $A_{11} + A_{22}$ is a $2 \times 2$ matrix $M$ where $M[0,0] = A_{00} + A_{20}$?
    # No. $A_{11}$ is a $2 \times 2$ matrix. $A_{22}$ is a $2 \times 2$ matrix.
    # $A_{11} + A_{22}$ is a $2 \times 2$ matrix.
    # Entry (0,0) of the sum is $A_{11}[0,0] + A_{22}[0,0] = A_{00} + A_{20}$?
    # $A_{22}$ is the block at $I=1, J=1$. Its local index (0,0) is global (2,2).
    # So $A_{22}[0,0] = A_{22}$.
    # $A_{11}[0,0] = A_{00}$.
    # So $S_A[0,0] = A_{00} + A_{22}$.
    # $S_A[0,1] = A_{01} + A_{23}$.
    # $S_A[1,0] = A_{10} + A_{32}$.
    # $S_A[1,1] = A_{11} + A_{33}$.
    
    # So, for each outer term, we construct $S_A$ and $S_B$ as $2 \times 2$ matrices of linear forms.
    # Then we apply inner Strassen to $S_A$ and $S_B$.
    
    # Inner Strassen on $S_A, S_B$:
    # $Q_1 = (S_A[0,0] + S_A[1,1])(S_B[0,0] + S_B[1,1])$
    # $Q_2 = (S_A[1,0] + S_A[1,1])S_B[0,0]$
    # $Q_3 = S_A[0,0](S_B[0,1] - S_B[1,1])$
    # $Q_4 = S_A[1,1](S_B[1,0] - S_B[0,0])$
    # $Q_5 = (S_A[0,0] + S_A[0,1])S_B[1,1]$
    # $Q_6 = (S_A[1,0] - S_A[0,0])(S_B[0,0] + S_B[0,1])$
    # $Q_7 = (S_A[0,1] - S_A[1,1])(S_B[1,0] + S_B[1,1])$
    
    # Each $Q$ is a product of two linear forms.
    # We add these to U, V, W.
    
    # Then we update W for the output C.
    # $C_{00} = P_1 + P_4 - P_5 + P_7$
    # $C_{01} = P_3 + P_5$
    # $C_{10} = P_2 + P_4$
    # $C_{11} = P_1 - P_2 + P_3 + P_6$
    # This is for the block output.
    # The block output $C_{IK}$ is a $2 \times 2$ matrix.
    # $C_{IK}[a,b]$ is the entry at local indices $a,b$.
    # Global $i = 2I+a, k = 2K+b$.
    
    # We need to map the inner Strassen results $Q_1 \dots Q_7$ to the block output entries.
    # The inner Strassen computes the product $S_A S_B$.
    # Let $P_{out} = S_A S_B$.
    # $P_{out}[0,0] = Q_1 + Q_4 - Q_5 + Q_7$
    # $P_{out}[0,1] = Q_3 + Q_5$
    # $P_{out}[1,0] = Q_2 + Q_4$
    # $P_{out}[1,1] = Q_1 - Q_2 + Q_3 + Q_6$
    
    # So for each outer term $t_{out}$, we have a $2 \times 2$ matrix of bilinear forms $P_{out}$.
    # This $P_{out}$ contributes to the block $C_{IK}$ corresponding to the outer term.
    # Wait, the outer Strassen formulas give $C_{IK}$ as a sum of outer terms.
    # $C_{00} = P_1 + P_4 - P_5 + P_7$
    # Here $P_1$ is the result of the outer term 1, which is a $2 \times 2$ matrix of bilinear forms.
    # So $C_{00}[a,b] = (P_1[a,b] + P_4[a,b] - P_5[a,b] + P_7[a,b])$.
    # And $C_{00}[a,b]$ is $C_{2*0+a, 2*0+b} = C_{a,b}$.
    
    # So the algorithm is:
    # 1. For each of the 7 outer terms, compute the 7 inner products (scalar products).
    # 2. These 7 scalar products form the $2 \times 2$ matrix $P_{out}$ for that outer term.
    # 3. Accumulate these $P_{out}$ matrices into the 4 $2 \times 2$ block outputs $C_{00}, C_{01}, C_{10}, C_{11}$ using the outer Strassen formulas.
    # 4. Map the block outputs to global $C_{ik}$.
    
    # Let's implement this.
    
    # Helper to add a scalar product to U, V, W
    def add_product(u_form, v_form, w_coeffs):
        # u_form: dict {(i,j): coeff}
        # v_form: dict {(j,k): coeff}
        # w_coeffs: dict {(k,i): coeff}
        nonlocal col
        for (i,j), c in u_form.items():
            if c != 0:
                U[i*m+j][col] += c
        for (j,k), c in v_form.items():
            if c != 0:
                V[j*p+k][col] += c
        for (k,i), c in w_coeffs.items():
            if c != 0:
                W[k*n+i][col] += c
        col += 1

    # Define the 7 outer terms' S_A and S_B matrices
    # S_A is 2x2, S_B is 2x2. Each entry is a linear form dict.
    
    # A blocks:
    # A00: I=0, J=0 -> i in 0,1; j in 0,1
    # A01: I=0, J=1 -> i in 0,1; j in 2,3
    # A10: I=1, J=0 -> i in 2,3; j in 0,1
    # A11: I=1, J=1 -> i in 2,3; j in 2,3
    
    # B blocks:
    # B00: J=0, K=0 -> j in 0,1; k in 0,1
    # B01: J=0, K=1 -> j in 0,1; k in 2,3
    # B10: J=1, K=0 -> j in 2,3; k in 0,1
    # B11: J=1, K=1 -> j in 2,3; k in 2,3
    
    def get_block_A(I, J):
        d = {}
        for a in range(2):
            for b in range(2):
                i = 2*I + a
                j = 2*J + b
                d[(i,j)] = 1
        return d

    def get_block_B(J, K):
        d = {}
        for b in range(2):
            for c in range(2):
                j = 2*J + b
                k = 2*K + c
                d[(j,k)] = 1
        return d

    def add_dict(d1, d2, sign=1):
        d = {}
        for key, val in d1.items():
            d[key] = d.get(key, 0) + val
        for key, val in d2.items():
            d[key] = d.get(key, 0) + sign * val
        # Remove zeros
        return {k: v for k, v in d.items() if v != 0}

    def sub_dict(d1, d2):
        return add_dict(d1, d2, -1)

    # Outer Terms
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B01-B11)
    # P4: A11(B10-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    A00 = get_block_A(0,0)
    A01 = get_block_A(0,1)
    A10 = get_block_A(1,0)
    A11 = get_block_A(1,1)
    
    B00 = get_block_B(0,0)
    B01 = get_block_B(0,1)
    B10 = get_block_B(1,0)
    B11 = get_block_B(1,1)
    
    outer_terms = [
        (add_dict(A00, A11), add_dict(B00, B11)), # P1
        (add_dict(A10, A11), B00),                # P2
        (A00, sub_dict(B01, B11)),                # P3
        (A11, sub_dict(B10, B00)),                # P4
        (add_dict(A00, A01), B11),                # P5
        (sub_dict(A10, A00), add_dict(B00, B01)), # P6
        (sub_dict(A01, A11), add_dict(B10, B11))  # P7
    ]
    
    # For each outer term, we have S_A and S_B as dicts of linear forms?
    # No, S_A is a 2x2 matrix of linear forms.
    # But in the outer terms above, S_A is a *single* linear form?
    # No. $A_{11} + A_{22}$ is a $2 \times 2$ matrix of linear forms.
    # My `add_dict` adds the dicts, which represents the sum of the blocks.
    # So `add_dict(A00, A11)` is the $2 \times 2$ matrix $S_A$ for P1.
    # Correct.
    
    # Now, for each outer term, we apply inner Strassen to S_A and S_B.
    # S_A is a dict {(i,j): coeff}. This is a $2 \times 2$ matrix of linear forms?
    # No, the dict keys are global (i,j).
    # The "matrix" structure is implicit in the indices.
    # To apply inner Strassen, we need to treat S_A as a $2 \times 2$ matrix of linear forms.
    # Let's extract the 4 "quadrants" of S_A.
    # S_A[0,0] is the linear form for local (0,0).
    # S_A[0,1] is the linear form for local (0,1).
    # etc.
    
    def extract_quadrants(form_dict):
        # Returns 4 dicts: q00, q01, q10, q11
        q00 = {}
        q01 = {}
        q10 = {}
        q11 = {}
        for (i,j), c in form_dict.items():
            # i, j are global indices.
            # Local indices a, b are i%2, j%2.
            a = i % 2
            b = j % 2
            if a == 0 and b == 0:
                q00[(i,j)] = c
            elif a == 0 and b == 1:
                q01[(i,j)] = c
            elif a == 1 and b == 0:
                q10[(i,j)] = c
            else:
                q11[(i,j)] = c
        return q00, q01, q10, q11

    # Inner Strassen terms for S_A, S_B
    # Q1 = (S_A00 + S_A11)(S_B00 + S_B11)
    # Q2 = (S_A10 + S_A11)S_B00
    # Q3 = S_A00(S_B01 - S_B11)
    # Q4 = S_A11(S_B10 - S_B00)
    # Q5 = (S_A00 + S_A01)S_B11
    # Q6 = (S_A10 - S_A00)(S_B00 + S_B01)
    # Q7 = S_A01(S_B10 + S_B11)
    
    # Output mapping for inner Strassen (Product P_out = S_A S_B)
    # P_out[0,0] = Q1 + Q4 - Q5 + Q7
    # P_out[0,1] = Q3 + Q5
    # P_out[1,0] = Q2 + Q4
    # P_out[1,1] = Q1 - Q2 + Q3 + Q6
    
    # Outer Strassen output mapping (C = A B)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to accumulate the contributions to C_ik.
    # C_ik is determined by the outer block (I,K) and inner local (a,b).
    # i = 2I+a, k = 2K+b.
    
    # For each outer term t_out, we compute 7 inner products.
    # Each inner product is a scalar product u*v.
    # This scalar product contributes to P_out[a,b] for that outer term.
    # Then P_out[a,b] contributes to C[2I+a, 2K+b] for the outer blocks I,K.
    
    # Let's create a matrix W_contrib of size 4x4 for each scalar product?
    # No, each scalar product contributes to specific C_ik.
    
    # Let's iterate through outer terms.
    for t_out_idx, (S_A, S_B) in enumerate(outer_terms):
        # Extract quadrants
        SA00, SA01, SA10, SA11 = extract_quadrants(S_A)
        SB00, SB01, SB10, SB11 = extract_quadrants(S_B)
        
        # Define inner products
        # Each is a pair (u_form, v_form)
        inner_products = [
            (add_dict(SA00, SA11), add_dict(SB00, SB11)), # Q1
            (add_dict(SA10, SA11), SB00),                 # Q2
            (SA00, sub_dict(SB01, SB11)),                 # Q3
            (SA11, sub_dict(SB10, SB00)),                 # Q4
            (add_dict(SA00, SA01), SB11),                 # Q5
            (sub_dict(SA10, SA00), add_dict(SB00, SB01)), # Q6
            (sub_dict(SA01, SA11), add_dict(SB10, SB11))  # Q7
        ]
        
        # For each inner product, determine which C_ik it contributes to.
        # Q1 contributes to P_out[0,0] and P_out[1,1].
        # Q2 contributes to P_out[1,0] and P_out[1,1].
        # Q3 contributes to P_out[0,1] and P_out[1,1].
        # Q4 contributes to P_out[0,0] and P_out[1,0].
        # Q5 contributes to P_out[0,0] and P_out[0,1].
        # Q6 contributes to P_out[1,1].
        # Q7 contributes to P_out[0,0].
        
        # Coefficients for P_out entries:
        # P_out[0,0]: Q1(+1), Q4(+1), Q5(-1), Q7(+1)
        # P_out[0,1]: Q3(+1), Q5(+1)
        # P_out[1,0]: Q2(+1), Q4(+1)
        # P_out[1,1]: Q1(+1), Q2(-1), Q3(+1), Q6(+1)
        
        coeff_map = {
            0: {0: 1, 3: 1, 4: -1, 6: 1}, # Q1, Q4, Q5, Q7
            1: {2: 1, 4: 1},               # Q3, Q5
            2: {1: 1, 3: 1},               # Q2, Q4
            3: {0: 1, 1: -1, 2: 1, 5: 1}   # Q1, Q2, Q3, Q6
        }
        
        for q_idx in range(7):
            u_form, v_form = inner_products[q_idx]
            # Determine which P_out entries this Q contributes to
            # And with what coefficient
            # Then map P_out entries to C_ik
            # P_out[a,b] corresponds to local indices a,b.
            # This P_out is for the outer term t_out.
            # The outer term t_out contributes to C blocks with coefficients given by outer Strassen.
            
            # Outer Strassen coefficients for P_t_out in C_IK:
            # C00: P1(+1), P4(+1), P5(-1), P7(+1)
            # C01: P3(+1), P5(+1)
            # C10: P2(+1), P4(+1)
            # C11: P1(+1), P2(-1), P3(+1), P6(+1)
            
            outer_coeff_map = {
                0: {0: 1, 3: 1, 4: -1, 6: 1}, # P1, P4, P5, P7 -> C00
                1: {2: 1, 4: 1},               # P3, P5 -> C01
                2: {1: 1, 3: 1},               # P2, P4 -> C10
                3: {0: 1, 1: -1, 2: 1, 5: 1}   # P1, P2, P3, P6 -> C11
            }
            
            # Wait, the outer term index t_out_idx is 0..6 corresponding to P1..P7.
            # The keys in outer_coeff_map are C block indices (0,1,2,3 for C00, C01, C10, C11).
            # The values are dicts {P_idx: coeff}.
            # So for a given t_out_idx, we find which C blocks it contributes to.
            
            w_coeffs = {}
            for c_block_idx, p_coeffs in outer_coeff_map.items():
                if t_out_idx in p_coeffs:
                    outer_coeff = p_coeffs[t_out_idx]
                    # This outer term contributes to C block c_block_idx with coeff outer_coeff.
                    # Now, for this inner product q_idx, it contributes to P_out entries.
                    for p_out_idx, q_coeffs in coeff_map.items():
                        if q_idx in q_coeffs:
                            inner_coeff = q_coeffs[q_idx]
                            total_coeff = outer_coeff * inner_coeff
                            if total_coeff != 0:
                                # p_out_idx is (a,b) local.
                                # c_block_idx is (I,K) block.
                                # I, K are c_block_idx // 2, c_block_idx % 2?
                                # C00: I=0, K=0. C01: I=0, K=1. C10: I=1, K=0. C11: I=1, K=1.
                                I = c_block_idx // 2
                                K = c_block_idx % 2
                                a = p_out_idx // 2
                                b = p_out_idx % 2
                                i = 2*I + a
                                k = 2*K + b
                                w_coeffs[(k,i)] = w_coeffs.get((k,i), 0) + total_coeff
            
            add_product(u_form, v_form, w_coeffs)
            
    # Now add the 4 terms for the 5th column (k=4)
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            u_form = {(i,j): 1}
            v_form = {(j,4): 1}
            w_coeffs = {(4,i): 1}
            add_product(u_form, v_form, w_coeffs)
            
    return {"U": U, "V": V, "W": W}

