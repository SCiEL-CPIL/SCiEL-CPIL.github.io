import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # Strassen 2x2 multiplication
    # Inputs: 
    # A rows i1, i2; cols j1, j2
    # B rows j1, j2; cols k1, k2
    # Output C rows i1, i2; cols k1, k2
    def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # P1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # This is not a single rank 1 term. It's a product of sums.
        # We need to expand this into rank 1 terms?
        # No, Strassen's algorithm computes 7 products of linear combinations.
        # Each product $X \cdot Y$ where $X$ is a linear combo of A's and $Y$ is a linear combo of B's
        # corresponds to a rank 1 tensor term?
        # $X = \sum u_{i,j} A_{i,j}$. $Y = \sum v_{j',k'} B_{j',k'}$.
        # Product $X Y = \sum_{i,j,j',k'} u_{i,j} v_{j',k'} A_{i,j} B_{j',k'}$.
        # This is a sum of rank 1 terms?
        # No, in the tensor decomposition, each $\alpha$ is a rank 1 term $u_\alpha \otimes v_\alpha \otimes w_\alpha$.
        # $C_{i,k} = \sum_\alpha W_{k,i,\alpha} (\sum_j U_{i,j,\alpha} A_{i,j}) (\sum_{j'} V_{j',k,\alpha} B_{j',k})$.
        # So if we have a term $X Y$, we set:
        # $U_{i,j,\alpha} = u_{i,j}$
        # $V_{j',k,\alpha} = v_{j',k}$
        # $W_{k,i,\alpha} = w_{k,i}$ (contribution to C)
        
        # For P1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # u: A[i1,j1] coeff 1, A[i2,j2] coeff 1.
        # v: B[j1,k1] coeff 1, B[j2,k2] coeff 1.
        # w: C[i1,k1] += 1, C[i1,k2] += 1, C[i2,k1] += 1, C[i2,k2] += 1?
        # No, $X Y$ contributes to all $C_{i,k}$?
        # $C_{i,k} = \sum_\alpha W_{k,i,\alpha} L_\alpha(A) M_\alpha(B)$.
        # If $L_\alpha(A) = A[i1,j1] + A[i2,j2]$ and $M_\alpha(B) = B[j1,k1] + B[j2,k2]$.
        # Then $L M = (A[i1,j1] + A[i2,j2])(B[j1,k1] + B[j2,k2])$.
        # This product is added to $C_{i,k}$ multiplied by $W_{k,i,\alpha}$.
        # We want $P_1$ to contribute to $C_{00}, C_{01}, C_{10}, C_{11}$ of the 2x2 block.
        # Specifically:
        # $C_{i1,k1} += P_1$
        # $C_{i1,k2} += P_1$
        # $C_{i2,k1} += P_1$
        # $C_{i2,k2} += P_1$
        # So $W_{k1,i1,\alpha} = 1, W_{k2,i1,\alpha} = 1, W_{k1,i2,\alpha} = 1, W_{k2,i2,\alpha} = 1$.
        
        # Let's verify with Strassen formulas.
        # $C_{00} = P_1 + P_4 - P_5 + P_7$.
        # So $P_1$ adds 1 to $C_{00}$.
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for (i, j), u in u_coeffs.items():
                U[i * m + j][col] += u
            for (j, k), v in v_coeffs.items():
                V[j * p + k][col] += v
            for (k, i), w in w_coeffs.items():
                W[k * n + i][col] += w
            col += 1

        # P1
        u1 = {(i1, j1): 1, (i2, j2): 1}
        v1 = {(j1, k1): 1, (j2, k2): 1}
        w1 = {(k1, i1): 1, (k2, i1): 1, (k1, i2): 1, (k2, i2): 1}
        add_strassen_term(u1, v1, w1)

        # P2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        u2 = {(i2, j1): 1, (i2, j2): 1}
        v2 = {(j1, k1): 1}
        w2 = {(k1, i2): 1} # C10 += P2
        add_strassen_term(u2, v2, w2)

        # P3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        u3 = {(i1, j1): 1}
        v3 = {(j1, k2): 1, (j2, k2): -1}
        w3 = {(k2, i1): 1} # C01 += P3
        add_strassen_term(u3, v3, w3)

        # P4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        u4 = {(i2, j2): 1}
        v4 = {(j2, k1): 1, (j1, k1): -1}
        w4 = {(k1, i1): 1, (k1, i2): 1} # C00 += P4, C10 += P4
        add_strassen_term(u4, v4, w4)

        # P5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        u5 = {(i1, j1): 1, (i1, j2): 1}
        v5 = {(j2, k2): 1}
        w5 = {(k2, i1): -1, (k2, i2): -1} # C00 -= P5, C11 -= P5?
        # C00 = P1 + P4 - P5 + P7. So w(k2,i1) = -1.
        # C11 = P1 - P2 + P3 + P6. No P5.
        # Wait, C11 formula: P1 - P2 + P3 + P6.
        # C01 = P3 + P5. So w(k2,i1) += 1 for P5?
        # Let's re-verify Strassen.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # So P5 contributes:
        # C00: -1
        # C01: +1
        # C10: 0
        # C11: 0
        w5 = {(k2, i1): -1, (k2, i1): 1} # Conflict?
        # No, P5 is one term.
        # $P_5 = (A_{00} + A_{01}) B_{11}$.
        # $C_{00} = P_1 + P_4 - P_5 + P_7$.
        # $C_{01} = P_3 + P_5$.
        # So $W_{k2, i1}$ gets -1 for C00 and +1 for C01?
        # $W_{k,i}$ is a single value.
        # $C_{i1, k2} = P_3 + P_5$. Here $i=i1, k=k2$. $W_{k2, i1} = 1$.
        # $C_{i1, k1} = P_1 + P_4 - P_5 + P_7$. Here $i=i1, k=k1$. $W_{k1, i1} = -1$.
        # So $W_{k2, i1} = 1$ and $W_{k1, i1} = -1$.
        # My w5 dict was keyed by (k, i).
        # w5 = {(k2, i1): 1, (k1, i1): -1}
        w5 = {(k2, i1): 1, (k1, i1): -1}
        # Wait, I already added P1, P2, P3, P4.
        # I need to reset or accumulate.
        # The function `add_strassen_term` increments col.
        # So each P is a separate rank-1 term.
        # So for P5:
        u5 = {(i1, j1): 1, (i1, j2): 1}
        v5 = {(j2, k2): 1}
        w5 = {(k2, i1): 1, (k1, i1): -1}
        add_strassen_term(u5, v5, w5)

        # P6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        u6 = {(i2, j1): 1, (i1, j1): -1}
        v6 = {(j1, k1): 1, (j1, k2): 1}
        w6 = {(k2, i2): 1} # C11 += P6
        add_strassen_term(u6, v6, w6)

        # P7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        u7 = {(i1, j2): 1, (i2, j2): -1}
        v7 = {(j2, k1): 1, (j2, k2): 1}
        w7 = {(k1, i1): 1} # C00 += P7
        add_strassen_term(u7, v7, w7)

    # 1. A1 (rows 0,1) x B1 (cols 0-3)
    # B1 is 4x4.
    # Blocks:
    # A11 (0,1 x 0,1), A12 (0,1 x 2,3)
    # B11 (0,1 x 0,1), B12 (0,1 x 2,3)
    # B21 (2,3 x 0,1), B22 (2,3 x 2,3)
    
    # C11 (0,1 x 0,1) = A11 B11 + A12 B21
    # C12 (0,1 x 2,3) = A11 B12 + A12 B22
    
    # A11 B11: i=0,1 j=0,1 k=0,1
    add_strassen_2x2(0, 1, 0, 1, 0, 1)
    # A12 B21: i=0,1 j=2,3 k=0,1
    add_strassen_2x2(0, 1, 2, 3, 0, 1)
    # A11 B12: i=0,1 j=0,1 k=2,3
    add_strassen_2x2(0, 1, 0, 1, 2, 3)
    # A12 B22: i=0,1 j=2,3 k=2,3
    add_strassen_2x2(0, 1, 2, 3, 2, 3)
    
    # 2. A1 (rows 0,1) x B2 (col 4)
    # B2 is 4x1.
    # Schoolbook: 2x4x1 = 8 terms.
    for i in range(2):
        for j in range(4):
            add_term(i, j, 4, 1, 1, 1)
            
    # 3. A2 (row 2) x B (4x5)
    # Schoolbook: 1x4x5 = 20 terms.
    for j in range(4):
        for k in range(5):
            add_term(2, j, k, 1, 1, 1)
            
    return {"U": U, "V": V, "W": W}
