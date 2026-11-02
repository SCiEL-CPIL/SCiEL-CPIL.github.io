import numpy as np

def construct():
    """
    Construct a rank-49 exact tensor decomposition for 4x4 x 4x4 matrix
    multiplication using the Strassen algorithm applied recursively.
    
    The decomposition is:
      C = sum_r U[:,r] * V[:,r] * W[:,r]
    
    where U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    with integer entries.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get the 2x2 block A_ij as a 2x2 matrix of indices
    # A is 4x4. A_ij is 2x2 block at row i, col j.
    # Indices in A: 0..15. A[r, c] is at index r*4+c.
    
    # Define the 7 Strassen forms for 2x2 matrices
    # Returns list of 7 tuples (U_vec, V_vec) where U_vec, V_vec are dicts or lists
    # But here we work with the 4x4 entries directly.
    
    # Let's represent linear forms as vectors of length 16.
    # U_form[i] is the coefficient of A[i] in the linear form.
    
    def add_forms(f1, f2):
        return [a + b for a, b in zip(f1, f2)]
    
    def sub_forms(f1, f2):
        return [a - b for a, b in zip(f1, f2)]
    
    def scale_form(f, s):
        return [a * s for a in f]
    
    # Identity forms for A entries
    # A[i, j] is at index i*4+j
    # We need to define the 16 basis vectors for A and B.
    
    # For each of the 7 outer Strassen steps (k=1..7), we define X_k and Y_k
    # X_k and Y_k are 2x2 matrices of linear forms (vectors of length 16).
    
    # Block A_00 corresponds to A[0:2, 0:2] -> indices 0, 1, 4, 5
    # Block A_01 corresponds to A[0:2, 2:4] -> indices 2, 3, 6, 7
    # Block A_10 corresponds to A[2:4, 0:2] -> indices 8, 9, 12, 13
    # Block A_11 corresponds to A[2:4, 2:4] -> indices 10, 11, 14, 15
    
    # Let's create a helper to get the 16-dim vector for a 2x2 block of A
    # block(i, j) returns a 2x2 matrix of 16-dim vectors
    
    def get_block_A(i, j):
        # i, j in {0, 1}
        # Returns 2x2 list of 16-dim vectors
        block = [[None for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                idx = (2*i + r) * 4 + (2*j + c)
                vec = [0] * 16
                vec[idx] = 1
                block[r][c] = vec
        return block
    
    def get_block_B(i, j):
        block = [[None for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                idx = (2*i + r) * 4 + (2*j + c)
                vec = [0] * 16
                vec[idx] = 1
                block[r][c] = vec
        return block
    
    def add_blocks(B1, B2):
        return [[add_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]
    
    def sub_blocks(B1, B2):
        return [[sub_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]
    
    def scale_blocks(B, s):
        return [[scale_forms(B[r][c], s) for c in range(2)] for r in range(2)]
    
    # Pre-compute blocks
    A00 = get_block_A(0, 0)
    A01 = get_block_A(0, 1)
    A10 = get_block_A(1, 0)
    A11 = get_block_A(1, 1)
    
    B00 = get_block_B(0, 0)
    B01 = get_block_B(0, 1)
    B10 = get_block_B(1, 0)
    B11 = get_block_B(1, 1)
    
    # Define the 7 X_k, Y_k pairs
    # k=1: X1 = A00+A11, Y1 = B00+B11
    # k=2: X2 = A10+A11, Y2 = B00
    # k=3: X3 = A00,       Y3 = B10-B11
    # k=4: X4 = A11,       Y4 = B01-B00
    # k=5: X5 = A00+A01,   Y5 = B11
    # k=6: X6 = A10-A00,   Y6 = B00+B10
    # k=7: X7 = A01-A11,   Y7 = B01+B11
    
    Xs = [
        add_blocks(A00, A11),
        add_blocks(A10, A11),
        A00,
        A11,
        add_blocks(A00, A01),
        sub_blocks(A10, A00),
        sub_blocks(A01, A11)
    ]
    
    Ys = [
        add_blocks(B00, B11),
        B00,
        sub_blocks(B10, B11),
        sub_blocks(B01, B00),
        B11,
        add_blocks(B00, B10),
        add_blocks(B01, B11)
    ]
    
    # For each k, apply Strassen to X_k, Y_k to get 7 bilinear forms
    # Strassen for 2x2:
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y10-y11)
    # m4 = x11(y01-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y10)
    # m7 = (x01-x11)(y01+y11)
    
    forms = [] # List of (U_vec, V_vec)
    
    for k in range(7):
        X = Xs[k]
        Y = Ys[k]
        
        # Compute linear forms for U and V
        u1 = add_forms(X[0][0], X[1][1])
        v1 = add_forms(Y[0][0], Y[1][1])
        forms.append((u1, v1))
        
        u2 = add_forms(X[1][0], X[1][1])
        v2 = Y[0][0]
        forms.append((u2, v2))
        
        u3 = X[0][0]
        v3 = sub_forms(Y[1][0], Y[1][1])
        forms.append((u3, v3))
        
        u4 = X[1][1]
        v4 = sub_forms(Y[0][1], Y[0][0])
        forms.append((u4, v4))
        
        u5 = add_forms(X[0][0], X[0][1])
        v5 = Y[1][1]
        forms.append((u5, v5))
        
        u6 = sub_forms(X[1][0], X[0][0])
        v6 = add_forms(Y[0][0], Y[1][0])
        forms.append((u6, v6))
        
        u7 = sub_forms(X[0][1], X[1][1])
        v7 = add_forms(Y[0][1], Y[1][1])
        forms.append((u7, v7))
        
    # Now we have 49 forms.
    # We need to fill U, V, W.
    
    # W coefficients:
    # C_ij = sum_{k,l} w_{ij}^{(k,l)} m_{k,l}
    # C is 4x4.
    # C_block(I, J) = sum_{k=1}^7 beta_{I,J}^{(k)} P_k
    # P_k[i', j'] = sum_{l=1}^7 gamma_{i', j'}^{(l)} m_{k,l}
    
    # Beta coefficients for blocks
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    betas = {
        (0,0): [1, 0, 0, 1, -1, 0, 1],
        (0,1): [0, 0, 1, 0, 1, 0, 0],
        (1,0): [0, 1, 0, 1, 0, 0, 0],
        (1,1): [1, -1, 1, 0, 0, 1, 0]
    }
    
    # Gamma coefficients for P_k entries
    # P00 = m1+m4-m5+m7
    # P01 = m3+m5
    # P10 = m2+m4
    # P11 = m1-m2+m3+m6
    
    gammas = {
        (0,0): [1, 0, 0, 1, -1, 0, 1],
        (0,1): [0, 0, 1, 0, 1, 0, 0],
        (1,0): [0, 1, 0, 1, 0, 0, 0],
        (1,1): [1, -1, 1, 0, 0, 1, 0]
    }
    
    # Fill U, V
    for idx, (u_vec, v_vec) in enumerate(forms):
        for i in range(16):
            U[i][idx] = u_vec[i]
            V[i][idx] = v_vec[i]
            
    # Fill W
    # W is indexed by k*n+i (trace convention)
    # C[i, k] is at W[k*4 + i]
    # Wait, the task says W[k*n+i, r].
    # C[i, k] = sum_j A[i, j] B[j, k].
    # In the tensor decomposition, the output index is (i, k).
    # The W matrix row index is k*n + i.
    
    for I in range(2):
        for J in range(2):
            for i_prime in range(2):
                for j_prime in range(2):
                    # Output index in 4x4: i = 2*I + i_prime, k = 2*J + j_prime
                    i = 2*I + i_prime
                    k = 2*J + j_prime
                    w_idx = k * 4 + i
                    
                    # Sum over k (outer strassen) and l (inner strassen)
                    for k_outer in range(7):
                        beta = betas[(I, J)][k_outer]
                        if beta == 0:
                            continue
                        for l_inner in range(7):
                            gamma = gammas[(i_prime, j_prime)][l_inner]
                            if gamma == 0:
                                continue
                            form_idx = k_outer * 7 + l_inner
                            W[w_idx][form_idx] += beta * gamma
    
    return {"U": U, "V": V, "W": W}
