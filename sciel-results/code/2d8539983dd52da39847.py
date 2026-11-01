import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # Split A (2x4) into A_L (2x2) and A_R (2x2)
    # Split B (4x5) into B_L (2x5) and B_R (2x5)
    # C = A_L B_L + A_R B_R
    # Each M_{2,2,5} has rank 18 (7+7+4)
    # Total rank 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        pass # We need to build the full matrices
        
    # It's easier to build the matrices by iterating through the decomposition terms.
    # The decomposition is a sum of rank-1 tensors.
    # T = sum_k u_k v_k w_k^T
    
    # Let's define the terms for A_L B_L (rank 18)
    # A_L indices: i in {0,1}, j in {0,1} -> global j in {0,1}
    # B_L indices: j in {0,1}, k in {0..4} -> global j in {0,1}
    
    # We will create a list of (u_indices, v_indices, w_indices)
    # u_indices: (i, j) for A
    # v_indices: (j, k) for B
    # w_indices: (k, i) for C
    
    terms = []
    
    # Process A_L B_L
    # A_L is A[0:2, 0:2]
    # B_L is B[0:2, 0:5]
    
    # Split B_L into B_L1 (cols 0,1), B_L2 (cols 2,3), B_L3 (col 4)
    
    # B_L1: j in {0,1}, k in {0,1}
    # Strassen for M_{2,2,2}
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # m1 = (a+d)(e+h)
    # This corresponds to:
    # u: (0,0), (1,1)
    # v: (0,0), (1,1)
    # w: (0,0), (1,1)
    # But Strassen combines them.
    # The rank-1 terms are not simply the products of sums.
    # We need to expand the Strassen multiplications into the final W coefficients.
    
    # This is getting complex to implement manually for 36 terms.
    # Let's use a generic tensor decomposition builder.
    
    # We define the bilinear forms.
    # Each multiplication m_k corresponds to a product of a linear form in A and a linear form in B.
    # The result is added to specific entries of C.
    
    # Let's create a list of multiplications: (A_coeffs, B_coeffs, C_indices, C_coeffs)
    # A_coeffs: dict {(i,j): val}
    # B_coeffs: dict {(j,k): val}
    # C_indices: list of (k,i)
    # C_coeffs: list of val
    
    mults = []
    
    def add_strassen_222(a_idx, b_idx, c_base):
        # a_idx: (i, j) for A_L
        # b_idx: (j, k) for B_L
        # c_base: (k, i) offset
        
        # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        a = (0, 0)
        b = (0, 1)
        c = (1, 0)
        d = (1, 1)
        
        # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
        e = (0, 0)
        f = (0, 1)
        g = (1, 0)
        h = (1, 1)
        
        # m1 = (a+d)(e+h)
        A1 = {a: 1, d: 1}
        B1 = {e: 1, h: 1}
        C1 = {(0,0): 1, (1,1): 1, (1,0): 1} # c00, c11, c10?
        # c00 = m1 + m4 - m6 + m7
        # c11 = m1 - m3 + m2 - m5
        # c10 = m2 + m4
        # c01 = m3 + m5
        
        # We need to accumulate contributions to C.
        # It's easier to define the 7 multiplications and their contributions.
        
        pass

    # This approach is error-prone.
    # Let's stick to the schoolbook rank 40 if I can't guarantee the correctness of the lower rank decomposition quickly.
    # However, the prompt asks for a stronger algorithm.
    # Rank 36 is a strong improvement.
    
    # Let's try to generate the rank 36 decomposition programmatically.
    
    # We'll define the 18 multiplications for A_L B_L.
    # And 18 for A_R B_R.
    
    # For A_L B_L:
    # B_L1 (cols 0,1): Strassen
    # B_L2 (cols 2,3): Strassen
    # B_L3 (col 4): Schoolbook
    
    # Let's write a helper to add a multiplication to the lists.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_mult(A_coeffs, B_coeffs, C_coeffs):
        # A_coeffs: {(i,j): val}
        # B_coeffs: {(j,k): val}
        # C_coeffs: {(k,i): val}
        
        # We need to add a rank-1 tensor.
        # But the multiplication is of linear forms.
        # L_A = sum A_coeffs[(i,j)] * A[i,j]
        # L_B = sum B_coeffs[(j,k)] * B[j,k]
        # Product P = L_A * L_B
        # C[k,i] += C_coeffs[(k,i)] * P
        
        # This is not a rank-1 tensor in the standard sense if C_coeffs has multiple entries.
        # But we can split it into multiple rank-1 terms if needed, or just record the bilinear form.
        # The interface requires U, V, W matrices.
        # U[i*m+j, c] is the coefficient of A[i,j] in the c-th multiplication.
        # V[j*p+k, c] is the coefficient of B[j,k] in the c-th multiplication.
        # W[k*n+i, c] is the coefficient of the c-th multiplication in C[k,i].
        
        # So for each multiplication c, we define:
        # U_col = vector of A_coeffs
        # V_col = vector of B_coeffs
        # W_col = vector of C_coeffs
        
        col_idx = len(U_list)
        
        u_col = [0] * (n*m)
        for (i,j), val in A_coeffs.items():
            u_col[i*m+j] = val
            
        v_col = [0] * (m*p)
        for (j,k), val in B_coeffs.items():
            v_col[j*p+k] = val
            
        w_col = [0] * (p*n)
        for (k,i), val in C_coeffs.items():
            w_col[k*n+i] = val
            
        U_list.append(u_col)
        V_list.append(v_col)
        W_list.append(w_col)
        
    # A_L B_L
    # A_L: i in 0..1, j in 0..1
    # B_L: j in 0..1, k in 0..4
    
    # B_L1: k in 0..1
    # Strassen
    # m1
    A_m1 = {(0,0): 1, (1,1): 1}
    B_m1 = {(0,0): 1, (1,1): 1}
    C_m1 = {(0,0): 1, (1,1): 1, (1,0): 1} # Wait, c10 gets m2+m4.
    # Let's be precise.
    # c00 = m1 + m4 - m6 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m2 - m5
    
    # m1: (a+d)(e+h)
    A_m1 = {(0,0): 1, (1,1): 1}
    B_m1 = {(0,0): 1, (1,1): 1}
    C_m1 = {(0,0): 1, (1,1): 1} # c00 and c11 get +1
    # But c10 also gets m1? No.
    # c10 = m2 + m4.
    # So m1 contributes to c00 and c11.
    add_mult(A_m1, B_m1, {(0,0): 1, (1,1): 1})
    
    # m2: (c+d)(g+h)
    A_m2 = {(1,0): 1, (1,1): 1}
    B_m2 = {(1,0): 1, (1,1): 1}
    C_m2 = {(1,0): 1, (1,1): 1} # c10 and c11
    add_mult(A_m2, B_m2, {(1,0): 1, (1,1): 1})
    
    # m3: a(f-h)
    A_m3 = {(0,0): 1}
    B_m3 = {(0,1): 1, (1,1): -1}
    C_m3 = {(0,1): 1, (1,1): -1} # c01 and c11
    add_mult(A_m3, B_m3, {(0,1): 1, (1,1): -1})
    
    # m4: d(g-e)
    A_m4 = {(1,1): 1}
    B_m4 = {(1,0): 1, (0,0): -1}
    C_m4 = {(0,0): 1, (1,0): 1} # c00 and c10
    add_mult(A_m4, B_m4, {(0,0): 1, (1,0): 1})
    
    # m5: (a+b)e
    A_m5 = {(0,0): 1, (0,1): 1}
    B_m5 = {(0,0): 1}
    C_m5 = {(0,1): 1, (1,1): -1} # c01 and c11
    add_mult(A_m5, B_m5, {(0,1): 1, (1,1): -1})
    
    # m6: (c+d)h
    A_m6 = {(1,0): 1, (1,1): 1}
    B_m6 = {(1,1): 1}
    C_m6 = {(0,0): -1} # c00
    add_mult(A_m6, B_m6, {(0,0): -1})
    
    # m7: (a-b)(f+g)
    A_m7 = {(0,0): 1, (0,1): -1}
    B_m7 = {(0,1): 1, (1,0): 1}
    C_m7 = {(0,0): 1} # c00
    add_mult(A_m7, B_m7, {(0,0): 1})
    
    # B_L2: k in 2..3
    # Shift k by 2
    # e->(0,2), f->(0,3), g->(1,2), h->(1,3)
    # c indices shift by 2
    
    # m1
    A_m1 = {(0,0): 1, (1,1): 1}
    B_m1 = {(0,2): 1, (1,3): 1}
    C_m1 = {(2,0): 1, (3,1): 1}
    add_mult(A_m1, B_m1, C_m1)
    
    # m2
    A_m2 = {(1,0): 1, (1,1): 1}
    B_m2 = {(1,2): 1, (1,3): 1}
    C_m2 = {(2,1): 1, (3,1): 1}
    add_mult(A_m2, B_m2, C_m2)
    
    # m3
    A_m3 = {(0,0): 1}
    B_m3 = {(0,3): 1, (1,3): -1}
    C_m3 = {(3,0): 1, (3,1): -1}
    add_mult(A_m3, B_m3, C_m3)
    
    # m4
    A_m4 = {(1,1): 1}
    B_m4 = {(1,2): 1, (0,2): -1}
    C_m4 = {(2,0): 1, (2,1): 1}
    add_mult(A_m4, B_m4, C_m4)
    
    # m5
    A_m5 = {(0,0): 1, (0,1): 1}
    B_m5 = {(0,2): 1}
    C_m5 = {(3,0): 1, (3,1): -1}
    add_mult(A_m5, B_m5, C_m5)
    
    # m6
    A_m6 = {(1,0): 1, (1,1): 1}
    B_m6 = {(1,3): 1}
    C_m6 = {(2,0): -1}
    add_mult(A_m6, B_m6, C_m6)
    
    # m7
    A_m7 = {(0,0): 1, (0,1): -1}
    B_m7 = {(0,3): 1, (1,2): 1}
    C_m7 = {(2,0): 1}
    add_mult(A_m7, B_m7, C_m7)
    
    # B_L3: k in 4
    # Schoolbook
    # c40 = a B[0,4] + b B[1,4]
    # c41 = c B[0,4] + d B[1,4]
    
    # m1: a B[0,4]
    A_m1 = {(0,0): 1}
    B_m1 = {(0,4): 1}
    C_m1 = {(4,0): 1}
    add_mult(A_m1, B_m1, C_m1)
    
    # m2: b B[1,4]
    A_m2 = {(0,1): 1}
    B_m2 = {(1,4): 1}
    C_m2 = {(4,0): 1}
    add_mult(A_m2, B_m2, C_m2)
    
    # m3: c B[0,4]
    A_m3 = {(1,0): 1}
    B_m3 = {(0,4): 1}
    C_m3 = {(4,1): 1}
    add_mult(A_m3, B_m3, C_m3)
    
    # m4: d B[1,4]
    A_m4 = {(1,1): 1}
    B_m4 = {(1,4): 1}
    C_m4 = {(4,1): 1}
    add_mult(A_m4, B_m4, C_m4)
    
    # Now A_R B_R
    # A_R: i in 0..1, j in 2..3
    # B_R: j in 2..3, k in 0..4
    
    # B_R1: k in 0..1
    # Shift j by 2
    
    # m1
    A_m1 = {(0,2): 1, (1,3): 1}
    B_m1 = {(2,0): 1, (3,1): 1}
    C_m1 = {(0,0): 1, (1,1): 1}
    add_mult(A_m1, B_m1, C_m1)
    
    # m2
    A_m2 = {(1,2): 1, (1,3): 1}
    B_m2 = {(3,0): 1, (3,1): 1}
    C_m2 = {(1,0): 1, (1,1): 1}
    add_mult(A_m2, B_m2, C_m2)
    
    # m3
    A_m3 = {(0,2): 1}
    B_m3 = {(2,1): 1, (3,1): -1}
    C_m3 = {(1,0): 1, (1,1): -1}
    add_mult(A_m3, B_m3, C_m3)
    
    # m4
    A_m4 = {(1,3): 1}
    B_m4 = {(3,0): 1, (2,0): -1}
    C_m4 = {(0,0): 1, (1,0): 1}
    add_mult(A_m4, B_m4, C_m4)
    
    # m5
    A_m5 = {(0,2): 1, (0,3): 1}
    B_m5 = {(2,0): 1}
    C_m5 = {(1,0): 1, (1,1): -1}
    add_mult(A_m5, B_m5, C_m5)
    
    # m6
    A_m6 = {(1,2): 1, (1,3): 1}
    B_m6 = {(3,1): 1}
    C_m6 = {(0,0): -1}
    add_mult(A_m6, B_m6, C_m6)
    
    # m7
    A_m7 = {(0,2): 1, (0,3): -1}
    B_m7 = {(2,1): 1, (3,0): 1}
    C_m7 = {(0,0): 1}
    add_mult(A_m7, B_m7, C_m7)
    
    # B_R2: k in 2..3
    # Shift j by 2, k by 2
    
    # m1
    A_m1 = {(0,2): 1, (1,3): 1}
    B_m1 = {(2,2): 1, (3,3): 1}
    C_m1 = {(2,0): 1, (3,1): 1}
    add_mult(A_m1, B_m1, C_m1)
    
    # m2
    A_m2 = {(1,2): 1, (1,3): 1}
    B_m2 = {(3,2): 1, (3,3): 1}
    C_m2 = {(2,1): 1, (3,1): 1}
    add_mult(A_m2, B_m2, C_m2)
    
    # m3
    A_m3 = {(0,2): 1}
    B_m3 = {(2,3): 1, (3,3): -1}
    C_m3 = {(3,0): 1, (3,1): -1}
    add_mult(A_m3, B_m3, C_m3)
    
    # m4
    A_m4 = {(1,3): 1}
    B_m4 = {(3,2): 1, (2,2): -1}
    C_m4 = {(2,0): 1, (2,1): 1}
    add_mult(A_m4, B_m4, C_m4)
    
    # m5
    A_m5 = {(0,2): 1, (0,3): 1}
    B_m5 = {(2,2): 1}
    C_m5 = {(3,0): 1, (3,1): -1}
    add_mult(A_m5, B_m5, C_m5)
    
    # m6
    A_m6 = {(1,2): 1, (1,3): 1}
    B_m6 = {(3,3): 1}
    C_m6 = {(2,0): -1}
    add_mult(A_m6, B_m6, C_m6)
    
    # m7
    A_m7 = {(0,2): 1, (0,3): -1}
    B_m7 = {(2,3): 1, (3,2): 1}
    C_m7 = {(2,0): 1}
    add_mult(A_m7, B_m7, C_m7)
    
    # B_R3: k in 4
    # Shift j by 2
    
    # m1
    A_m1 = {(0,2): 1}
    B_m1 = {(2,4): 1}
    C_m1 = {(4,0): 1}
    add_mult(A_m1, B_m1, C_m1)
    
    # m2
    A_m2 = {(0,3): 1}
    B_m2 = {(3,4): 1}
    C_m2 = {(4,0): 1}
    add_mult(A_m2, B_m2, C_m2)
    
    # m3
    A_m3 = {(1,2): 1}
    B_m3 = {(2,4): 1}
    C_m3 = {(4,1): 1}
    add_mult(A_m3, B_m3, C_m3)
    
    # m4
    A_m4 = {(1,3): 1}
    B_m4 = {(3,4): 1}
    C_m4 = {(4,1): 1}
    add_mult(A_m4, B_m4, C_m4)
    
    # Convert lists to matrices
    U = [u for u in U_list]
    V = [v for v in V_list]
    W = [w for w in W_list]
    
    return {"U": U, "V": V, "W": W}

