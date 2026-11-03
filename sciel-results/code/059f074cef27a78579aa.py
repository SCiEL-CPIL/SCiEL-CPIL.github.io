import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=7
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U.append([u_val])
        V.append([v_val])
        W.append([w_val])
        # Note: The interface expects matrices. 
        # The seed program returns lists of lists.
        # I will build the lists and then convert to the required format.
        # Actually, it's easier to build the full matrices directly or append to lists of rows.
        
    # Let's build the matrices as lists of lists (rows).
    # Rank is not known in advance, so we append to a list of terms.
    terms = []
    
    def add_term_vals(u_row, u_val, v_row, v_val, w_row, w_val):
        terms.append((u_row, u_val, v_row, v_val, w_row, w_val))

    # Strassen 2x2x2 decomposition
    # A: [[a, b], [c, d]], B: [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-c)
    # P5 = (a+b)h
    # P6 = (c-d)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Indices for A: i, j. B: j, k. C: i, k.
    # For a block i0, i1, j0, j1, k0, k1:
    # a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
    # e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # P1
        add_term_vals(i0*m+j0, 1, j0*p+k0, 1, k0*n+i0, 1) # a*e
        add_term_vals(i0*m+j0, 1, j0*p+k1, 1, k1*n+i0, 1) # a*f
        add_term_vals(i1*m+j0, 1, j0*p+k0, 1, k0*n+i1, 1) # c*e
        add_term_vals(i1*m+j0, 1, j0*p+k1, 1, k1*n+i1, 1) # c*f
        add_term_vals(i0*m+j1, 1, j1*p+k0, 1, k0*n+i0, 1) # b*g
        add_term_vals(i0*m+j1, 1, j1*p+k1, 1, k1*n+i0, 1) # b*h
        add_term_vals(i1*m+j1, 1, j1*p+k0, 1, k0*n+i1, 1) # d*g
        add_term_vals(i1*m+j1, 1, j1*p+k1, 1, k1*n+i1, 1) # d*h
        
        # This is wrong. Strassen combines the multiplications.
        # The terms in the tensor decomposition are the P's.
        # Each P is a product of a linear form in A and a linear form in B.
        # P1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # This expands to 4 terms in the schoolbook sense, but in the tensor decomposition,
        # it is ONE term if we can represent the linear forms.
        # U column has 1s at positions corresponding to the linear form in A.
        # V column has 1s at positions corresponding to the linear form in B.
        # W column has 1s at positions corresponding to the linear form in C.
        
        # Let's re-implement add_term to handle linear forms.
        pass

    # Correct approach:
    # A term in the decomposition is defined by:
    # u_vec: vector of length n*m
    # v_vec: vector of length m*p
    # w_vec: vector of length p*n
    # such that sum_t u_vec_t[i*m+j] * v_vec_t[j*p+k] * w_vec_t[k*n+i] = C[i,k]
    
    # For Strassen P1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
    # u_vec has 1 at i0*m+j0 and i1*m+j1.
    # v_vec has 1 at j0*p+k0 and j1*p+k1.
    # w_vec has 1 at k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1?
    # No, the output is C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1].
    # P1 contributes to all 4 outputs with coefficient 1.
    # So w_vec has 1 at k0*n+i0, k1*n+i0, k0*n+i1, k1*n+i1.
    
    # Let's rewrite.
    terms = []
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        # u_indices: list of (row, col) in U
        # v_indices: list of (row, col) in V
        # w_indices: list of (row, col) in W
        terms.append((u_indices, v_indices, w_indices, coeff))

    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # P1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        add_term([(i0*m+j0, 1), (i1*m+j1, 1)], 
                 [(j0*p+k0, 1), (j1*p+k1, 1)], 
                 [(k0*n+i0, 1), (k1*n+i0, 1), (k0*n+i1, 1), (k1*n+i1, 1)], 1)
        
        # P2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        add_term([(i1*m+j0, 1), (i1*m+j1, 1)], 
                 [(j0*p+k0, 1)], 
                 [(k0*n+i1, 1)], 1)
                 
        # P3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        add_term([(i0*m+j0, 1)], 
                 [(j0*p+k1, 1), (j1*p+k1, -1)], 
                 [(k1*n+i0, 1)], 1)
                 
        # P4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        add_term([(i1*m+j1, 1)], 
                 [(j1*p+k0, 1), (j0*p+k0, -1)], 
                 [(k0*n+i1, 1)], 1)
                 
        # P5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        add_term([(i0*m+j0, 1), (i0*m+j1, 1)], 
                 [(j1*p+k1, 1)], 
                 [(k1*n+i0, 1)], 1)
                 
        # P6 = (A[i1,j0] - A[i1,j1]) * (B[j0,k0] + B[j0,k1])
        add_term([(i1*m+j0, 1), (i1*m+j1, -1)], 
                 [(j0*p+k0, 1), (j0*p+k1, 1)], 
                 [(k0*n+i1, 1), (k1*n+i1, 1)], 1)
                 
        # P7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        add_term([(i0*m+j1, 1), (i1*m+j1, -1)], 
                 [(j1*p+k0, 1), (j1*p+k1, 1)], 
                 [(k0*n+i0, 1), (k1*n+i0, 1)], 1)
        
        # Now we need to combine these P's into C.
        # C[i0,k0] = P1 + P4 - P5 + P7
        # C[i0,k1] = P3 + P5
        # C[i1,k0] = P2 + P4
        # C[i1,k1] = P1 - P2 + P3 + P6
        
        # The W matrix entries are the coefficients of the P's in the C entries.
        # My add_term above just added the P's as separate terms with W=1.
        # This is incorrect. The W matrix must encode the linear combination.
        
        # Let's restart the term construction properly.
        pass

    # Correct construction:
    # Each Strassen block generates 7 terms (columns in U, V, W).
    # The W matrix for these columns will have non-zero entries at the 4 output positions.
    
    terms = []
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # P1
        u1 = [(i0*m+j0, 1), (i1*m+j1, 1)]
        v1 = [(j0*p+k0, 1), (j1*p+k1, 1)]
        w1 = [(k0*n+i0, 1), (k1*n+i0, 1), (k0*n+i1, 1), (k1*n+i1, 1)]
        terms.append((u1, v1, w1))
        
        # P2
        u2 = [(i1*m+j0, 1), (i1*m+j1, 1)]
        v2 = [(j0*p+k0, 1)]
        w2 = [(k0*n+i1, 1)]
        terms.append((u2, v2, w2))
        
        # P3
        u3 = [(i0*m+j0, 1)]
        v3 = [(j0*p+k1, 1), (j1*p+k1, -1)]
        w3 = [(k1*n+i0, 1)]
        terms.append((u3, v3, w3))
        
        # P4
        u4 = [(i1*m+j1, 1)]
        v4 = [(j1*p+k0, 1), (j0*p+k0, -1)]
        w4 = [(k0*n+i1, 1)]
        terms.append((u4, v4, w4))
        
        # P5
        u5 = [(i0*m+j0, 1), (i0*m+j1, 1)]
        v5 = [(j1*p+k1, 1)]
        w5 = [(k1*n+i0, 1)]
        terms.append((u5, v5, w5))
        
        # P6
        u6 = [(i1*m+j0, 1), (i1*m+j1, -1)]
        v6 = [(j0*p+k0, 1), (j0*p+k1, 1)]
        w6 = [(k0*n+i1, 1), (k1*n+i1, 1)]
        terms.append((u6, v6, w6))
        
        # P7
        u7 = [(i0*m+j1, 1), (i1*m+j1, -1)]
        v7 = [(j1*p+k0, 1), (j1*p+k1, 1)]
        w7 = [(k0*n+i0, 1), (k1*n+i0, 1)]
        terms.append((u7, v7, w7))
        
        # The W entries are not just 1s. They are the coefficients in the final sum.
        # C[i0,k0] = P1 + P4 - P5 + P7
        # C[i0,k1] = P3 + P5
        # C[i1,k0] = P2 + P4
        # C[i1,k1] = P1 - P2 + P3 + P6
        
        # So for each term t (P1..P7), we need to update the W vector.
        # My current terms list stores w as a list of (index, coeff).
        # I need to combine the w vectors for the 4 outputs.
        
        # Let's store the terms separately and then combine.
        pass

    # This is getting messy. Let's just build the matrices directly.
    # We know the rank is 99.
    # We can initialize U, V, W with zeros of size (15, 99), (35, 99), (21, 99).
    # Then fill in the entries.
    
    rank = 99
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    # Strassen block
    def add_strassen(i0, i1, j0, j1, k0, k1):
        # P1
        u = {i0*m+j0: 1, i1*m+j1: 1}
        v = {j0*p+k0: 1, j1*p+k1: 1}
        w = {k0*n+i0: 1, k1*n+i0: 1, k0*n+i1: 1, k1*n+i1: 1}
        add_term(u, v, w)
        
        # P2
        u = {i1*m+j0: 1, i1*m+j1: 1}
        v = {j0*p+k0: 1}
        w = {k0*n+i1: 1}
        add_term(u, v, w)
        
        # P3
        u = {i0*m+j0: 1}
        v = {j0*p+k1: 1, j1*p+k1: -1}
        w = {k1*n+i0: 1}
        add_term(u, v, w)
        
        # P4
        u = {i1*m+j1: 1}
        v = {j1*p+k0: 1, j0*p+k0: -1}
        w = {k0*n+i1: 1}
        add_term(u, v, w)
        
        # P5
        u = {i0*m+j0: 1, i0*m+j1: 1}
        v = {j1*p+k1: 1}
        w = {k1*n+i0: 1}
        add_term(u, v, w)
        
        # P6
        u = {i1*m+j0: 1, i1*m+j1: -1}
        v = {j0*p+k0: 1, j0*p+k1: 1}
        w = {k0*n+i1: 1, k1*n+i1: 1}
        add_term(u, v, w)
        
        # P7
        u = {i0*m+j1: 1, i1*m+j1: -1}
        v = {j1*p+k0: 1, j1*p+k1: 1}
        w = {k0*n+i0: 1, k1*n+i0: 1}
        add_term(u, v, w)
        
        # Now adjust W for the final combination.
        # The W values added above are just 1s.
        # We need to multiply them by the coefficients.
        # P1 coeff: 1 for C00, 0 for C01, 0 for C10, 1 for C11?
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # So for P1 (col-7):
        W[k0*n+i0][col-7] += 1 # P1
        W[k1*n+i0][col-7] += 0 # P1
        W[k0*n+i1][col-7] += 0 # P1
        W[k1*n+i1][col-7] += 1 # P1
        
        # Wait, the W vector for P1 already has 1s at all 4 positions.
        # I need to set them to the correct coefficients.
        # Let's clear and set.
        
        # This approach is error-prone.
        pass

    # Let's use a simpler, verified method:
    # Just use schoolbook for everything except the 2x2x2 blocks where we use Strassen.
    # But Strassen replaces the 8 schoolbook terms with 7 terms.
    # The 7 terms have specific U, V, W values.
    
    # I will implement the Strassen terms explicitly.
    
    # Clear U, V, W
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # P1
        U[i0*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i0][col] = 0
        W[k0*n+i1][col] = 0
        W[k1*n+i1][col] = 1
        col += 1
        
        # P2
        U[i1*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        col += 1
        
        # P3
        U[i0*m+j0][col] = 1
        V[j0*p+k1][col] = 1
        V[j1*p+k1][col] = -1
        W[k1*n+i0][col] = 1
        col += 1
        
        # P4
        U[i1*m+j1][col] = 1
        V[j1*p+k0][col] = 1
        V[j0*p+k0][col] = -1
        W[k0*n+i1][col] = 1
        col += 1
        
        # P5
        U[i0*m+j0][col] = 1
        U[i0*m+j1][col] = 1
        V[j1*p+k1][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
        
        # P6
        U[i1*m+j0][col] = 1
        U[i1*m+j1][col] = -1
        V[j0*p+k0][col] = 1
        V[j0*p+k1][col] = 1
        W[k0*n+i1][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        
        # P7
        U[i0*m+j1][col] = 1
        U[i1*m+j1][col] = -1
        V[j1*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
        
        # Now add the coefficients for C
        # C00 = P1 + P4 - P5 + P7
        W[k0*n+i0][col-7] += 1 # P1
        W[k0*n+i0][col-4] += 1 # P4
        W[k0*n+i0][col-3] -= 1 # P5
        W[k0*n+i0][col-1] += 1 # P7
        
        # C01 = P3 + P5
        W[k1*n+i0][col-5] += 1 # P3
        W[k1*n+i0][col-3] += 1 # P5
        
        # C10 = P2 + P4
        W[k0*n+i1][col-6] += 1 # P2
        W[k0*n+i1][col-4] += 1 # P4
        
        # C11 = P1 - P2 + P3 + P6
        W[k1*n+i1][col-7] += 1 # P1
        W[k1*n+i1][col-6] -= 1 # P2
        W[k1*n+i1][col-5] += 1 # P3
        W[k1*n+i1][col-2] += 1 # P6

    # Schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1

    # Apply Strassen to i in {0,1}, k in {0,1}, j in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)
    # Apply Strassen to i in {0,1}, k in {0,1}, j in {2,3}
    add_strassen(0, 1, 2, 3, 0, 1)
    # Apply Strassen to i in {0,1}, k in {2,3}, j in {0,1}
    add_strassen(0, 1, 0, 1, 2, 3)
    # Apply Strassen to i in {0,1}, k in {2,3}, j in {2,3}
    add_strassen(0, 1, 2, 3, 2, 3)
    # Apply Strassen to i in {0,1}, k in {4,5}, j in {0,1}
    add_strassen(0, 1, 0, 1, 4, 5)
    # Apply Strassen to i in {0,1}, k in {4,5}, j in {2,3}
    add_strassen(0, 1, 2, 3, 4, 5)
    
    # Schoolbook for remaining
    # i in {0,1}, k in {0,1}, j=4
    for i in [0,1]:
        for k in [0,1]:
            add_schoolbook(i, 4, k)
            
    # i in {0,1}, k in {2,3}, j=4
    for i in [0,1]:
        for k in [2,3]:
            add_schoolbook(i, 4, k)
            
    # i in {0,1}, k in {4,5}, j=4
    for i in [0,1]:
        for k in [4,5]:
            add_schoolbook(i, 4, k)
            
    # i in {0,1}, k=6, all j
    for i in [0,1]:
        for j in range(5):
            add_schoolbook(i, j, 6)
            
    # i=2, all j, k
    for j in range(5):
        for k in range(7):
            add_schoolbook(2, j, k)
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
