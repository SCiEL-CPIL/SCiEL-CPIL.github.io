import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    # U shape (n*m, rank) = (9, rank)
    # V shape (m*p, rank) = (12, rank)
    # W shape (p*n, rank) = (12, rank)
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Indices:
    # A[i, j] -> U[i*m + j]
    # B[j, k] -> V[j*p + k]
    # C[i, k] -> W[k*n + i]
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] (Strassen)
    # A blocks: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B blocks: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # Result C[0:2, 0:2]
    
    # M1 = (a+d)(e+h)
    add_term(0, 0, 0, 1, 1, 1) # a*e
    add_term(0, 1, 0, 1, 1, 1) # a*f
    add_term(3, 0, 0, 1, 1, 1) # d*e
    add_term(3, 1, 0, 1, 1, 1) # d*f
    # Wait, Strassen terms are sums. We need to represent the sum of products.
    # The decomposition is a sum of rank-1 tensors.
    # Term: (a+d)(e+h) = ae + af + de + dh.
    # This is 4 rank-1 terms if expanded? No, (a+d) is a linear combination of inputs.
    # In tensor decomposition, we define U, V, W such that T = sum U[:,r] V[:,r] W[:,r].
    # If we use a term like (a+d)(e+h), it corresponds to 4 rank-1 terms in the standard basis?
    # No, Strassen's algorithm reduces the number of *multiplications*.
    # In tensor rank terms, a multiplication of two linear forms is a rank-1 tensor?
    # Yes, $(u \cdot a) (v \cdot b) w$ is a rank-1 tensor if u, v, w are vectors.
    # So each Strassen product is 1 rank unit.
    
    # So for M1 = (A[0,0]+A[1,1]) (B[0,0]+B[1,1]):
    # U vector: 1 at 0, 1 at 3.
    # V vector: 1 at 0, 1 at 1. (B[0,0], B[0,1])? No, e+h is B[0,0]+B[1,1].
    # e=B[0,0] (idx 0), h=B[1,1] (idx 1*4+1=5).
    # So V: 1 at 0, 1 at 5.
    # W vector: contributes to C[0,0], C[0,1], C[1,0], C[1,1].
    # C[0,0] = M1 + ...
    # C[0,1] = ...
    # C[1,0] = ...
    # C[1,1] = ...
    # M1 contributes to C[0,0] with +1, C[1,1] with -1?
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,1] = M1 - M2 + M3 + M6
    # So M1 has +1 at C[0,0] and +1 at C[1,1].
    # W indices: C[0,0] is W[0*3+0]=0. C[1,1] is W[1*3+1]=4.
    
    # Let's implement Strassen properly.
    
    def strassen_2x2_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        # A_rows, A_cols: indices for A
        # B_rows, B_cols: indices for B
        # C_rows, C_cols: indices for C
        # Returns list of (u_vec, v_vec, w_vec)
        terms = []
        
        a_idx = (A_rows[0]*m + A_cols[0], A_rows[0]*m + A_cols[1], A_rows[1]*m + A_cols[0], A_rows[1]*m + A_cols[1])
        e_idx = (B_rows[0]*p + B_cols[0], B_rows[0]*p + B_cols[1], B_rows[1]*p + B_cols[0], B_rows[1]*p + B_cols[1])
        
        # M1 = (a+d)(e+h)
        u1 = {a_idx[0]: 1, a_idx[3]: 1}
        v1 = {e_idx[0]: 1, e_idx[3]: 1}
        w1 = {C_rows[0]*n + C_cols[0]: 1, C_rows[1]*n + C_cols[1]: 1} # C00, C11
        terms.append((u1, v1, w1))
        
        # M2 = (c+d)(e) -- Wait, standard is (c+d)e? No, (c+d)B11?
        # Standard: M2 = (A21+A22)B11 = (c+d)e
        u2 = {a_idx[2]: 1, a_idx[3]: 1}
        v2 = {e_idx[0]: 1}
        w2 = {C_rows[1]*n + C_cols[0]: 1, C_rows[1]*n + C_cols[1]: -1} # C21 (+), C22 (-)
        terms.append((u2, v2, w2))
        
        # M3 = A11(B12-B22) = a(f-h)
        u3 = {a_idx[0]: 1}
        v3 = {e_idx[1]: 1, e_idx[3]: -1}
        w3 = {C_rows[0]*n + C_cols[1]: 1, C_rows[1]*n + C_cols[1]: 1} # C01, C11
        terms.append((u3, v3, w3))
        
        # M4 = A22(B21-B11) = d(g-e)
        u4 = {a_idx[3]: 1}
        v4 = {e_idx[2]: 1, e_idx[0]: -1}
        w4 = {C_rows[0]*n + C_cols[0]: 1, C_rows[1]*n + C_cols[0]: 1} # C00, C10
        terms.append((u4, v4, w4))
        
        # M5 = (A11+A12)B22 = (a+b)h
        u5 = {a_idx[0]: 1, a_idx[1]: 1}
        v5 = {e_idx[3]: 1}
        w5 = {C_rows[0]*n + C_cols[0]: -1, C_rows[0]*n + C_cols[1]: 1} # C00 (-), C01 (+)
        terms.append((u5, v5, w5))
        
        # M6 = (A21-A11)(B11+B12) = (c-b)(e+f)
        u6 = {a_idx[2]: 1, a_idx[0]: -1}
        v6 = {e_idx[0]: 1, e_idx[1]: 1}
        w6 = {C_rows[1]*n + C_cols[1]: 1} # C11
        terms.append((u6, v6, w6))
        
        # M7 = (A12-A22)(B21+B22) = (b-d)(g+h)
        u7 = {a_idx[1]: 1, a_idx[3]: -1}
        v7 = {e_idx[2]: 1, e_idx[3]: 1}
        w7 = {C_rows[0]*n + C_cols[0]: 1} # C00
        terms.append((u7, v7, w7))
        
        return terms

    # Apply Strassen to A[0:2, 0:2] and B[0:2, 0:2]
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1. C rows 0,1 cols 0,1.
    terms1 = strassen_2x2_2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    for u, v, w in terms1:
        add_term_vec(u, v, w)
        
    # Apply Strassen to A[0:2, 0:2] and B[0:2, 2:4]
    # B cols 2,3. C cols 2,3.
    terms2 = strassen_2x2_2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    for u, v, w in terms2:
        add_term_vec(u, v, w)
        
    # Part 3: A[0:2, 2] * B[2, 0:4]
    # A col 2. B row 2.
    # A[0,2] -> U[0*3+2]=2
    # A[1,2] -> U[1*3+2]=5
    # B[2,0] -> V[2*4+0]=8
    # B[2,1] -> V[2*4+1]=9
    # B[2,2] -> V[2*4+2]=10
    # B[2,3] -> V[2*4+3]=11
    # C[0,0] -> W[0], C[0,1] -> W[1], C[0,2] -> W[2], C[0,3] -> W[3]
    # C[1,0] -> W[3], C[1,1] -> W[4], C[1,2] -> W[5], C[1,3] -> W[6]
    
    for k in range(4):
        # A[0,2] * B[2,k] -> C[0,k]
        add_term(2, 8+k, k, 1, 1, 1)
        # A[1,2] * B[2,k] -> C[1,k]
        add_term(5, 8+k, 3+k, 1, 1, 1)
        
    # Part 4: A[2, 0:2] * B[0:2, 0:4]
    # A row 2, cols 0,1. U[6], U[7]
    # B rows 0,1, cols 0..3. V[0..3], V[4..7]
    # C row 2, cols 0..3. W[6..9]
    
    for k in range(4):
        # A[2,0] * B[0,k] -> C[2,k]
        add_term(6, k, 6+k, 1, 1, 1)
        # A[2,1] * B[1,k] -> C[2,k]
        add_term(7, 4+k, 6+k, 1, 1, 1)
        
    # Part 5: A[2,2] * B[2, 0:4]
    # A[2,2] -> U[8]
    # B[2,k] -> V[8+k]
    # C[2,k] -> W[6+k]
    
    for k in range(4):
        add_term(8, 8+k, 6+k, 1, 1, 1)
        
    return {"U": U, "V": V, "W": W}

def add_term_vec(u_dict, v_dict, w_dict):
    global U, V, W, col
    # This needs to be inside construct or use nonlocal
    # I'll refactor to use a class or closure properly
# EVOLVE-BLOCK-END
