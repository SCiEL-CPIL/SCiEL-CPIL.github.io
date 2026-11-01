import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 construction:
    # Split m=3 into 2+1.
    # Part 1: 2x2x8 (4 blocks of 2x2x2, each rank 7) -> 28
    # Part 2: 2x1x8 (16 terms) -> 16
    # Total 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0, 1}
    # Split k into 4 groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    # For each group, use Strassen for 2x2x2
    # Strassen for 2x2x2:
    # Inputs A (2x2), B (2x2). Output C (2x2).
    # 7 multiplications.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # We need to map these to the global U, V, W.
    # The indices for Part 1 are i in {0,1}, j in {0,1}, k in {0..7}.
    # For a specific block q (0..3), k = 2q, 2q+1.
    # A indices: i, j. (i in 0..1, j in 0..1)
    # B indices: j, k. (j in 0..1, k in 2q, 2q+1)
    # C indices: k, i. (k in 2q, 2q+1, i in 0..1)
    
    for q in range(4):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11 -> 1, 1
        # V: B00, B11 -> 1, 1
        # W: C00, C11? No, W contributes to C.
        # The term is sum U V W.
        # We need to add to W[k*n+i]
        
        # Helper to add a term
        def add_term(i_u, j_u, i_v, j_v, k_w, i_w, coeff=1):
            nonlocal col
            U[i_u * m + j_u][col] += coeff
            V[i_v * p + j_v][col] += coeff # Wait, V index is j*p+k. So i_v is j, j_v is k.
            W[k_w * n + i_w][col] += coeff
            col += 1

        # Strassen M1: (A00+A11)(B00+B11)
        # A00: i=0, j=0. A11: i=1, j=1.
        # B00: j=0, k=k0. B11: j=1, k=k1.
        # This product contributes to C.
        # M1 is used in C00 and C11.
        # We need to represent the product (A00+A11)(B00+B11).
        # This is a single multiplication in the Strassen sense.
        # But in the tensor decomposition, we need to express the sum.
        # Actually, the Strassen algorithm computes 7 products of sums.
        # Each product is a scalar.
        # Then C is formed by sums of these scalars.
        # The tensor decomposition rank is the number of scalar multiplications.
        # So we have 7 columns for this block.
        
        # M1 = (A00+A11)(B00+B11)
        # U part: A00 + A11. So U[0,0]=1, U[1,1]=1.
        # V part: B00 + B11. So V[0,k0]=1, V[1,k1]=1.
        # W part: This product contributes to C00 and C11.
        # C00 = M1 + M4 - M5 + M7.
        # C11 = M1 - M2 + M3 + M6.
        # So W for this column should have 1 at C00 and 1 at C11.
        # C00 index: k=k0, i=0. -> k0*n + 0.
        # C11 index: k=k1, i=1. -> k1*n + 1.
        
        # We need to be careful. The tensor decomposition is linear.
        # The term is U V W.
        # If we set U, V as above, the product is (A00+A11)(B00+B11).
        # We want this to add to C00 and C11.
        # So W[k0*n+0] = 1, W[k1*n+1] = 1.
        
        add_term(0, 0, 0, k0, k0, 0) # This is not how add_term works.
        
    # Let's rewrite add_term to accept lists of indices.
    
    def add_strassen_term(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col
        for i, j in u_indices:
            U[i * m + j][col] += coeff
        for j, k in v_indices:
            V[j * p + k][col] += coeff
        for k, i in w_indices:
            W[k * n + i][col] += coeff
        col += 1

    # M1: (A00+A11)(B00+B11)
    # U: (0,0), (1,1)
    # V: (0,k0), (1,k1)
    # W: (k0,0), (k1,1)
    add_strassen_term([(0,0), (1,1)], [(0,k0), (1,k1)], [(k0,0), (k1,1)])
    
    # M2: (A10+A11)B00
    # U: (1,0), (1,1)
    # V: (0,k0)
    # W: (k0,1)  # C10 = M2 + M4
    add_strassen_term([(1,0), (1,1)], [(0,k0)], [(k0,1)])
    
    # M3: A00(B10-B11)
    # U: (0,0)
    # V: (1,k0), (1,k1) with coeff 1, -1
    # W: (k1,0) # C01 = M3 + M5
    add_strassen_term([(0,0)], [(1,k0), (1,k1)], [(k1,0)])
    # Wait, V needs coeff -1 for (1,k1).
    # My add_strassen_term adds coeff to all.
    # I need to handle signs.
    
    # Let's just build the matrices directly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for idx, val in u_list:
            U[idx][col] += val
        for idx, val in v_list:
            V[idx][col] += val
        for idx, val in w_list:
            W[idx][col] += val
        col += 1

    for q in range(4):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # M1: (A00+A11)(B00+B11)
        add_term([(0*m+0, 1), (1*m+1, 1)], 
                 [(0*p+k0, 1), (1*p+k1, 1)], 
                 [(k0*n+0, 1), (k1*n+1, 1)])
        
        # M2: (A10+A11)B00
        add_term([(1*m+0, 1), (1*m+1, 1)], 
                 [(0*p+k0, 1)], 
                 [(k0*n+1, 1)])
                 
        # M3: A00(B10-B11)
        add_term([(0*m+0, 1)], 
                 [(1*p+k0, 1), (1*p+k1, -1)], 
                 [(k1*n+0, 1)])
                 
        # M4: A11(B01-B00)
        add_term([(1*m+1, 1)], 
                 [(0*p+k1, 1), (0*p+k0, -1)], 
                 [(k0*n+1, 1), (k1*n+0, -1)]) # C10 gets M4, C00 gets M4?
        # C00 = M1 + M4 - M5 + M7. C10 = M2 + M4.
        # So M4 contributes to C00 (k0,0) and C10 (k0,1)?
        # Wait, C indices are (k, i).
        # C00 is k0, i=0. C10 is k0, i=1.
        # M4 is A11(B01-B00).
        # B01 is j=0, k=k1. B00 is j=0, k=k0.
        # So V is (0,k1) and (0,k0).
        # This product contributes to C.
        # In Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M4 contributes to C00 (k0,0) and C10 (k0,1).
        # My W list for M4 above: [(k0*n+1, 1), (k1*n+0, -1)] is wrong.
        # It should be [(k0*n+0, 1), (k0*n+1, 1)].
        
        # Let's restart the W assignments carefully.
        
    # Clear and redo
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for q in range(4):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # M1: (A00+A11)(B00+B11)
        # C00, C11
        add_term([(0, 1), (m+1, 1)], 
                 [(k0, 1), (p+k1, 1)], 
                 [(k0*n, 1), (k1*n+1, 1)])
                 
        # M2: (A10+A11)B00
        # C10
        add_term([(m, 1), (m+1, 1)], 
                 [(k0, 1)], 
                 [(k0*n+1, 1)])
                 
        # M3: A00(B10-B11)
        # C01
        add_term([(0, 1)], 
                 [(p+k0, 1), (p+k1, -1)], 
                 [(k1*n, 1)])
                 
        # M4: A11(B01-B00)
        # C00, C10
        add_term([(m+1, 1)], 
                 [(k1, 1), (k0, -1)], 
                 [(k0*n, 1), (k0*n+1, 1)])
                 
        # M5: (A00+A01)B11
        # C00, C01
        add_term([(0, 1), (1, 1)], 
                 [(p+k1, 1)], 
                 [(k0*n, -1), (k1*n, 1)])
                 
        # M6: (A10-A00)(B00+B10)
        # C11
        add_term([(m, 1), (0, -1)], 
                 [(k0, 1), (p+k0, 1)], 
                 [(k1*n+1, 1)])
                 
        # M7: (A01-A11)(B01+B11)
        # C00, C11
        add_term([(1, 1), (m+1, -1)], 
                 [(k1, 1), (p+k1, 1)], 
                 [(k0*n, 1), (k1*n+1, 1)])

    # Part 2: j=2
    # A_i2 B_2k
    for i in range(2):
        for k in range(8):
            add_term([(i*m+2, 1)], 
                     [(2*p+k, 1)], 
                     [(k*n+i, 1)])
                     
    return {"U": U, "V": V, "W": W}
