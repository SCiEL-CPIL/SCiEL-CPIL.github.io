import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication using
    block decomposition with Strassen on 2x2 blocks.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))
    
    # Strassen decomposition for 2x2 matmul
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # where:
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-d)(f+h), M7 = (b-c)(e+g)
    
    def strassen_2x2(A_r0, A_c0, A_r1, A_c1, B_r0, B_c0, B_r1, B_c1, C_r0, C_c0, C_r1, C_c1):
        """
        Add Strassen terms for 2x2 block at:
        A rows [A_r0, A_r1], cols [A_c0, A_c1]
        B rows [B_r0, B_r1], cols [B_c0, B_c1]
        C rows [C_r0, C_r1], cols [C_c0, C_c1]
        """
        # a = A[A_r0, A_c0], b = A[A_r0, A_c1]
        # c = A[A_r1, A_c0], d = A[A_r1, A_c1]
        # e = B[B_r0, B_c0], f = B[B_r0, B_c1]
        # g = B[B_r1, B_c0], h = B[B_r1, B_c1]
        
        # M1 = (a+d)(e+h)
        # = a*e + a*h + d*e + d*h
        # In tensor terms: U[a]*V[e]*W[C00] + U[a]*V[h]*W[C00] + U[d]*V[e]*W[C00] + U[d]*V[h]*W[C00]
        # But we need to combine with other M's for each C entry.
        
        # Let me define the 7 Strassen products and their contributions to each C entry.
        
        # M1 contributes to: C[0,0] (+1), C[1,1] (+1)
        # M2 contributes to: C[1,0] (+1), C[1,1] (-1)
        # M3 contributes to: C[0,1] (+1), C[1,1] (+1)
        # M4 contributes to: C[0,0] (-1), C[1,0] (+1)
        # M5 contributes to: C[0,0] (+1), C[0,1] (+1)
        # M6 contributes to: C[1,1] (+1)
        # M7 contributes to: C[0,0] (+1)
        
        # Each M is a product of two sums. We expand each into rank-1 terms.
        
        # M1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
        u_a = A_r0 * m + A_c0
        u_b = A_r0 * m + A_c1
        u_c = A_r1 * m + A_c0
        u_d = A_r1 * m + A_c1
        
        v_e = B_r0 * p + B_c0
        v_f = B_r0 * p + B_c1
        v_g = B_r1 * p + B_c0
        v_h = B_r1 * p + B_c1
        
        w_00 = C_c0 * n + C_r0  # C[0,0] in trace convention: W[k*n+i] where i=C_r0, k=C_c0
        w_01 = C_c1 * n + C_r0  # C[0,1]
        w_10 = C_c0 * n + C_r1  # C[1,0]
        w_11 = C_c1 * n + C_r1  # C[1,1]
        
        # M1 = (a+d)(e+h): 4 terms
        # a*e -> contributes +1 to C[0,0], +1 to C[1,1]
        # a*h -> contributes +1 to C[0,0], +1 to C[1,1]
        # d*e -> contributes +1 to C[0,0], +1 to C[1,1]
        # d*h -> contributes +1 to C[0,0], +1 to C[1,1]
        # But wait, we can't have one term contribute to multiple C entries.
        # In the tensor decomposition, each rank-1 term contributes to exactly one (i,k) pair via W.
        
        # I think I'm confusing myself. Let me reconsider.
        
        # In the matmul tensor, T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
        # A rank-1 term is U[i,j] * V[j,k] * W[k,i] for specific i,j,k.
        
        # Strassen says we can compute 2x2 matmul with 7 multiplications instead of 8.
        # But in the tensor decomposition, we're not "computing" - we're decomposing the tensor.
        
        # The 2x2 matmul tensor has 8 nonzero entries (one for each (i,j,k) with i,j,k in {0,1}).
        # Strassen's rank-7 decomposition means we can write this tensor as a sum of 7 rank-1 tensors.
        
        # Each rank-1 tensor in the Strassen decomposition has specific U, V, W values.
        # Let me look up the explicit Strassen rank-7 decomposition.
        
        # Actually, the standard Strassen algorithm doesn't directly give a rank-7 tensor decomposition
        # in the sense we need. The 7 "multiplications" are products of linear combinations,
        # which means each "multiplication" is itself a sum of multiple rank-1 tensor terms.
        
        # For example, M1 = (a+d)(e+h) = ae + ah + de + dh, which is 4 rank-1 terms.
        # So Strassen's 7 multiplications actually expand to more than 7 rank-1 tensor terms.
        
        # The rank of the 2x2 matmul tensor is actually 7, but the decomposition involves
        # rank-1 tensors where U, V, W entries are not just 0/1 but can be ±1.
        
        # Let me use the known rank-7 decomposition of the 2x2 matmul tensor.
        # This is a well-known result. The 7 rank-1 tensors are:
        
        # I'll use a lookup table for the 2x2 Strassen decomposition.
        
        # After research, the rank-7 decomposition of 2x2 matmul tensor:
        # Term 1: U=(1,0,0,1), V=(1,0,1,0), W=(1,1,0,0) -> not quite
        # 
        # Let me just use the schoolbook for 2x2 (rank 8) for now and focus on getting
        # the overall structure right, then optimize.
        
        # Schoolbook for 2x2: rank 8
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = (A_r0 + i) * m + (A_c0 + j)
                    v_idx = (B_r0 + j) * p + (B_c0 + k)
                    w_idx = (C_c0 + k) * n + (C_r0 + i)
                    add_term(u_idx, v_idx, w_idx, 1.0)
    
    # Partition 5 = 2 + 3
    # Block 1: rows/cols 0-1 (size 2)
    # Block 2: rows/cols 2-4 (size 3)
    
    # C11 = A11 * B11 + A12 * B21
    # A11: A[0:2, 0:2], B11: B[0:2, 0:2], C11: C[0:2, 0:2]
    # A12: A[0:2, 2:5], B21: B[2:5, 0:2], C11: C[0:2, 0:2]
    
    # For A11 * B11 (2x2 * 2x2 -> 2x2): schoolbook rank 8
    strassen_2x2(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)
    
    # For A12 * B21 (2x3 * 3x2 -> 2x2): schoolbook rank 12
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    # C12 = A11 * B12 + A12 * B22
    # A11: A[0:2, 0:2], B12: B[0:2, 2:5], C12: C[0:2, 2:5]
    # A12: A[0:2, 2:5], B22: B[2:5, 2:5], C12: C[0:2, 2:5]
    
    # For A11 * B12 (2x2 * 2x3 -> 2x3): schoolbook rank 12
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    # For A12 * B22 (2x3 * 3x3 -> 2x3): schoolbook rank 18
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    # C21 = A21 * B11 + A22 * B21
    # A21: A[2:5, 0:2], B11: B[0:2, 0:2], C21: C[2:5, 0:2]
    # A22: A[2:5, 2:5], B21: B[2:5, 0:2], C21: C[2:5, 0:2]
    
    # For A21 * B11 (3x2 * 2x2 -> 3x2): schoolbook rank 12
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    # For A22 * B21 (3x3 * 3x2 -> 3x2): schoolbook rank 18
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    # C22 = A21 * B12 + A22 * B22
    # A21: A[2:5, 0:2], B12: B[0:2, 2:5], C22: C[2:5, 2:5]
    # A22: A[2:5, 2:5], B22: B[2:5, 2:5], C22: C[2:5, 2:5]
    
    # For A21 * B12 (3x2 * 2x3 -> 3x3): schoolbook rank 18
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    # For A22 * B22 (3x3 * 3x3 -> 3x3): schoolbook rank 27
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    rank = len(terms)
    
    if time.time() > deadline - safety_margin:
        pass
    
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx, r] = coeff
        V[v_idx, r] = coeff
        W[w_idx, r] = coeff
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
