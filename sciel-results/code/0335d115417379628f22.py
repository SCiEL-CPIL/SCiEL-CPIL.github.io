import numpy as np

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    Each term is a tuple (u, v, w) where u, v, w are 2x2 matrices (flattened to 4x1 vectors? No, just indices).
    Actually, we need the coefficients for U, V, W.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    For a 2x2x2 block, n=2, m=2, p=2.
    U indices: 0..3 (i*m+j)
    V indices: 0..3 (j*p+k)
    W indices: 0..3 (k*n+i)
    """
    # Strassen's 7 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # We need to express these as sums of rank-1 tensors.
    # Each M corresponds to a product of two linear combinations.
    # This product expands into 4 rank-1 terms? No, the product of two linear combinations is a single "multiplication" in the tensor decomposition sense?
    # Wait. The rank of the tensor is the number of rank-1 terms.
    # In Strassen, we replace 8 multiplications with 7.
    # Each "multiplication" in the algorithm corresponds to a rank-1 term in the decomposition of the *algorithm*, but the algorithm itself is a decomposition of the tensor.
    # Yes, the 7 products M1...M7 are the rank-1 terms.
    # So we need to define U, V, W such that U * V * W sums to the tensor.
    # For each M, we have a specific U vector, V vector, W vector.
    
    # M1: (A00+A11)(B00+B11)
    # A00 is index 0 in A, A11 is index 3.
    # B00 is index 0 in B, B11 is index 3.
    # C00 gets M1, C01 gets -M1, C10 gets -M1, C11 gets M1? No.
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M4 + M6
    
    # This means the rank-1 terms are not just M1...M7 directly.
    # The decomposition is:
    # T = sum_{r=1}^7 U_r \otimes V_r \otimes W_r
    # Where U_r, V_r, W_r are vectors.
    
    # Let's define the 7 terms explicitly.
    # Term 1: M1
    # U: A00 + A11 -> [1, 0, 0, 1]
    # V: B00 + B11 -> [1, 0, 0, 1]
    # W: C00 + C11? No.
    # The contribution of M1 to C is:
    # C00 += M1, C01 += 0, C10 += 0, C11 += M1?
    # Actually, the standard Strassen decomposition maps M1 to C00 and C11 with sign +, and C01, C10 with sign 0?
    # No, C01 = M3 + M7. M1 does not appear in C01.
    # So W for M1 has non-zero entries only where M1 contributes.
    # M1 contributes to C00 and C11.
    # So W1 = [1, 0, 0, 1] (indices: k*n+i. C00 is 0, C11 is 3).
    
    # Let's list the 7 terms (U, V, W) for 2x2x2.
    # Indices: A: 0,1,2,3 (i*m+j). B: 0,1,2,3 (j*p+k). C: 0,1,2,3 (k*n+i).
    
    terms = []
    # 1. M1 = (A00+A11)(B00+B11) -> C00, C11
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # 2. M2 = (A10+A11)B00 -> C10
    # A10 is 2, A11 is 3. B00 is 0.
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0])) # C10 is index 2 (k=1, i=0 -> 1*2+0=2)
    # 3. M3 = A00(B01-B11) -> C01
    # A00 is 0. B01 is 1, B11 is 3.
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0])) # C01 is index 1 (k=0, i=1 -> 0*2+1=1)
    # 4. M4 = A11(B10-B00) -> C10, C11
    # A11 is 3. B10 is 2, B00 is 0.
    # M4 contributes to C10 (-) and C11 (+).
    terms.append(([0,0,0,1], [0,0,1,-1], [0,0,-1,1]))
    # 5. M5 = (A00+A01)B11 -> C00, C01
    # A00 is 0, A01 is 1. B11 is 3.
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # 6. M6 = (A10-A00)(B00+B10) -> C10, C11
    # A10 is 2, A00 is 0. B00 is 0, B10 is 2.
    terms.append(([0,0,1,-1], [1,0,1,0], [0,0,1,1]))
    # 7. M7 = (A01-A11)(B01+B11) -> C01, C11
    # A01 is 1, A11 is 3. B01 is 1, B11 is 3.
    terms.append(([0,1,0,-1], [0,1,0,1], [0,1,0,1]))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 12 blocks of 2x2x2 (rank 7) + 6 blocks of 2x1x2 (rank 1) = 90.
    rank = 90
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    # Iterate over blocks
    # i_block: 0, 1 (size 2)
    # j_block: 0, 1 (size 2), 2 (size 1)
    # k_block: 0, 1, 2 (size 2)
    
    for i_b in range(2):
        for j_b in range(3):
            for k_b in range(3):
                # Determine block sizes
                i_start = i_b * 2
                i_end = i_start + 2
                i_size = 2
                
                if j_b < 2:
                    j_start = j_b * 2
                    j_end = j_start + 2
                    j_size = 2
                else:
                    j_start = 4
                    j_end = 5
                    j_size = 1
                
                k_start = k_b * 2
                k_end = k_start + 2
                k_size = 2
                
                if j_size == 2:
                    # Use Strassen 2x2x2
                    for t in strassen_terms:
                        u_local, v_local, w_local = t
                        # Map local indices to global
                        # A indices: i in [i_start, i_end), j in [j_start, j_end)
                        # Local A index: li * 2 + lj
                        # Global A index: (i_start + li) * m + (j_start + lj)
                        for li in range(2):
                            for lj in range(2):
                                if u_local[li * 2 + lj] != 0:
                                    gi = i_start + li
                                    gj = j_start + lj
                                    U[gi * m + gj][col] = u_local[li * 2 + lj]
                        
                        # B indices: j in [j_start, j_end), k in [k_start, k_end)
                        # Local B index: lj * 2 + lk
                        # Global B index: (j_start + lj) * p + (k_start + lk)
                        for lj in range(2):
                            for lk in range(2):
                                if v_local[lj * 2 + lk] != 0:
                                    gj = j_start + lj
                                    gk = k_start + lk
                                    V[gj * p + gk][col] = v_local[lj * 2 + lk]
                        
                        # C indices: k in [k_start, k_end), i in [i_start, i_end)
                        # Local C index: lk * 2 + li
                        # Global C index: (k_start + lk) * n + (i_start + li)
                        for lk in range(2):
                            for li in range(2):
                                if w_local[lk * 2 + li] != 0:
                                    gk = k_start + lk
                                    gi = i_start + li
                                    W[gk * n + gi][col] = w_local[lk * 2 + li]
                        
                        col += 1
                else:
                    # j_size == 1
                    # This is a 2x1x2 block.
                    # A is 2x1, B is 1x2.
                    # C = A B.
                    # C00 = A00 B00
                    # C01 = A00 B01
                    # C10 = A10 B00
                    # C11 = A10 B01
                    # This is 4 rank-1 terms?
                    # No, it's just 2 multiplications: A00*B00, A00*B01, A10*B00, A10*B01.
                    # Wait, B has only 1 column (j fixed).
                    # B indices: j=j_start, k in [k_start, k_end).
                    # So B is a vector of length 2.
                    # A is a vector of length 2.
                    # C is 2x2.
                    # C[li, lk] = A[li] * B[lk].
                    # This is the outer product of A and B.
                    # The outer product of two vectors is a rank-1 matrix?
                    # No, the tensor T_{li, lk} = A_li B_lk.
                    # This is a single rank-1 term!
                    # U: A vector. V: B vector. W: Identity?
                    # U[li] = A_li. V[lk] = B_lk. W[li, lk] = 1.
                    # So 1 rank-1 term.
                    
                    # A indices: i in [i_start, i_end), j = j_start
                    # B indices: j = j_start, k in [k_start, k_end)
                    # C indices: k in [k_start, k_end), i in [i_start, i_end)
                    
                    # U
                    for li in range(2):
                        gi = i_start + li
                        U[gi * m + j_start][col] = 1
                    # V
                    for lk in range(2):
                        gk = k_start + lk
                        V[j_start * p + gk][col] = 1
                    # W
                    for lk in range(2):
                        for li in range(2):
                            gk = k_start + lk
                            gi = i_start + li
                            W[gk * n + gi][col] = 1
                            
                    col += 1

    return {"U": U, "V": V, "W": W}
