"""Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition.

Strategy:
  Split A (2x4) into two 2x2 blocks A1=A[:,0:2], A2=A[:,2:4].
  Split B (4x4) into four 2x2 blocks:
    B11=B[0:2,0:2], B12=B[0:2,2:4]
    B21=B[2:4,0:2], B22=B[2:4,2:4]
  
  C = A1*B11 + A2*B21 + A1*B12 + A2*B22
    = (A1*B11 + A2*B21) + (A1*B12 + A2*B22)
  
  For each pair (left half and right half), we compute a (2x2)*(2x2) + (2x2)*(2x2)
  which is equivalent to computing a (2x4)*(2x2) product, but we can use a
  combined Strassen scheme.

  Actually, the cleanest rank-28 construction:
  
  For C_left (2x2) = A1*B11 + A2*B21:
    Use 7 Strassen multiplications on the "sum" structure.
    Specifically, define:
      M1 = (A1[0]+A2[0]) * (B11[0]+B21[0])  -- but this doesn't directly work.
    
  Let me use the standard block Strassen approach:
  
  For computing X*Y + Z*W where all are 2x2:
    We can use the identity:
      X*Y + Z*W = (X+Z)*(Y+W) - X*W - Z*Y
    That's 3 multiplications instead of 4, but we still need X*W and Z*Y.
    So total is 3+2 = 5? No, that's worse than 4.
    
    Actually: X*Y + Z*W. If we set P = X+Z, Q = Y+W, then:
    P*Q = X*Y + X*W + Z*Y + Z*W
    So X*Y + Z*W = P*Q - X*W - Z*Y
    That's 3 mults (P*Q, X*W, Z*Y) but we need X*W and Z*Y which are 2 more, total 3.
    Wait, P*Q is 1 mult, X*W is 1, Z*Y is 1, total 3. But we want X*Y+Z*W.
    P*Q - X*W - Z*Y = X*Y + X*W + Z*Y + Z*W - X*W - Z*Y = X*Y + Z*W. Yes!
    So 3 multiplications for the "cross" part. But we still need to compute X*W and Z*Y
    which are 2x2*2x2 products, each needing 4 mults (schoolbook) or 7 (Strassen).
    That makes it worse.
    
  Better approach: Just use Strassen independently on each 2x2*2x2 product.
  Each (2x2)*(2x2) takes 7 mults with Strassen.
  We have 4 such products: A1*B11, A2*B21, A1*B12, A2*B22.
  Total: 4*7 = 28. This is the rank-28 decomposition.
  
  The key insight: we compute each 2x2 block product using Strassen's 7-multiplication
  algorithm, and the results are added to form C.
"""
import time

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    
    For C = A*B where A,B,C are 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    Strassen's 7 products:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)(b11)
    M3 = (a11)(b12-b22)
    M4 = (a22)(b21-b11)
    M5 = (a11+a12)(b22)
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4, M1-M2+M3+M6]]
    
    We return for each of the 7 multiplications:
      - The linear combination of A entries (for U matrix)
      - The linear combination of B entries (for V matrix)
      - The contributions to each C entry (for W matrix)
    """
    # A entries indexed as: a11=0, a12=1, a21=2, a22=3
    # B entries indexed as: b11=0, b12=1, b21=2, b22=3
    # C entries indexed as: c11=0, c12=1, c21=2, c22=3
    
    # Each entry: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from A-index to coefficient
    # b_coeffs: dict from B-index to coefficient  
    # c_coeffs: dict from C-index to coefficient
    
    patterns = []
    
    # M1 = (a11+a22)(b11+b22)
    patterns.append({
        'a': {0: 1, 3: 1},
        'b': {0: 1, 3: 1},
        'c': {0: 1, 2: -1, 3: 1}  # c11+=M1, c21-=M1, c22+=M1 ... wait let me recheck
    })
    # Actually:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Let me redo carefully.
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11)
    # M3 = (a11)(b12-b22)
    # M4 = (a22)(b21-b11)
    # M5 = (a11+a12)(b22)
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    patterns = [
        # M1
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm on each of the four 2x2 block products:
      A1*B11, A2*B21, A1*B12, A2*B22
    
    Total rank: 4 * 7 = 28.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)  
      W shape: (p*n, rank) = (8, 28)
    
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i (trace convention W[k*n+i])
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    strassen_patterns = _strassen_2x2()
    
    # Four block products:
    # Block 1: A1 * B11 -> contributes to C[:, 0:2] (C_left)
    #   A1 = A[:, 0:2]: rows i=0,1; cols j=0,1
    #   B11 = B[0:2, 0:2]: rows j=0,1; cols k=0,1
    #   Result: C[i, k] for i in {0,1}, k in {0,1}
    #
    # Block 2: A2 * B21 -> contributes to C[:, 0:2] (C_left)
    #   A2 = A[:, 2:4]: rows i=0,1; cols j=2,3
    #   B21 = B[2:4, 0:2]: rows j=2,3; cols k=0,1
    #   Result: C[i, k] for i in {0,1}, k in {0,1}
    #
    # Block 3: A1 * B12 -> contributes to C[:, 2:4] (C_right)
    #   A1 = A[:, 0:2]: rows i=0,1; cols j=0,1
    #   B12 = B[0:2, 2:4]: rows j=0,1; cols k=2,3
    #   Result: C[i, k] for i in {0,1}, k in {2,3}
    #
    # Block 4: A2 * B22 -> contributes to C[:, 2:4] (C_right)
    #   A2 = A[:, 2:4]: rows i=0,1; cols j=2,3
    #   B22 = B[2:4, 2:4]: rows j=2,3; cols k=2,3
    #   Result: C[i, k] for i in {0,1}, k in {2,3}
    
    # For each block product, we use 7 Strassen multiplications.
    # The local 2x2 A matrix entries map to global A indices:
    #   local (r,c) -> global (i=r, j=base_j+c) where base_j is 0 or 2
    # The local 2x2 B matrix entries map to global B indices:
    #   local (r,c) -> global (j=base_j_r+c, k=base_k+c) where base_j_r is 0 or 2, base_k is 0 or 2
    # The local 2x2 C matrix entries map to global C indices:
    #   local (r,c) -> global (i=r, k=base_k+c) where base_k is 0 or 2
    
    # Block definitions: (a_base_j, b_base_j, b_base_k, c_base_k)
    blocks = [
        (0, 0, 0, 0),  # A1*B11 -> C[:, 0:2]
        (2, 2, 0, 0),  # A2*B21 -> C[:, 0:2]
        (0, 0, 2, 2),  # A1*B12 -> C[:, 2:4]
        (2, 2, 2, 2),  # A2*B22 -> C[:, 2:4]
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen_patterns:
            # Fill U: A entries
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U[global_i * m + global_j][col] = coeff
            
            # Fill V: B entries
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V[global_j * p + global_k][col] = coeff
            
            # Fill W: C entries
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                # W index: k*n + i (trace convention)
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
