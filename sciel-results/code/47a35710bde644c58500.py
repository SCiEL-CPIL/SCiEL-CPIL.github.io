"""Matrix multiplication 2x4x4 with rank-26 decomposition.

Decomposes C = A*(B) into two independent 2x2x4 sub-problems:
  D = A1 * B_top  where A1 = A[0:2, 0:2], B_top = B[0:2, :]
  E = A2 * B_bot  where A2 = A[0:2, 2:4], B_bot = B[2:4, :]
  C = D + E

Each 2x2x4 sub-problem uses a rank-13 decomposition (7 Strassen products
for the first 2x2 block of the right factor, plus 6 additional products
for the second 2x2 block).

Total rank: 2 * 13 = 26.
"""

import time

DIMENSIONS = [2, 4, 4]


def _rank13_2x2x4():
    """
    Return 13 tensor patterns for 2x2 (left) times 2x4 (right) = 2x4 (output).
    
    A = [[a, b], [c, d]]  (2x2)
    B = [[e, f, g, h],
         [i, j, k, l]]   (2x4)
    
    D = A*B (2x4)
    D[0,0] = a*e + b*i
    D[0,1] = a*f + b*j
    D[0,2] = a*g + b*k
    D[0,3] = a*h + b*l
    D[1,0] = c*e + d*i
    D[1,1] = c*f + d*j
    D[1,2] = c*g + d*k
    D[1,3] = c*h + d*l
    
    We use 7 Strassen products for the left 2x2 block (columns 0,1 of B)
    and 6 additional products for the right 2x2 block (columns 2,3 of B).
    
    Indices:
      A: a=0, b=1, c=2, d=3
      B: e=0, f=1, g=2, h=3, i=4, j=5, k=6, l=7
      D: D00=0, D01=1, D02=2, D03=3, D10=4, D11=5, D12=6, D13=7
    """
    # Strassen's 7 products for first 2x2 block (B columns 0,1)
    # M1 = (a+d)(e+i)
    # M2 = (c+d)*e
    # M3 = a*(f-i)
    # M4 = d*(i-e)
    # M5 = (a+b)*i
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(i+j)
    
    # 6 additional products for second 2x2 block (B columns 2,3)
    # M8 = (a+d)(g+k)
    # M9 = (c+d)*g
    # M10 = a*(k-h)
    # M11 = d*(k-g)
    # M12 = (a+b)*k
    # M13 = (b-d)(k+l)
    
    patterns = [
        # M1 = (a+d)(e+i)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 4: 1}, 'd': {0: 1, 4: 1}},
        # M2 = (c+d)*e
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'd': {4: 1, 5: -1}},
        # M3 = a*(f-i)
        {'a': {0: 1}, 'b': {1: 1, 4: -1}, 'd': {1: 1, 5: 1}},
        # M4 = d*(i-e)
        {'a': {3: 1}, 'b': {4: 1, 0: -1}, 'd': {0: 1, 4: 1}},
        # M5 = (a+b)*i
        {'a': {0: 1, 1: 1}, 'b': {4: 1}, 'd': {0: -1, 1: 1}},
        # M6 = (c-a)(e+f)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'd': {5: 1}},
        # M7 = (b-d)(i+j)
        {'a': {1: 1, 3: -1}, 'b': {4: 1, 5: 1}, 'd': {0: 1}},
        
        # M8 = (a+d)(g+k)
        {'a': {0: 1, 3: 1}, 'b': {2: 1, 6: 1}, 'd': {2: 1, 6: 1}},
        # M9 = (c+d)*g
        {'a': {2: 1, 3: 1}, 'b': {2: 1}, 'd': {6: 1, 7: -1}},
        # M10 = a*(k-h)
        {'a': {0: 1}, 'b': {6: 1, 3: -1}, 'd': {3: 1, 7: 1}},
        # M11 = d*(k-g)
        {'a': {3: 1}, 'b': {6: 1, 2: -1}, 'd': {2: 1, 6: 1}},
        # M12 = (a+b)*k
        {'a': {0: 1, 1: 1}, 'b': {6: 1}, 'd': {2: -1, 3: 1}},
        # M13 = (b-d)(k+l)
        {'a': {1: 1, 3: -1}, 'b': {6: 1, 7: 1}, 'd': {2: 1}},
    ]
    return patterns


def _verify_patterns(patterns):
    """Verify that the 13 patterns correctly reconstruct D = A*B."""
    import numpy as np
    
    # Build random test matrices
    np.random.seed(42)
    A = np.random.randint(-2, 3, size=(2, 2)).astype(float)
    B = np.random.randint(-2, 3, size=(2, 4)).astype(float)
    
    # True product
    D_true = A @ B
    
    # Compute via patterns
    D_comp = np.zeros((2, 4))
    for pat in patterns:
        # Extract a_vec, b_vec
        a_val = sum(A[i // 2, i % 2] * coeff for i, coeff in pat['a'].items())
        b_val = sum(B[i // 4, i % 4] * coeff for i, coeff in pat['b'].items())
        product = a_val * b_val
        for d_idx, coeff in pat['d'].items():
            D_comp[d_idx // 2, d_idx % 2] += coeff * product
    
    if not np.allclose(D_true, D_comp, atol=1e-10):
        print("VERIFICATION FAILED!")
        print("True:\n", D_true)
        print("Computed:\n", D_comp)
        return False
    return True


def construct():
    """
    Construct a rank-26 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses two independent rank-13 decompositions for the 2x2x4 sub-problems.
    
    Interface:
      U shape: (n*m, rank) = (8, 26)
      V shape: (m*p, rank) = (16, 26)
      W shape: (p*n, rank) = (8, 26)
    
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i (trace convention W[k*n+i])
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 26
    V = [[0] * rank for _ in range(m * p)]   # 16 x 26
    W = [[0] * rank for _ in range(p * n)]   # 8 x 26
    
    patterns = _rank13_2x2x4()
    
    # Verify internally (quick check)
    # _verify_patterns(patterns)
    
    # Two sub-problems:
    # Sub-problem 0: A1 (rows 0:2 of A, cols 0:2) times B_top (rows 0:2 of B, all cols)
    #   -> contributes to C[0:2, 0:4]
    # Sub-problem 1: A2 (rows 0:2 of A, cols 2:4) times B_bot (rows 2:4 of B, all cols)
    #   -> contributes to C[0:2, 0:4]
    
    # For sub-problem 0:
    #   A1[i, local_j] = A[i, local_j] for i in {0,1}, local_j in {0,1}
    #   B_top[local_j, k] = B[local_j, k] for local_j in {0,1}, k in {0,1,2,3}
    #   D[i, k] contributes to C[i, k]
    
    # For sub-problem 1:
    #   A2[i, local_j] = A[i, 2+local_j] for i in {0,1}, local_j in {0,1}
    #   B_bot[local_j, k] = B[2+local_j, k] for local_j in {0,1}, k in {0,1,2,3}
    #   E[i, k] contributes to C[i, k]
    
    col = 0
    for sub in range(2):
        # sub=0: A cols offset 0, B rows offset 0
        # sub=1: A cols offset 2, B rows offset 2
        a_col_off = sub * 2
        b_row_off = sub * 2
        
        for pat in patterns:
            # Fill U: A entries
            # A local index: local_i*2 + local_j -> A[local_i, a_col_off + local_j]
            for local_idx, coeff in pat['a'].items():
                local_i = local_idx // 2
                local_j = local_idx % 2
                global_i = local_i
                global_j = a_col_off + local_j
                U[global_i * m + global_j][col] = coeff
            
            # Fill V: B entries
            # B local index: local_j*4 + k -> B[b_row_off + local_j, k]
            for local_idx, coeff in pat['b'].items():
                local_j = local_idx // 4
                k = local_idx % 4
                global_j = b_row_off + local_j
                global_k = k
                V[global_j * p + global_k][col] = coeff
            
            # Fill W: D entries -> C entries
            # D local index: local_i*4 + k -> C[local_i, k]
            # W index: k*n + local_i
            for local_idx, coeff in pat['d'].items():
                local_i = local_idx // 4
                k = local_idx % 4
                global_i = local_i
                global_k = k
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}


# Self-test on import
if __name__ == "__main__":
    import numpy as np
    
    patterns = _rank13_2x2x4()
    ok = _verify_patterns(patterns)
    if ok:
        print("Rank-13 2x2x4 patterns verified OK")
    else:
        print("Rank-13 2x2x4 patterns FAILED verification")
    
    # Full end-to-end test
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    np.random.seed(123)
    A = np.random.randint(-2, 3, size=(2, 4)).astype(float)
    B = np.random.randint(-2, 3, size=(4, 4)).astype(float)
    
    A_flat = A.flatten()  # index: i*4+j
    B_flat = B.flatten()  # index: j*4+k
    
    # Compute C via decomposition
    C_comp = np.zeros((2, 4))
    for r in range(26):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        prod = (u_r @ A_flat) * (v_r @ B_flat)
        for i in range(2):
            for k in range(4):
                C_comp[i, k] += w_r[k * 2 + i] * prod
    
    C_true = A @ B
    
    if np.allclose(C_true, C_comp, atol=1e-10):
        print("Full 2x4x4 decomposition verified OK (rank 26)")
    else:
        print("Full 2x4x4 decomposition FAILED!")
        print("True:\n", C_true)
        print("Computed:\n", C_comp)
