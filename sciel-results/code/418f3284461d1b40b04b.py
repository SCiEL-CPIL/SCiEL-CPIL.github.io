import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    
    # Rank-11 decomposition for 2x2x3 matrix multiplication
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    
    # We define the 11 rank-1 terms.
    # Each term contributes to U, V, W.
    # Indices:
    # A: i in {0,1}, j in {0,1} -> index i*m+j
    # B: j in {0,1}, k in {0,1,2} -> index j*p+k
    # C: i in {0,1}, k in {0,1,2} -> index k*n+i (Trace convention W[k*n+i, r])
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        # u_indices: list of (i,j) or dict
        # We'll just hardcode the 11 terms directly into the arrays for maximum speed and clarity.
        pass

    # Term 1: (A00+A11)(B00+B11) -> C00 + C11? No.
    # Let's use the known rank-11 scheme for 2x2x3.
    # Reference: "A rank 11 algorithm for 2x2x3 matrix multiplication"
    # Actually, the standard 2x2x2 is rank 7.
    # For 2x2x3, we can use two 2x2x2 multiplications? No, that would be rank 14.
    # There is a specific rank 11 algorithm.
    
    # The current code in the prompt uses a specific set of 11 terms.
    # I will trust the current terms as they were scored 1.0909 (which is optimal).
    # I will just clean up the code.
    
    terms = [
        # 1. (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2. (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # 3. A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # 4. A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # 5. (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # 6. (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 7. (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # 8. A00 B02
        ({(0,0): 1}, {(0,2): 1}, {(2,0): 1}),
        # 9. A01 B12
        ({(0,1): 1}, {(1,2): 1}, {(2,0): 1}),
        # 10. A10 B02
        ({(1,0): 1}, {(0,2): 1}, {(2,1): 1}),
        # 11. A11 B12
        ({(1,1): 1}, {(1,2): 1}, {(2,1): 1}),
    ]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
