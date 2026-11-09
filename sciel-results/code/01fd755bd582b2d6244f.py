import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    
    # We construct a decomposition with rank 39.
    # Part 1: M_{2,1,7} using columns 2 of A and row 2 of B. Rank 14.
    # Part 2: M_{2,2,7} using columns 0,1 of A and rows 0,1 of B. Rank 25.
    #   - 3 Strassen blocks (2x2x2) for cols (0,1), (2,3), (4,5). Rank 3*7=21.
    #   - 1 Trivial block (2x2x1) for col 6. Rank 4.
    
    total_rank = 14 + 21 + 4
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Helper to add a trivial 1x1x1 term
    # u_idx, v_idx, w_idx
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    # Part 1: M_{2,1,7}
    # A is 2x3, use col 2. B is 3x7, use row 2.
    # C is 2x7.
    # Term for i in 0..1, k in 0..6
    for i in range(n):
        for k in range(p):
            # A index: i * m + 2
            # B index: 2 * p + k
            # W index: k * n + i
            add_term(i * m + 2, 2 * p + k, k * n + i)
            
    # Part 2: M_{2,2,7}
    # A is 2x3, use cols 0,1. B is 3x7, use rows 0,1.
    # We process B columns in pairs.
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(k1, k2):
        """
        Adds 7 terms for multiplying A[0:2, 0:2] by B[0:2, k1:k2+1]
        k1, k2 are the column indices in the full B matrix.
        """
        nonlocal col
        
        # A indices:
        # A00: 0*m + 0 = 0
        # A01: 0*m + 1 = 1
        # A10: 1*m + 0 = 3
        # A11: 1*m + 1 = 4
        
        # B indices:
        # B00 (row 0, col k1): 0*p + k1 = k1
        # B01 (row 0, col k2): 0*p + k2 = k2
        # B10 (row 1, col k1): 1*p + k1 = p + k1
        # B11 (row 1, col k2): 1*p + k2 = p + k2
        
        # W indices:
        # C00 (row 0, col k1): k1*n + 0 = k1*2
        # C01 (row 0, col k2): k2*n + 0 = k2*2
        # C10 (row 1, col k1): k1*n + 1 = k1*2 + 1
        # C11 (row 1, col k2): k2*n + 1 = k2*2 + 1
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11
        U[0, col] = 1; U[4, col] = 1
        V[k1, col] = 1; V[p+k2, col] = 1
        W[k1*2, col] = 1; W[k2*2, col] = 1; W[k1*2+1, col] = 1; W[k2*2+1, col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[3, col] = 1; U[4, col] = 1
        V[k1, col] = 1
        W[k1*2+1, col] = 1; W[k2*2+1, col] = 1
        col += 1
        
        # Term 3: A00(B01-B11)
        U[0, col] = 1
        V[k2, col] = 1; V[p+k2, col] = -1
        W[k2*2, col] = 1; W[k2*2+1, col] = 1
        col += 1
        
        # Term 4: A11(B10-B00)
        U[4, col] = 1
        V[p+k1, col] = 1; V[k1, col] = -1
        W[k1*2, col] = 1; W[k1*2+1, col] = 1
        col += 1
        
        # Term 5: (A00+A10)B11
        U[0, col] = 1; U[3, col] = 1
        V[p+k2, col] = 1
        W[k1*2, col] = -1; W[k2*2, col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[3, col] = 1; U[0, col] = -1
        V[k1, col] = 1; V[k2, col] = 1
        W[k2*2+1, col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[1, col] = 1; U[4, col] = -1
        V[p+k1, col] = 1; V[p+k2, col] = 1
        W[k1*2, col] = 1; W[k2*2+1, col] = 1
        col += 1

    # Apply Strassen for cols (0,1), (2,3), (4,5)
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    
    # Apply Trivial for col 6
    # 2x2x1
    # A00 B00 -> C00
    # A01 B00 -> C01? No.
    # A is 2x2, B is 2x1.
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    
    # Term 1: A00 B00
    U[0, col] = 1
    V[0*p + 6, col] = 1 # B06
    W[6*2 + 0, col] = 1 # C06
    col += 1
    
    # Term 2: A01 B10
    U[1, col] = 1
    V[1*p + 6, col] = 1 # B16
    W[6*2 + 0, col] = 1 # C06
    col += 1
    
    # Term 3: A10 B00
    U[3, col] = 1
    V[0*p + 6, col] = 1 # B06
    W[6*2 + 1, col] = 1 # C16
    col += 1
    
    # Term 4: A11 B10
    U[4, col] = 1
    V[1*p + 6, col] = 1 # B16
    W[6*2 + 1, col] = 1 # C16
    col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
