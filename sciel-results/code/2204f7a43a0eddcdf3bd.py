import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 6 blocks of 2 columns using Strassen (rank 7 each)
    # 1 block of 1 column using schoolbook (rank 4)
    rank = 6 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_indices: list of (index, coeff)
    # v_indices: list of (index, coeff)
    # w_indices: list of (index, coeff)
    def add_term(u_idx, v_idx, w_idx):
        nonlocal col
        for i, c in u_idx:
            U[i][col] += c
        for i, c in v_idx:
            V[i][col] += c
        for i, c in w_idx:
            W[i][col] += c
        col += 1
        if time.time() > deadline:
            # Fallback to schoolbook if we somehow run out of time (unlikely)
            # But since it's O(1), this won't happen.
            pass

    # Strassen block for columns k, k+1
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 
    #   0, k -> k
    #   0, k+1 -> k+1
    #   1, k -> 13+k
    #   1, k+1 -> 13+k+1
    # W indices:
    #   0, k -> 2k
    #   0, k+1 -> 2k+2
    #   1, k -> 2k+1
    #   1, k+1 -> 2k+3
    
    for t in range(6):
        k = 2 * t
        k1 = 2 * t + 1
        
        # P1 = (A00 + A11)(B0k + B1,k+1)
        # U: A00(0), A11(3)
        # V: B0k(k), B1,k+1(13+k+1)
        # W: C0k(2k), C1,k+1(2k+3)
        add_term([(0, 1), (3, 1)], 
                 [(k, 1), (13 + k + 1, 1)], 
                 [(2 * k, 1), (2 * k + 3, 1)])
        
        # P2 = (A10 + A11) B0k
        # U: A10(2), A11(3)
        # V: B0k(k)
        # W: C1k(2k+1), C1,k+1(2k+3) -> C1k gets +1, C1,k+1 gets -1
        add_term([(2, 1), (3, 1)], 
                 [(k, 1)], 
                 [(2 * k + 1, 1), (2 * k + 3, -1)])
        
        # P3 = A00 (B1k - B0k)
        # U: A00(0)
        # V: B1k(13+k), B0k(k) -> B1k +1, B0k -1
        # W: C0,k+1(2k+2), C1,k+1(2k+3) -> both +1
        add_term([(0, 1)], 
                 [(13 + k, 1), (k, -1)], 
                 [(2 * k + 2, 1), (2 * k + 3, 1)])
        
        # P4 = A11 (B0,k+1 - B1,k+1)
        # U: A11(3)
        # V: B0,k+1(k+1), B1,k+1(13+k+1) -> B0,k+1 +1, B1,k+1 -1
        # W: C0k(2k), C1k(2k+1) -> both +1
        add_term([(3, 1)], 
                 [(k + 1, 1), (13 + k + 1, -1)], 
                 [(2 * k, 1), (2 * k + 1, 1)])
        
        # P5 = (A00 + A01) B1,k+1
        # U: A00(0), A01(1)
        # V: B1,k+1(13+k+1)
        # W: C0k(2k), C0,k+1(2k+2) -> C0k -1, C0,k+1 +1
        add_term([(0, 1), (1, 1)], 
                 [(13 + k + 1, 1)], 
                 [(2 * k, -1), (2 * k + 2, 1)])
        
        # P6 = (A10 - A00) (B0k + B0,k+1)
        # U: A10(2), A00(0) -> A10 +1, A00 -1
        # V: B0k(k), B0,k+1(k+1)
        # W: C1,k+1(2k+3) -> +1
        add_term([(2, 1), (0, -1)], 
                 [(k, 1), (k + 1, 1)], 
                 [(2 * k + 3, 1)])
        
        # P7 = (A01 - A11) (B1k + B1,k+1)
        # U: A01(1), A11(3) -> A01 +1, A11 -1
        # V: B1k(13+k), B1,k+1(13+k+1)
        # W: C0k(2k) -> +1
        add_term([(1, 1), (3, -1)], 
                 [(13 + k, 1), (13 + k + 1, 1)], 
                 [(2 * k, 1)])

    # Last column k=12
    k = 12
    # Term 1: A00 B0k -> C0k
    add_term([(0, 1)], [(k, 1)], [(2 * k, 1)])
    # Term 2: A01 B1k -> C0k
    add_term([(1, 1)], [(13 + k, 1)], [(2 * k, 1)])
    # Term 3: A10 B0k -> C1k
    add_term([(2, 1)], [(k, 1)], [(2 * k + 1, 1)])
    # Term 4: A11 B1k -> C1k
    add_term([(3, 1)], [(13 + k, 1)], [(2 * k + 1, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
