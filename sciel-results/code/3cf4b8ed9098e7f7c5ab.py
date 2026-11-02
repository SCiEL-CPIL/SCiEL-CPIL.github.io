import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin before 150s limit
    
    n, m, p = DIMENSIONS
    # 13 = 6 * 2 + 1
    # Use Strassen (rank 7) for 6 blocks of 2 columns, and schoolbook (rank 4) for the last column
    # Total rank = 6 * 7 + 4 = 46
    
    rank = 46
    
    # U: (n*m) x rank = 4 x 46
    # V: (m*p) x rank = 26 x 46
    # W: (p*n) x rank = 26 x 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")

    # Strassen coefficients for 2x2x2:
    # U indices: i*m + j (i=0,1; j=0,1) -> 0:00, 1:01, 2:10, 3:11
    # V indices: j*p + k (j=0,1; k=0,1) -> k, 13+k
    # W indices: k*n + i (k=0,1; i=0,1) -> 2k, 2k+1
    
    # M1: (a00+a11)(b00+b11) -> c00 += 1, c01 += 1, c10 += 1, c11 += 1
    # M2: (a10+a11)b00 -> c10 += 1, c11 -= 1
    # M3: a00(b01-b11) -> c01 += 1, c11 += 1
    # M4: a11(b10-b00) -> c00 += 1, c10 += 1
    # M5: (a00+a01)b11 -> c00 -= 1, c01 += 1
    # M6: (a10-a00)(b00+b01) -> c11 += 1
    # M7: (a01-a11)(b10+b11) -> c00 += 1
    
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # M1
        add_term(
            {0: 1, 3: 1},
            {k0: 1, 13 + k1: 1},
            {2 * k0: 1, 2 * k1 + 1: 1}
        )
        
        # M2
        add_term(
            {2: 1, 3: 1},
            {k0: 1},
            {2 * k0 + 1: 1, 2 * k1 + 1: -1}
        )
        
        # M3
        add_term(
            {0: 1},
            {k1: 1, 13 + k1: -1},
            {2 * k1: 1, 2 * k1 + 1: 1}
        )
        
        # M4
        add_term(
            {3: 1},
            {13 + k0: 1, k0: -1},
            {2 * k0: 1, 2 * k0 + 1: 1}
        )
        
        # M5
        add_term(
            {0: 1, 1: 1},
            {13 + k1: 1},
            {2 * k0: -1, 2 * k1: 1}
        )
        
        # M6
        add_term(
            {2: 1, 0: -1},
            {k0: 1, k1: 1},
            {2 * k1 + 1: 1}
        )
        
        # M7
        add_term(
            {1: 1, 3: -1},
            {13 + k0: 1, 13 + k1: 1},
            {2 * k0: 1}
        )
    
    # Last column k=12: schoolbook rank 4
    k = 12
    add_term({0: 1}, {k: 1}, {2 * k: 1})
    add_term({1: 1}, {13 + k: 1}, {2 * k: 1})
    add_term({2: 1}, {k: 1}, {2 * k + 1: 1})
    add_term({3: 1}, {13 + k: 1}, {2 * k + 1: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
