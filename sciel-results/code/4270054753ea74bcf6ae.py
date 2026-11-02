import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy: Partition 12 columns into 4 groups of 3.
    # For each group of 3 columns, use a rank-5 decomposition for 2x2x3 matrix multiplication.
    # Total rank = 4 * 5 = 20.
    
    num_groups = 4
    cols_per_group = 3
    rank_per_group = 5
    rank = num_groups * rank_per_group  # 20
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 20
    V = [[0] * rank for _ in range(m * p)]  # 24 x 20
    W = [[0] * rank for _ in range(p * n)]  # 24 x 20
    
    # U indices: row = i*m + j
    u_a00 = 0
    u_a01 = 1
    u_a10 = 2
    u_a11 = 3
    
    # Define the 5 terms for a 2x2x3 block
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict { (i,j): val }
    # V_coeffs: dict { (j,k_local): val } where k_local in {0,1,2}
    # W_coeffs: dict { (k_local,i): val } where k_local in {0,1,2}
    
    terms = []
    
    # M1 = (A00 + A11)(B00 + B10)
    # C00 += M1, C21 += M1  (C[k=2, i=1])
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,0): 1},
        'W': {(0,0): 1, (2,1): 1}
    })
    
    # M2 = (A10 + A11)B00
    # C10 += M2, C21 -= M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(0,1): 1, (2,1): -1}
    })
    
    # M3 = A00(B01 - B11)
    # C01 += M3, C21 += M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1, (2,1): 1}
    })
    
    # M4 = A11(B10 - B00)
    # C00 += M4, C10 += M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A00 + A01)B11
    # C00 -= M5, C01 += M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (1,0): 1}
    })
    
    # Wait, the above 5 terms are just Strassen for 2x2x2 applied to the first 2 columns.
    # This does NOT compute column 2 (index 2) of C.
    # C[k=2, i=0] and C[k=2, i=1] are missing.
    # My "rank-5" hypothesis was wrong. The rank of 2x2x3 is NOT 5.
    # The rank of 2x2x3 is likely 8 or 9.
    # If I use 5 terms, I am missing terms.
    
    # Let's revert to the safe Strassen 42.
    # The prompt asks to "combine" but if the strategy is invalid, I must fall back.
    # However, I should check if there is a valid rank-8 or rank-9 for 2x2x3?
    # Rank of 2x2x2 is 7.
    # Rank of 2x2x3 is at least 7.
    # Is it 8?
    # If I use 8 per group, total rank 32.
    # If I use 9 per group, total rank 36.
    
    # Since I don't have a verified rank-5 decomposition, I will stick to the known valid Strassen 42.
    # But wait, the directive says "Apply a direct sum decomposition... using the proven rank-5...".
    # If it's not proven, I can't use it.
    # The current program has a fallback to Strassen 42.
    # I will implement the Strassen 42 correctly as the primary solution because the rank-5 is likely a hallucination or error in the prompt's inspiration.
    
    # Re-implement Strassen 42
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        base = t * 7
        
        v_b00 = 0 * p + 2 * t
        v_b01 = 0 * p + 2 * t + 1
        v_b10 = 1 * p + 2 * t
        v_b11 = 1 * p + 2 * t + 1
        
        w_c00 = (2 * t) * n + 0
        w_c01 = (2 * t + 1) * n + 0
        w_c10 = (2 * t) * n + 1
        w_c11 = (2 * t + 1) * n + 1
        
        u_a00 = 0
        u_a01 = 1
        u_a10 = 2
        u_a11 = 3
        
        c = base + 0
        U[u_a00][c] = 1; U[u_a11][c] = 1
        V[v_b00][c] = 1; V[v_b11][c] = 1
        W[w_c00][c] = 1; W[w_c11][c] = 1
        
        c = base + 1
        U[u_a10][c] = 1; U[u_a11][c] = 1
        V[v_b00][c] = 1
        W[w_c10][c] = 1; W[w_c11][c] = -1
        
        c = base + 2
        U[u_a00][c] = 1
        V[v_b01][c] = 1; V[v_b11][c] = -1
        W[w_c01][c] = 1; W[w_c11][c] = 1
        
        c = base + 3
        U[u_a11][c] = 1
        V[v_b10][c] = 1; V[v_b00][c] = -1
        W[w_c00][c] = 1; W[w_c10][c] = 1
        
        c = base + 4
        U[u_a00][c] = 1; U[u_a01][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = -1; W[w_c01][c] = 1
        
        c = base + 5
        U[u_a10][c] = 1; U[u_a00][c] = -1
        V[v_b00][c] = 1; V[v_b01][c] = 1
        W[w_c11][c] = 1
        
        c = base + 6
        U[u_a01][c] = 1; U[u_a11][c] = -1
        V[v_b10][c] = 1; V[v_b11][c] = 1
        W[w_c00][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
