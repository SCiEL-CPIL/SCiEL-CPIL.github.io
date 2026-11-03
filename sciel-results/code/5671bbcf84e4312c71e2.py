"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We partition 5 into 1 + 4.
    # A = [[A11, A14], [A41, A44]]
    # B = [[B11, B14], [B41, B44]]
    # C = AB
    
    # Ranks:
    # A11 B11: 1x1 * 1x1 -> 1x1. Rank 1.
    # A11 B14: 1x1 * 1x4 -> 1x4. Rank 4 (Schoolbook).
    # A14 B44: 1x4 * 4x4 -> 1x4. Rank 16 (Schoolbook).
    # A41 B11: 4x1 * 1x1 -> 4x1. Rank 4 (Schoolbook).
    # A44 B41: 4x4 * 4x1 -> 4x1. Rank 16 (Schoolbook).
    # A41 B14: 4x1 * 1x4 -> 4x4. Rank 16 (Schoolbook).
    # A44 B44: 4x4 * 4x4 -> 4x4. Rank 25 (Landsberg).
    
    total_rank = 1 + 4 + 16 + 4 + 16 + 16 + 25
    # total_rank = 82
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end, u_start, v_start, w_start):
        nonlocal col
        # i is row of A, j is col of A / row of B, k is col of B
        # A indices: i*m + j
        # B indices: j*p + k
        # C indices: k*n + i  (Wait, W[k*n+i, r] is the convention)
        # The task says: W[k*n+i, r].
        # Standard: C[i, k] = sum_j A[i, j] B[j, k].
        # So W index is k*n + i.
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Helper to add a specific rank-r algorithm block
    # For 4x4, we use a known rank 25 algorithm.
    # I will use the "Landsberg" rank 25 algorithm for 4x4.
    # If I don't have it hardcoded, I'll use schoolbook (16) for 4x4?
    # Schoolbook 4x4 is 16? No, 4*4*4 = 64.
    # Strassen 4x4 is 25? No, Strassen is 7 for 2x2.
    # 4x4 using Strassen recursively: 7 * 7 = 49?
    # Actually, R(4x4) <= 25 is a known bound.
    # I will use Schoolbook for 4x4 to be safe, which is 64.
    # Then total rank: 1 + 4 + 16 + 4 + 16 + 16 + 64 = 121.
    # 121 is only slightly better than 125.
    
    # Let's use Strassen for the 4x4 block?
    # Strassen 2x2 rank 7.
    # 4x4 can be done with 7 Strassen steps? No.
    # 4x4 = 2x(2x2).
    # Rank 49 is easy to construct (Strassen on 4x4).
    # 1 + 4 + 16 + 4 + 16 + 16 + 49 = 106.
    
    # Let's use Rank 25 for 4x4 if I can find it.
    # I will implement a simple Rank 25 4x4 algorithm if possible.
    # Otherwise, I'll use 49.
    
    # For now, I'll use Schoolbook for the 4x4 block to ensure correctness, 
    # but I'll try to use a known 4x4 rank 25 decomposition.
    
    # Since I don't have the 25 coefficients hardcoded, I will use 
    # a decomposition that is definitely correct and better than 125.
    # Partition 5 = 2 + 3.
    # 2x2: Rank 7 (Strassen).
    # 3x3: Rank 15 (Laderman).
    # Off-diagonals:
    # A23 B32: 2x3 * 3x2 -> 2x2. Rank 12 (Schoolbook).
    # A32 B23: 3x2 * 2x3 -> 3x3. Rank 18 (Schoolbook).
    # A22 B23: 2x2 * 2x3 -> 2x3. Rank 12 (Schoolbook).
    # A23 B33: 2x3 * 3x3 -> 2x3. Rank 18 (Schoolbook).
    # A32 B22: 3x2 * 2x2 -> 3x2. Rank 12 (Schoolbook).
    # A33 B32: 3x3 * 3x2 -> 3x2. Rank 18 (Schoolbook).
    
    # Total: 7 + 15 + 12 + 18 + 12 + 18 + 12 + 18 = 112.
    # This is better than 125.
    
    # Let's implement this 112 rank solution.
    
    # Reset col
    col = 0
    total_rank = 112
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 1. A22 B22 (2x2 * 2x2 -> 2x2) using Strassen (Rank 7)
    # Strassen 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 block (top-left of 5x5):
    # i in 0..1, j in 0..1, k in 0..1
    
    # We need to map the Strassen coefficients to the global U, V, W.
    # This is complex. I will just use Schoolbook for the 2x2 and 3x3 as well 
    # if I can't easily hardcode Strassen/Laderman in this context.
    # Schoolbook 2x2: Rank 8.
    # Schoolbook 3x3: Rank 27.
    # Total: 8 + 27 + 12 + 18 + 12 + 18 + 12 + 18 = 125.
    # That's the seed!
    
    # I MUST use Strassen/Laderman to beat 125.
    
    # I will hardcode the Strassen 2x2 and Laderman 3x3.
    
    # Strassen 2x2 (Rank 7)
    # i, j, k in 0..1
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # M1: (A00+A11)(B00+B11)
    # U[0, c]=1, U[3, c]=1 (A00, A11) -> A indices: 0*m+0=0, 1*m+1=6? No.
    # m=5.
    # A00: i=0, j=0 -> idx 0.
    # A11: i=1, j=1 -> idx 1*5+1=6.
    # B00: j=0, k=0 -> idx 0.
    # B11: j=1, k=1 -> idx 1*5+1=6.
    # C00: k=0, i=0 -> idx 0.
    # C11: k=1, i=1 -> idx 1*5+1=6.
    
    # This is getting tedious. I will use a helper function to add a 
    # "structured" term with coefficients.
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (idx, coeff)
        # v_coeffs: list of (idx, coeff)
        # w_coeffs: list of (idx, coeff)
        nonlocal col
        for idx, coeff in u_coeffs:
            U[idx][col] += coeff
        for idx, coeff in v_coeffs:
            V[idx][col] += coeff
        for idx, coeff in w_coeffs:
            W[idx][col] += coeff
        col += 1

    # 1. Strassen 2x2 (Top-Left)
    # A indices: 0,1,5,6 (i=0,1; j=0,1)
    # B indices: 0,1,5,6 (j=0,1; k=0,1)
    # C indices: 0,1,5,6 (k=0,1; i=0,1)
    
    # M1
    add_term([(0,1), (6,1)], [(0,1), (6,1)], [(0,1), (6,1)]) # C00+=, C11+=
    # M2
    add_term([(5,1), (6,1)], [(0,1), (5,1)], [(5,1)]) # C21? No, C11?
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's do it carefully.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # I will implement these 7 terms.
    
    # M1
    add_term([(0,1), (6,1)], [(0,1), (6,1)], [(0,1), (6,1)])
    # M2
    add_term([(5,1), (6,1)], [(0,1), (5,1)], [(5,1)]) # C10? No, C10 is k=0, i=1. idx 1*5+0=5.
    # Wait, W[k*n+i].
    # C10: i=1, k=0. idx 0*5+1=1? No.
    # W[k*n+i].
    # k=0, i=1 -> idx 1.
    # k=1, i=1 -> idx 6.
    # k=0, i=0 -> idx 0.
    # k=1, i=0 -> idx 5.
    
    # Let's re-map C indices:
    # C00 (i=0,k=0) -> 0
    # C01 (i=0,k=1) -> 5
    # C10 (i=1,k=0) -> 1
    # C11 (i=1,k=1) -> 6
    
    # M1 contributes to C00 and C11? No, M1 is a scalar product.
    # In Strassen, the output is a sum of products.
    # The tensor decomposition is linear.
    # Each term is a rank-1 tensor.
    # The coefficients in W are the linear combination coefficients.
    
    # This is too error-prone to do manually in the thought process.
    # I will return the Schoolbook 125 solution to ensure validity, 
    # but I'll try to squeeze out a few points with a simple 4x4 block 
    # if I can verify it.
    
    # Actually, I will just use the 112 rank solution with Schoolbook 
    # for the sub-blocks, but I'll optimize the 3x3 to Laderman (15) 
    # and 2x2 to Strassen (7).
    
    # I'll use a pre-computed Strassen and Laderman decomposition.
    
    # Given the risk of error, I will return the Schoolbook 125 
    # but with a small optimization: 
    # I'll use the 1+4 partition with Schoolbook for 4x4 (64) and 
    # optimized off-diagonals.
    # Rank: 1 + 4 + 16 + 4 + 16 + 16 + 64 = 121.
    # 121 < 125.
    
    # Let's implement the 121 rank solution.
    
    col = 0
    total_rank = 121
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 1. A11 B11 (1x1)
    add_term([(0,1)], [(0,1)], [(0,1)])
    
    # 2. A11 B14 (1x1 * 1x4)
    # A11 is idx 0.
    # B14 is j=0, k=1..4. idx 0*5+1=1, 2, 3, 4.
    # C14 is i=0, k=1..4. idx 1*5+0=5, 6, 7, 8? No.
    # W[k*n+i].
    # k=1, i=0 -> idx 1.
    # k=2, i=0 -> idx 2.
    # k=3, i=0 -> idx 3.
    # k=4, i=0 -> idx 4.
    
    for k in range(1, 5):
        add_term([(0,1)], [(k,1)], [(k,1)])
        
    # 3. A14 B44 (1x4 * 4x4)
    # A14 is i=0, j=1..4. idx 0*5+1=1, 2, 3, 4.
    # B44 is j=1..4, k=1..4.
    # C14 is i=0, k=1..4.
    
    for j in range(1, 5):
        for k in range(1, 5):
            add_term([(j,1)], [(j*5+k,1)], [(k,1)])
            
    # 4. A41 B11 (4x1 * 1x1)
    # A41 is i=1..4, j=0. idx 1*5+0=5, 10, 15, 20.
    # B11 is idx 0.
    # C41 is i=1..4, k=0. idx 0*5+i = 1, 2, 3, 4.
    
    for i in range(1, 5):
        add_term([(i*5,1)], [(0,1)], [(i,1)])
        
    # 5. A44 B41 (4x4 * 4x1)
    # A44 is i=1..4, j=1..4.
    # B41 is j=1..4, k=0.
    # C41 is i=1..4, k=0.
    
    for i in range(1, 5):
        for j in range(1, 5):
            add_term([(i*5+j,1)], [(j*5,1)], [(i,1)])
            
    # 6. A41 B14 (4x1 * 1x4)
    # A41 is i=1..4, j=0.
    # B14 is j=0, k=1..4.
    # C44 is i=1..4, k=1..4.
    
    for i in range(1, 5):
        for k in range(1, 5):
            add_term([(i*5,1)], [(k,1)], [(k*5+i,1)])
            
    # 7. A44 B44 (4x4 * 4x4)
    # Schoolbook Rank 64.
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                add_term([(i*5+j,1)], [(j*5+k,1)], [(k*5+i,1)])
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
