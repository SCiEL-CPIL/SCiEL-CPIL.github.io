import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Block decomposition:
    # Block 1: cols 0-2 (3x3x3) -> rank 23 Winograd
    # Block 2: cols 3-5 (3x3x3) -> rank 23 Winograd  
    # Block 3: cols 6-7 (3x3x2) -> rank 18 schoolbook
    # Total: 23 + 23 + 18 = 64
    
    total_rank = 64
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Winograd 3x3x3 rank-23 decomposition
    # U indices: i*m+j (i,j in 0..2)
    # V indices: j*p+k where p=3 for this block (j,k in 0..2)
    # W indices: k*n+i (k,i in 0..2)
    
    # Known rank-23 decomposition for 3x3x3
    # Using the classical Winograd scheme with 23 multiplications
    # Each term: (u_vec, v_vec, w_vec) of length 9
    
    winograd_terms = [
        # Term 0: identity-like terms for diagonal
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Terms for off-diagonal with +1/-1 patterns
        ([1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,1,0, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 1,0,0, 0,1,0], [0,0,0, 0,0,1, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # More terms to reach rank 23
        ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Additional terms
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Verify and fix: use a proper known decomposition
    # Let me use a verified rank-23 set from literature
    
    # Actually, let me use a simpler verified approach:
    # Use the known fact that we can achieve rank 23 for 3x3x3
    # I'll use a construction based on 2x2x2 Strassen blocks
    
    # For safety, let me use a verified rank-24 decomposition
    # which is easier to construct correctly
    
    # Reset and use a cleaner approach
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Use verified Winograd 3x3x3 rank-23
    # Based on the standard Winograd algorithm
    
    # The 23 terms for 3x3x3:
    # Using the (1,1,1) and (1,1,-1) type patterns
    
    # Let me use a different strategy: construct via 2x2x2 Strassen + corrections
    # 3x3x3 can be decomposed as 2x2x2 (rank 7) + boundary terms
    
    # Simpler: use a hardcoded verified decomposition
    
    # I'll use the following verified rank-23 decomposition:
    # Source: Bini, Capovani, Lotti, Romani (1979)
    
    w23_U = [
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
    ]
    
    # This is getting too complex to hardcode correctly without verification.
    # Let me use a safer approach: use rank 27 for 3x3x3 but try to merge
    # with the 3x3x2 block.
    
    # Actually, let me just use schoolbook for everything = rank 72,
    # but that's not an improvement.
    
    # Let me try: use Strassen-style for the 3x3x3 blocks.
    # 3x3 can be padded to 4x4 and use Strassen? No, that's worse.
    
    # Best safe option: use a verified rank-24 decomposition for 3x3x3
    # which is easier to construct than rank-23.
    
    # For now, return schoolbook to ensure validity
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
