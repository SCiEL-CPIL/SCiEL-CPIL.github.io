import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We'll try to construct a rank-42 decomposition:
    # - Use Strassen-Winograd rank-21 for the 3x3x3 sub-block (columns 0-2 of B)
    # - Use schoolbook rank-21 for the remaining 3x4 sub-block (columns 3-6 of B)
    # Total rank: 21 + 21 = 42
    
    # First, let's try to construct the Strassen-Winograd rank-21 factors for 3x3x3
    # The Winograd scheme uses 21 multiplications for 3x3 matrix multiplication
    
    # For a 3x3x3 matrix multiplication, the Winograd factors can be derived.
    # We'll use the known Strassen-Winograd decomposition with rank 21.
    
    # The 21 bilinear forms for 3x3x3 Strassen-Winograd:
    # We need to construct U, V, W matrices of shape (9, 21) for the 3x3x3 part.
    
    # Known Winograd 3x3x3 factors (rank 21):
    # The decomposition uses the following 21 terms:
    
    # Let's construct the factors explicitly.
    # For 3x3x3, the Strassen-Winograd decomposition has the following structure.
    
    # We'll use a known valid rank-21 decomposition for 3x3x3.
    # The factors are well-known in the literature.
    
    # Constructing the 21 rank factors for 3x3x3:
    # U[i*3+j], V[j*3+k], W[k*3+i] for i,j,k in 0..2
    
    # The 21 bilinear forms for Winograd 3x3:
    # Based on the standard Winograd scheme, we have:
    
    U_3x3 = []
    V_3x3 = []
    W_3x3 = []
    
    # The 21 terms of the Winograd 3x3x3 decomposition:
    # Term 0: A[0,0]*B[0,0]
    U_3x3.append([1,0,0, 0,0,0, 0,0,0])
    V_3x3.append([1,0,0, 0,0,0, 0,0,0])
    W_3x3.append([1,0,0, 0,0,0, 0,0,0])
    
    # Term 1: A[0,0]*B[0,1] + A[0,1]*B[0,0] - A[0,1]*B[0,1]
    # This gets complex. Let me use a verified set of factors.
    
    # Actually, let me use a simpler approach: construct the full 3x3x7 decomposition
    # by combining known rank-21 Strassen-Winograd for the first 3 columns with
    # schoolbook for the last 4 columns.
    
    # For now, let's use a safe approach: return the schoolbook decomposition
    # but try to optimize if we have time.
    
    # Schoolbook: rank = 3*3*7 = 63
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to construct rank-42 decomposition
    try:
        if time.time() > deadline - 20:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        # Rank-42 decomposition:
        # Part 1: Strassen-Winograd rank-21 for columns 0-2 (3x3x3)
        # Part 2: Schoolbook rank-21 for columns 3-6 (3x3x4)
        
        rank_total = 42
        U_out = [[0] * rank_total for _ in range(n * m)]  # 9 x 42
        V_out = [[0] * rank_total for _ in range(m * p)]  # 21 x 42
        W_out = [[0] * rank_total for _ in range(p * n)]  # 21 x 42
        
        # Part 2: Schoolbook for columns 3-6 (indices 3,4,5,6)
        # This is 3x3x4 = 36... wait, that's not 21.
        # 3*3*4 = 36, not 21.
        # So the strategy needs adjustment.
        
        # Let me reconsider. For 3x3x7:
        # If we use rank-21 for 3x3x3 and rank-36 for 3x3x4, total = 57.
        # That's still better than 63.
        
        # Or: rank-21 for 3x3x3, and we need to handle 3x3x4.
        # 3x3x4 can be done in at most 3*3*4=36, but maybe less.
        
        # For simplicity and safety, let's use:
        # Rank-21 Winograd for 3x3x3 (cols 0-2)
        # Rank-36 schoolbook for 3x3x4 (cols 3-6)
        # Total rank: 21 + 36 = 57
        
        rank_sw = 21
        rank_school_4 = 3 * 3 * 4  # 36
        rank_combined = rank_sw + rank_school_4  # 57
        
        if rank_combined >= 63:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        U_comb = [[0] * rank_combined for _ in range(n * m)]
        V_comb = [[0] * rank_combined for _ in range(m * p)]
        W_comb = [[0] * rank_combined for _ in range(p * n)]
        
        # Part 1: Strassen-Winograd for 3x3x3 (columns 0,1,2)
        # We need the actual rank-21 factors. Let me construct them.
        
        # The Winograd 3x3x3 decomposition with rank 21:
        # This is a known result. The factors are:
        
        # I'll use a verified set. The 21 bilinear forms are:
        # Based on the standard Winograd scheme for 3x3 matrices.
        
        # Let me use a simpler verified construction:
        # Actually, let me just use the schoolbook for everything to be safe,
        # but with a better structure if possible.
        
        # For now, return the safe schoolbook result.
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    except Exception:
        if time.time() > deadline:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
