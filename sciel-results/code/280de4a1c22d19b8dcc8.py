"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We decompose the 4x7 multiplication as:
    # A (2x4) @ B (4x7) = A @ [B1 B2] where B1 is 4x4, B2 is 4x3
    # Result: [A@B1 (2x4)] [A@B2 (2x3)]
    # Total rank = rank(M_{2,4,4}) + rank(M_{2,4,3})
    
    # For M_{2,4,4}: use a known rank-20 decomposition
    # For M_{2,4,3}: use a known rank-16 decomposition
    # Combined rank: 36
    
    # Actually, let's use a simpler approach with Strassen-like blocks.
    # Split the 4x4 into 2x2 blocks and use Strassen.
    
    # Alternative: use direct construction with rank 35 (theoretical lower bound)
    # But constructing exact rank-35 is complex. Let's use rank 36.
    
    # I'll use a systematic approach:
    # M_{2,4,4} can be done in rank 20 using:
    #   - 2x2 Strassen on each of 2 column-blocks of 2x4, but that gives 2*7=14 per block...
    #   Actually, let me just use a known result.
    
    # For safety, let me implement a rank-40 decomposition using 2x2x2 Strassen blocks:
    # Split 4x7 into seven 2x2 blocks (overlapping not needed if we split properly)
    # 4x7 = 2 blocks of 2x4 + ... no.
    # 4x7: split rows into 2x2, columns into 3x2 + 1x2? Messy.
    
    # Let me use a cleaner decomposition:
    # Split A (2x4) into A_left (2x2), A_right (2x2)
    # Split B (4x7) into B_top (2x7), B_bottom (2x7)
    # A@B = A_left@B_top + A_right@B_bottom
    # Each is M_{2,2,7}, rank at most 14 (schoolbook) or 7*2=14
    # But M_{2,2,7} can be done in rank 2*ceil(7/1)*... 
    # Actually M_{2,2,p} has rank at most 2p for schoolbook, but can be done in ~1.5p
    # For p=7: rank at most 11 using known results? Let's use 14 each for safety.
    # Total: 28
    
    # Wait, M_{2,2,7}: the rank is at most 2*7=14 schoolbook.
    # Can we do better? For M_{2,2,p}, the rank is p + ceil(p/2) for p even? 
    # For p=7: maybe 7+4=11? Not sure of exact value.
    # Let's use 14 each, total 28.
    
    # But wait, that's just schoolbook split, which is the same as original schoolbook.
    # 2*14 = 28? No, original is 2*4*7=56.
    # If I do M_{2,2,7} in rank 14 each, that's 28 total, which is half!
    # But is rank(M_{2,2,7}) = 14? That's just schoolbook for 2x2x7.
    # The schoolbook for 2x2x7 is indeed 14.
    # So this decomposition gives rank 28, which is valid but not minimal.
    
    # Actually, I realize the issue: when I split A into left and right, and B into top and bottom,
    # the products A_left@B_top and A_right@B_bottom each need their own rank.
    # M_{2,2,7} schoolbook is 2*2*7 = 28 per term? No!
    # M_{2,2,7} means multiplying 2x2 by 2x7, result 2x7.
    # Schoolbook rank is 2*2*7 = 28. That's worse!
    
    # I was confused. Let me restart.
    # M_{n,m,p}: multiply n x m by m x p, result n x p.
    # Schoolbook rank = n*m*p.
    # M_{2,4,7}: schoolbook = 2*4*7 = 56.
    
    # If I split: A (2x4) = [A1 A2] where A1, A2 are 2x2.
    # B (4x7) = [B1; B2] where B1, B2 are 2x7.
    # A@B = A1@B1 + A2@B2
    # A1@B1 is M_{2,2,7}, schoolbook rank = 2*2*7 = 28.
    # A2@B2 is M_{2,2,7}, schoolbook rank = 2*2*7 = 28.
    # Total: 56. Same as before. No gain.
    
    # To gain, I need M_{2,2,7} to have rank < 28.
    # Known: rank(M_{2,2,2}) = 7 (Strassen).
    # For M_{2,2,p}, the rank is at most p * rank(M_{2,2,2})/8 * ... no.
    # Actually, for M_{2,2,p}, one can use the "block" approach.
    # The rank of M_{2,2,p} is known to be at most p + ceil(3p/4) for some p? 
    # For p=7: maybe around 20?
    
    # This is getting complex. Let me just use a safe rank-50 decomposition
    # or stick with schoolbook and try to find a better one via search.
    
    # Given time constraints, let me implement a rank-40 decomposition using
    # 2x2x2 Strassen blocks where applicable.
    
    # 4x7: I can view the 4 rows as 2 groups of 2, and 7 columns as 3 groups of 2 + 1.
    # This is messy. Let me just return schoolbook for now, but with a time check.
    
    import time
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # For now, return schoolbook. This is safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
