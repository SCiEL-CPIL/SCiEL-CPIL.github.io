"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition search."""

def construct():
    n, m, p = 4, 5, 5
    
    # Build a rank-69 decomposition using:
    #  - Strassen rank-49 decomposition for the 4x4x4 core (i in 0..3, j in 0..3, k in 0..3)
    #  - Trivial outer-product terms for the remaining slices:
    #      j=4, k in 0..3 : 4*4 = 16 terms
    #      k=4, j in 0..3 : 4*4 = 16 terms
    #      i=4, j in 0..3, k in 0..3 : 4*4 = 16 terms
    #      i=4, j=4, k in 0..3 : 4 terms
    #      i=4, k=4, j in 0..3 : 4 terms
    #      i=4, j=4, k=4 : 1 term
    # Total extra = 16+16+16+4+4+1 = 57 ... wait that's 57, but 49+57=106, not 69.
    # Let me recount: the 4x4x4 core covers i in 0..3, j in 0..3, k in 0..3 (64 entries).
    # Remaining entries in the full 5x5x5 tensor (125 entries total):
    #   j=4: i in 0..4, k in 0..4 -> 5*5=25
    #   k=4, j in 0..3: i in 0..4, j in 0..3 -> 5*4=20
    #   i=4, j in 0..3, k in 0..3: 4*4=16
    #   But we've double-counted some. Let's just enumerate all (i,j,k) not in core:
    #   Total = 125 - 64 = 61 remaining entries.
    #   Using trivial for each gives 61 terms, so 49+61=110. Still not 69.
    #
    # Actually, let me just use the trivial rank-100 decomposition as a safe fallback.
    # The random search in the original was ineffective. Let me try a smarter approach:
    # Use the known Strassen decomposition for 4x4x4 (rank 49) and handle the rest.
    #
    # For now, return the trivial schoolbook decomposition which is guaranteed correct.
    rank_trivial = n * m * p  # 100
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
