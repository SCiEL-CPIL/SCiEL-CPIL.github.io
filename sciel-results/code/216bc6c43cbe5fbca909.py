"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    # Rank-40 decomposition for 3x3 * 3x5 matrix multiplication
    # Using a structured construction based on embedding and correction terms
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 9 x 40
    V = [[0] * rank for _ in range(m * p)]  # 15 x 40
    W = [[0] * rank for _ in range(p * n)]  # 15 x 40
    
    # We use the following construction:
    # Split the 5 columns into 3 + 2
    # For the 3x3 part: use a rank-23 Strassen-like decomposition
    # For the remaining 2 columns: use rank-17 terms
    # Total: 23 + 17 = 40
    
    # Actually, let me use a simpler explicit construction.
    # Term structure:
    # Terms 0-17: Handle columns 0,1,2 with a 3x3x3 decomposition (rank 18 standard, but we can do better)
    # Terms 18-39: Handle columns 3,4 with additional terms
    
    # Let me use a clean rank-40 construction based on the following:
    # For k in 0..4, i in 0..2, j in 0..2, we need C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Construction:
    # Phase 1 (terms 0-26): 3x3x3 for k=0,1,2 using 27 standard terms but merge some
    # Phase 2 (terms 27-39): handle k=3,4 with 13 terms
    #
    # This is getting complex. Let me use a verified rank-40 construction.
    
    # Explicit rank-40 construction:
    # Use the fact that we can reuse some Strassen terms.
    
    # I'll use a known rank-40 construction from literature.
    # The construction uses 40 terms with specific (i,j,k) patterns.
    
    terms = []
    
    # Generate 40 terms based on a structured pattern
    # For each term t in 0..39:
    #   Determine which (i,j,k) contributions are nonzero
    
    # Use a grid-based construction:
    # Terms 0-29: cover a 3x3x3 grid with some merging (30 terms, but we need 30)
    # Terms 30-39: cover the remaining 2 columns
    
    # Let me just use a straightforward but optimized construction:
    # 30 terms for 3x3x3 (standard would be 27, so this is worse)
    # That won't work.
    
    # New approach: use the embedding (3,3,5) into (4,4,4) which has rank 25 (Strassen-like)
    # But 4x4x4 rank is 25, and we can restrict to our sub-tensor.
    # However, the restriction may not be exact.
    
    # Let me use a different strategy: direct enumeration-free construction.
    
    # Known result: rank(3,3,5) <= 40
    # Construction:
    # - 18 terms for a 3x3x3 block (using 18-term 3x3x3 decomposition, which exists)
    # - 22 terms for the remaining 2 columns of the 3x5 output
    
    # Actually, 3x3x3 has rank at most 23 (Strassen), not 18.
    # Let me use: rank(3,3,3)=23, and for the extra 2 columns, add 17 terms.
    # 23 + 17 = 40.
    
    # But we need to be careful about overlaps.
    
    # Let me use a simpler valid construction: rank 40 via explicit formula.
    
    # For term r = 0, ..., 39:
    #   a = r % 3
    #   b = (r // 3) % 3  
    #   c = (r // 9) % 5
    #   This gives 45 terms for full coverage.
    #   
    # To get 40, we need to merge 5 terms.
    
    # I'll use a rank-40 construction based on the following insight:
    # The last column (k=4) can be computed using combinations of other terms.
    
    # For now, let me return a valid rank-45 decomposition (schoolbook)
    # but structured differently, and note that a better one exists.
    # Actually, let me try to implement a rank-40 construction.
    
    # Rank-40 construction for (3,3,5):
    # Use 20 terms for columns 0-2 (a 3x3x3 decomposition of rank 20 doesn't exist; min is 23)
    # So that won't work either.
    
    # Let me just return the schoolbook rank-45 for now, but with a cleaner implementation.
    # The task says "rank-minimal" and the score is the rank, so lower is better.
    # If I can't find a provably correct rank < 45, I should return 45.
    
    # However, I recall that rank(3,3,5) is indeed 40 or less.
    # Let me implement a rank-40 construction using a known method.
    
    # Method: Use the "lattice" method or "Kronecker" embedding.
    
    # I'll implement a rank-40 decomposition based on the following explicit construction:
    # Terms are indexed by (a, b, c) where a in {0,1,2}, b in {0,1,2}, c in {0,1,2,3,4}
    # but with some terms merged.
    
    # For safety, let me return a valid rank-45 decomposition.
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
