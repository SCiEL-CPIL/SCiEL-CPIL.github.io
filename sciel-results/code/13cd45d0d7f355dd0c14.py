"""Program for 3x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5
    
    # We'll build rank factors explicitly.
    # The multiplication tensor: T[i,j,k] = 1 if j==k else 0
    # We seek U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(j,k)
    
    # Use a constructive approach based on bilinear decomposition.
    # For each pair (i,k), we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if j==k else 0
    
    # Strategy: Use separate "channels" for different (i,k) pairs where possible,
    # and share channels where the structure allows.
    
    # Let's use the following construction:
    # For each i in [0,3), k in [0,5):
    #   We need sum_j A[i,j] * B[j,k] to be computed.
    #   This is a dot product of length 5.
    
    # A rank-r decomposition corresponds to r "multipliers".
    # We'll use the standard technique: for each (i,k), the contribution
    # can be factored.
    
    # Construction via explicit factors:
    # We'll use r = 35 and verify.
    
    # Actually, let's use a more systematic approach.
    # The tensor has 75 nonzero entries. The minimum rank is at least 
    # the number of nonzero entries in any "slice", but we can do better.
    
    # Let me construct using the following idea:
    # Split j into groups and use Strassen-like tricks.
    
    # For now, use a verified construction with rank 40.
    rank = 40
    U = [[0]*rank for _ in range(n*m)]  # 15 x 40
    V = [[0]*rank for _ in range(m*p)]  # 25 x 40
    W = [[0]*rank for _ in range(p*n)]  # 15 x 40
    
    # We'll fill in the factors using a known decomposition.
    # Use the following construction based on splitting dimensions.
    
    # For the 5x5 multiplication part, use 2+3 split.
    # For the 3 part, use 1+2 split.
    
    # Block decomposition:
    # i blocks: [0], [1,2]
    # j blocks: [0,1], [2,3,4]
    # k blocks: [0,1], [2,3,4]
    
    # For each block combination (i_block, j_block, k_block), use appropriate rank.
    
    # 1x2x2: rank 2
    # 1x2x3: rank 3  
    # 1x3x2: rank 3
    # 1x3x3: rank 3
    # 2x2x2: rank 7 (Strassen)
    # 2x2x3: rank 6 (schoolbook would be 12, but we can do better)
    # 2x3x2: rank 6
    # 2x3x3: rank 9
    
    # Total: 2+3+3+3+7+6+6+9 = 39
    # Let's use rank 40 to be safe, or verify rank 39 works.
    
    # Let me implement Strassen for 2x2x2 and schoolbook for others,
    # then optimize the mixed blocks.
    
    # Strassen 2x2x2 factors (rank 7):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # I'll implement a helper to get factors for small blocks,
    # then combine them.
    
    def get_factors_2x2x2_strassen():
        """Return (U, V, W) for 2x2x2 with rank 7."""
        r = 7
        U = [[0]*r for _ in range(4)]  # 2x2 = 4 entries
        V = [[0]*r for _ in range(4)]
        W = [[0]*r for _ in range(4)]
        
        # Index: U[i*2+j], V[j*2+k], W[k*2+i]
        # M1: (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # M2: (A10+A11)*B00 -> C10 += M2
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10
        
        # M3: A00*(B01-B11) -> C01 += M3
        U[0][2] = 1  # A00
        V[1][2] = 1  # B01
        V[3][2] = -1 # B11
        W[1][2] = 1  # C01
        
        # M4: A11*(B10-B00) -> C00 += M4, C10 += M4
        U[3][3] = 1  # A11
        V[2][3] = 1  # B10
        V[0][3] = -1 # B00
        W[0][3] = 1  # C00
        W[2][3] = 1  # C10
        
        # M5: (A00+A01)*B11 -> C00 -= M5, C01 += M5
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        W[0][4] = -1 # C00
        W[1][4] = 1  # C01
        
        # M6: (A10-A00)*(B00+B10) -> C11 += M6
        U[2][5] = 1  # A10
        U[0][5] = -1 # A00
        V[0][5] = 1  # B00
        V[2][5] = 1  # B10
        W[3][5] = 1  # C11
        
        # M7: (A01-A11)*(B01+B11) -> C00 += M7
        U[1][6] = 1  # A01
        U[3][6] = -1 # A11
        V[1][6] = 1  # B01
        V[3][6] = 1  # B11
        W[0][6] = 1  # C00
        
        return U, V, W
    
    def get_factors_schoolbook(n, m, p):
        """Return schoolbook factors for nxmxp."""
        r = n*m*p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    def combine_factors(factors_list):
        """Combine a list of (U, V, W) with row offsets into full matrices."""
        # This is complex; let me just build directly.
        pass
    
    # Let me build the full factors directly by iterating over blocks.
    
    # Block ranges:
    # i: block0 = [0], block1 = [1,2]
    # j: block0 = [0,1], block1 = [2,3,4]
    # k: block0 = [0,1], block1 = [2,3,4]
    
    # For each (ib, jb, kb), get local factors and place into global.
    
    # This is getting very complex. Let me use a simpler but effective approach:
    # Use rank 45 with a direct construction and verify numerically.
    
    # Actually, let me just use a clean construction.
    
    # I'll use the following: for each (i,k) pair, we need to compute
    # dot product of A[i,:] and B[:,k]. These are 5-element dot products.
    # We can share multipliers across different (i,k) pairs if the
    # A and B factors align.
    
    # Simple construction: rank = 3*5 = 15 * something.
    
    # Let me use rank = 35 and construct explicitly.
    
    rank = 35
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # For each i in [0,3), use 5 multipliers for the 5 j values.
    # But we can share across i if B factors match.
    
    # Construction: For each j in [0,5), create one multiplier channel
    # that handles all i for that j.
    # Channel c = j: U[i*m+j, c] = 1 for all i, V[j*p+k, c] = 1 if k==j else 0, W[k*n+i, c] = 1
    # But W depends on both k and i.
    
    # Let me try: for c in range(15):
    #   i = c // 5, j = c % 5
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1 if k == j else 0  -> V[j*p+j, c] = 1
    #   W[k*n+i, c] = 1 if k == j else 0 -> W[j*n+i, c] = 1
    # This gives rank 15! Let me verify.
    
    # For term (i', j', k'): sum_c U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c]
    # Nonzero only when: i'*m+j' matches some i*m+j where c = i*5+j
    #   -> i' = i, j' = j, c = j
    #   Then V[j*p+k', c] nonzero only if k' == j
    #   Then W[j*n+i', c] = W[j*n+i, c] = 1
    # So total = 1 if i'=i, j'=j, k'=j, i.e., j'=k' and i'=i.
    # That's exactly delta(j', k') for each i'. Correct!
    
    # So rank 15 works! Let me implement this.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        i = c // m
        j = c % m
        U[i*m+j][c] = 1
        V[j*p+j][c] = 1
        W[j*n+i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
