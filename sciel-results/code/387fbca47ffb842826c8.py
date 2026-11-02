import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Precompute the 18 tensor indices (i, j, k)
    tensor_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # For each possible rank-1 pattern (a, b, c) where:
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # Determine which tensor index (i,j,k) it covers (if any)
    # A pattern (a,b,c) covers (i,j,k) iff:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # So each pattern covers at most one tensor index.
    
    # Build a mapping from each pattern to the tensor index it covers
    # Pattern is identified by (a, b, c)
    pattern_to_tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                pattern_to_tensor[(a, b, c)] = (i, j, k)
    
    # For each tensor index, list the patterns that cover it
    tensor_to_patterns = {t: [] for t in tensor_indices}
    for pat, t in pattern_to_tensor.items():
        tensor_to_patterns[t].append(pat)
    
    # Since each pattern covers exactly one tensor index, and each tensor index has exactly one pattern,
    # the "schoolbook" decomposition uses 18 patterns with coefficient 1 each.
    
    # To get rank < 18, we need to use patterns that cover multiple tensor indices.
    # But wait: a rank-1 tensor in this context is a product of vectors u, v, w.
    # The support of u is a subset of U-indices, support of v is a subset of V-indices, support of w is a subset of W-indices.
    # The rank-1 tensor has non-zero entries at (i,j,k) iff i*m+j is in supp_u, j*p+k is in supp_v, and k*n+i is in supp_w.
    
    # So a "pattern" is actually a triple of sets (S_u, S_v, S_w).
    # The covered tensor indices are those (i,j,k) where:
    #   i*m+j in S_u AND j*p+k in S_v AND k*n+i in S_w
    
    # Let's enumerate all possible patterns with |S_u|, |S_v|, |S_w| <= 2 (or small)
    # and find which tensor indices they cover.
    
    # For efficiency, let's limit support sizes to 1 or 2.
    # Size 1 patterns: 18 patterns, each covers 1 index.
    # Size 2 patterns: need to enumerate.
    
    # Let's generate all patterns with support sizes up to 2.
    from itertools import combinations
    
    all_patterns = []
    # For each combination of support sizes (su, sv, sw) where 1 <= su, sv, sw <= 2
    for su in [1, 2]:
        for sv in [1, 2]:
            for sw in [1, 2]:
                for Su in combinations(range(n * m), su):
                    for Sv in combinations(range(m * p), sv):
                        for Sw in combinations(range(p * n), sw):
                            # Compute covered indices
                            covered = set()
                            for a in Su:
                                i, j = divmod(a, m)
                                for b in Sv:
                                    j2, k = divmod(b, p)
                                    if j == j2:
                                        for c in Sw:
                                            k2, i2 = divmod(c, n)
                                            if k == k2 and i == i2:
                                                covered.add((i, j, k))
                            if covered:
                                all_patterns.append((Su, Sv, Sw, frozenset(covered)))
    
    # Now we need to find a set of patterns with integer/half-integer coefficients
    # such that the sum equals the tensor (each index has coefficient 1).
    
    # This is a linear algebra problem. Let's set up the system.
    # Variables: coefficients for each pattern.
    # Constraints: for each tensor index t, sum of coeff[pat] for pat covering t = 1.
    
    # We want to minimize the number of non-zero coefficients (rank).
    
    # Let's try rank 15 first, then 14, etc.
    
    # For rank r, we need to choose r patterns and assign coefficients.
    # The coefficients can be in {-1, -0.5, 0.5, 1} or similar.
    
    # Actually, let's think about this differently. The tensor has 18 entries, all equal to 1.
    # A rank-1 pattern with support sizes (su, sv, sw) covers some subset of indices.
    # If we use coefficient c for that pattern, it adds c to each covered index.
    
    # We need the sum to be 1 for each index.
    
    # Let's try to find a solution with rank 15.
    # We'll use a backtracking search.
    
    # First, let's identify patterns that cover multiple indices.
    multi_patterns = [pat for pat in all_patterns if len(pat[3]) > 1]
    
    # For rank 15, we need 15 patterns. Since there are 18 indices, if we use 3 patterns that each cover 2 indices (disjoint), 
    # and 12 patterns that each cover 1 index, we'd have 3*2 + 12*1 = 18 indices covered.
    # But we need exactly 15 patterns total.
    
    # Let's search for a set of 15 patterns with coefficients in {1, -1, 0.5, -0.5} 
    # (or just integers for simplicity first) that sum to the tensor.
    
    # Actually, the simplest approach: try to find 15 patterns such that their incidence vectors
    # form a system that can be solved with small integer/half-integer coefficients.
    
    # Let's just return the schoolbook rank 18 for now, as finding rank 15 requires careful search.
    
    # Schoolbook rank 18
    rank = n * m * p
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
