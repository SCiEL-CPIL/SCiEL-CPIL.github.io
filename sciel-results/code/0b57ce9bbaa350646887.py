"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    vectors = []
    
    # Type 1: Vectors with two non-zero entries (±1, ±1)
    # These have norm sqrt(2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    # Type 2: Vectors with three non-zero entries (±1, ±1, ±1)
    # These have norm sqrt(3)
    # Need to check distance constraint: min distance >= max norm = sqrt(3)
    # Distance between two such vectors: |v1 - v2|^2
    # If they share k coordinates with same sign, (3-k) with opposite sign, and rest zero
    # |v1-v2|^2 = 2*(number of differing non-zero positions)
    # For distance >= sqrt(3), need |v1-v2|^2 >= 3
    # So need at least 2 differing positions (2*2=4 >= 3)
    # Actually let's be more careful. Two vectors with 3 non-zeros each.
    # Let a = number of positions where both have same non-zero value
    # Let b = number of positions where both have opposite non-zero values  
    # Let c = number of positions where one has non-zero, other has zero
    # a + b + c/2 = 3 (for each vector)
    # |v1-v2|^2 = 2b + c
    # We need 2b + c >= 3
    # Also a + b = 3 - c/2, so a = 3 - c/2 - b
    # Since a >= 0: 3 - c/2 - b >= 0, so b <= 3 - c/2
    # Try c=0: b <= 3, need 2b >= 3, so b >= 2. Possible: b=2,3
    #   b=2: a=1, two vectors share 1 same, 2 opposite
    #   b=3: a=0, three opposite
    # Try c=2: b <= 2, need 2b+2 >= 3, so b >= 1 (since 2b+2>=3 means b>=0.5)
    #   b=1: a=3-1-1=1
    #   b=2: a=3-1-2=0
    # Try c=4: b <= 1, need 2b+4 >= 3, always true
    #   b=0: a=3-2-0=1
    #   b=1: a=3-2-1=0
    # Try c=6: b <= 0, need 6 >= 3, true
    #   b=0: a=3-3-0=0
    # So we need to exclude pairs with 2b+c < 3, i.e., b=0,c=0 (impossible since a+b=3)
    # Actually the only bad case is when they're very similar.
    # Let me just generate and filter.
    
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    vectors.append(v)
    
    # Now filter: keep only vectors where min pairwise distance >= max norm
    if len(vectors) > 0:
        vecs = np.array(vectors, dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1)
        max_norm = np.max(norms)
        
        # Compute pairwise distances efficiently in chunks
        kept = []
        n = len(vecs)
        
        # Use a greedy approach: add vectors one by one if they satisfy constraint
        # with all previously kept vectors
        kept_indices = [0]  # Start with first vector
        kept_vecs = [vecs[0]]
        kept_norms = [norms[0]]
        
        for idx in range(1, n):
            if time.time() > deadline:
                break
            
            v = vecs[idx]
            norm_v = norms[idx]
            
            # Check against all kept vectors
            # Distance squared >= max(max_norm, norm_v)^2
            # Actually the constraint is: min pairwise distance >= max vector norm
            # So for any pair (i,j), ||vi - vj|| >= max(||vi||, ||vj||, ..., ||vk||)
            # This is a global constraint. Let me re-read.
            
            # "minimum pairwise distance is at least the maximum vector norm"
            # So: min_{i≠j} ||vi - vj|| >= max_i ||vi||
            
            # This means all pairwise distances must be >= the overall max norm.
            # Let M = max_i ||vi||
            # Need ||vi - vj|| >= M for all i≠j
            
            # When adding a new vector with norm norm_v, the new max norm becomes
            # M_new = max(M_old, norm_v)
            # Need ||v_new - v_old|| >= M_new for all old vectors
            
            # This is complex to maintain incrementally. Let me just do a full check
            # but it's O(n^2). For n ~ 264 + 176 = 440, that's ~100k checks, fine.
            
            # Actually let me just collect all candidates and do one final filter.
            pass
        
        # Simpler: just collect all, then filter
        # But 440 vectors, pairwise is 440*439/2 ≈ 96k, very fast
        pass
    
    # Just return the Type 1 vectors for now (known valid)
    # They have norm sqrt(2), and distances between them:
    # Two vectors (si,sj) at (i,j) and (sk,sl) at (k,l)
    # If {i,j} = {k,l}: distance = 2*sqrt(2) if opposite signs, 0 if same (but we have both signs)
    #   Actually for same pair {i,j}, we have (1,1),(1,-1),(-1,1),(-1,-1)
    #   Distances: (1,1)-(1,-1) = (0,2) norm=2
    #   (1,1)-(-1,1) = (2,0) norm=2
    #   (1,1)-(-1,-1) = (2,2) norm=2*sqrt(2)
    #   So min distance within same pair is 2
    # If {i,j} != {k,l} but share one index, e.g., (i,j) and (i,k):
    #   (si,sj,0...) and (sk,0,sl,...)
    #   Difference: (si-sk, sj, sl, ...)
    #   Norm^2 = (si-sk)^2 + sj^2 + sl^2
    #   si-sk is 0 or ±2, sj, sl are ±1
    #   Min when si=sk: norm^2 = 0 + 1 + 1 = 2, norm = sqrt(2)
    #   So min distance can be sqrt(2) which equals max norm sqrt(2). OK!
    # If disjoint: norm^2 = 1+1+1+1 = 4, norm = 2
    
    # So Type 1 alone gives 2*C(11,2)*4 = 2*55*4 = 440 vectors, all valid.
    
    # Can we add Type 2 vectors (norm sqrt(3))?
    # Max norm would become sqrt(3)
    # Need all pairwise distances >= sqrt(3)
    # Type 1 vectors have norm sqrt(2) < sqrt(3), so they're fine as long as
    # distance to Type 2 vectors >= sqrt(3)
    
    # Distance between Type 1 (at positions i,j with values si,sj) and Type 2 (at i,j,k with si',sj',sk')
    # If they share positions i,j:
    #   Difference at i: si - si', at j: sj - sj', at k: 0 - sk'
    #   Norm^2 = (si-si')^2 + (sj-sj')^2 + sk'^2
    #   si-si' is 0 or ±2, similarly for sj
    #   Min: si=si', sj=sj' gives norm^2 = 0+0+1 = 1 < 3. BAD!
    #   So Type 1 and Type 2 vectors that share both non-zero positions with same signs are too close.
    
    # This makes it hard to combine. Let me stick with Type 1 for now and see if we can
    # find a better pure construction.
    
    # Alternative: use vectors with entries in {-1, 0, 1} more cleverly.
    # The seed gives 440. Let me try to add more by being smarter.
    
    # Actually, let me just return the seed result for safety, since the inspiration
    # with score 1.1455 suggests there's room for improvement but I need to be careful.
    
    # Let me try: all vectors with exactly 2 non-zero entries of value ±1,
    # PLUS all vectors with exactly 1 non-zero entry of value ±2.
    # Norm of type 1: sqrt(2), type 2: 2
    # Max norm = 2
    # Need all distances >= 2
    # Type 1 to Type 1: min distance sqrt(2) as computed above. sqrt(2) < 2. BAD.
    
    # So I can't mix norms easily. Let me just go with the seed.
    
    result = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                result.append(v)
    
    return np.array(result, dtype=np.int64)
# EVOLVE-BLOCK-END
