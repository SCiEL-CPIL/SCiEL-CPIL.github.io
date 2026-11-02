"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)


def construct():
    """
    Construct a kissing configuration in R^11.
    Strategy:
      1. Generate a pool of candidate integer vectors with small norms.
      2. Use symmetry breaking (first nonzero coordinate positive) to reduce candidates.
      3. Greedily select vectors such that min pairwise distance >= max norm among selected.
    
    The key constraint: for selected vectors v_i, we need ||v_i - v_j|| >= max(||v_i||) for all i != j.
    Since all vectors are normalized to norm 2 for the kissing number, the actual constraint in the
    integer lattice is: ||v_i - v_j||^2 >= max_i ||v_i||^2.
    
    If we restrict to vectors of equal norm R, then we need ||v_i - v_j||^2 >= R^2 for all pairs.
    This means v_i . v_j <= (R^2 - R^2)/2 ... wait:
      ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j = 2R^2 - 2 v_i.v_j >= R^2
      => v_i.v_j <= R^2/2
    
    So we need vectors of equal norm R with pairwise dot product <= R^2/2.
    
    Let's use weight-2 vectors (two nonzero entries of magnitude 1), so R^2 = 2.
    Then we need v_i . v_j <= 1.
    Two weight-2 vectors: if they share one coordinate, dot product can be -1, 0, or 1.
    If they share no coordinates, dot product is 0.
    
    The seed program uses all weight-2 vectors. Let's count: C(11,2)*4 = 55*4 = 220 vectors.
    But with symmetry breaking (first nonzero positive), we get half: 110 vectors.
    
    However, the constraint is pairwise dot <= 1. For weight-2 vectors with R^2=2:
    - If two vectors share exactly one coordinate: dot product is ±1 (if same sign) or -1 (if opposite).
      Actually if v_i has entries at positions a,b and v_j at positions a,c:
      v_i . v_j = v_i[a]*v_j[a] which is ±1.
      We need this <= 1, which is always true.
    - If they share both coordinates: v_i = v_j or v_i = -v_j.
      v_i = v_j: dot = 2 > 1, NOT allowed.
      v_i = -v_j: dot = -2 <= 1, allowed.
    - If they share no coordinates: dot = 0 <= 1, allowed.
    
    So the only forbidden pairs are identical vectors. With symmetry breaking (first nonzero positive),
    no two distinct vectors in our pool are identical, so ALL pairs are valid!
    
    Wait, let me re-examine. The seed already does this. Let me think about what can be improved.
    
    Actually, the constraint is: min pairwise distance >= max norm.
    If all vectors have norm R, then min pairwise distance >= R means ||v_i - v_j|| >= R for all i≠j.
    
    For the seed: all vectors have norm sqrt(2), and we need ||v_i - v_j|| >= sqrt(2).
    ||v_i - v_j||^2 >= 2.
    
    For two weight-2 vectors:
    - Same positions, same signs: ||v_i - v_j||^2 = 0 < 2. FORBIDDEN.
    - Same positions, opposite signs: ||v_i - v_j||^2 = 4 >= 2. OK.
    - Share one position: ||v_i - v_j||^2 = 2 + 1 + 1 - 2*(±1) = 4 ∓ 2. So 2 or 4. Both >= 2. OK.
    - No shared positions: ||v_i - v_j||^2 = 2 + 2 = 4 >= 2. OK.
    
    So with symmetry breaking, we avoid the "same positions, same signs" case, and all pairs work.
    The seed gets 110 vectors.
    
    Can we do better? Let's try mixing different weights.
    
    Actually, let's think about the kissing number in dimension 11. The known kissing number k(11) is between 504 and 505... actually it's not known exactly, but bounds are known. The seed gives 110 which is quite low.
    
    Let me try a different approach: use vectors of norm 2 (weight 4 with entries ±1, or weight 2 with entries ±√2... no, integer vectors).
    
    Weight 4 vectors with entries ±1: norm^2 = 4. Need ||v_i - v_j||^2 >= 4.
    v_i . v_j <= (4+4-4)/2 = 2.
    
    Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280. With symmetry breaking: 2640.
    
    For two weight-4 vectors with k overlapping positions:
    dot product = sum over overlapping positions of v_i[p]*v_j[p], each term ±1.
    Max dot = k (all same sign).
    We need dot <= 2, so k <= 2 if all signs match, or we need careful sign choices.
    
    This is getting complex. Let me try a greedy approach with a mixed pool.
    
    Alternative: use the known construction from the E8 lattice or similar.
    
    Actually, let me try a simpler improvement: include weight-1 vectors (unit vectors) along with weight-2 vectors.
    
    Weight-1: norm^2 = 1. Weight-2: norm^2 = 2.
    Max norm^2 = 2.
    Need ||v_i - v_j||^2 >= 2 for all pairs.
    
    - Two weight-1 vectors at different positions: ||v_i - v_j||^2 = 1+1 = 2. OK.
    - Two weight-1 vectors at same position: ||v_i - v_j||^2 = 0 or 4. 0 is forbidden.
      With symmetry breaking (first nonzero positive), we only have +e_i, so no conflicts.
    - Weight-1 at pos a, weight-2 at positions {b,c} (b≠a, c≠a): ||v_i - v_j||^2 = 1+2 = 3 >= 2. OK.
    - Weight-1 at pos a, weight-2 at positions {a,b}: ||v_i - v_j||^2 = (1-v_j[a])^2 + v_j[b]^2 = 0+1=1 < 2. FORBIDDEN!
      Or (1-(-1))^2 + 1 = 4+1=5 >= 2. OK if v_j[a] = -1.
    
    So we can add weight-1 vectors, but they conflict with weight-2 vectors that have a +1 in the same position.
    
    This is getting complicated. Let me just go with the greedy approach on a larger pool.
    
    Let me try: pool = all weight-2 vectors with first nonzero positive (110 vectors) PLUS all weight-1 vectors (11 vectors, since first nonzero is the only nonzero and must be positive).
    
    Total pool: 121 vectors. Greedy selection with the constraint.
    
    Actually, let me just implement a general greedy algorithm that works on any pool.
    """
    # Generate candidate pool
    candidates = []
    
    # Weight 1 vectors: e_i (first nonzero is positive, so just +1)
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
    
    # Weight 2 vectors with first nonzero positive
    for i in range(D):
        for j in range(i + 1, D):
            # First nonzero is at position i, must be positive
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 1
                v[j] = sj
                candidates.append(v)
    
    # Weight 3 vectors with first nonzero positive
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = 1
                        v[j] = sj
                        v[k] = sk
                        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Greedy selection
    selected = []
    selected_norms_sq = []
    max_norm_sq = 0
    
    # Sort candidates by norm squared (smaller first to allow more selections)
    order = np.argsort(norms_sq)
    
    for idx in order:
        if _time_left() < 5.0:
            break
        
        v = candidates[idx]
        n_sq = norms_sq[idx]
        
        # Check if adding v violates the constraint
        # New max norm squared would be max(max_norm_sq, n_sq)
        new_max = max(max_norm_sq, n_sq)
        
        ok = True
        for s in selected:
            diff = v - s
            dist_sq = np.dot(diff, diff)
            if dist_sq < new_max:
                ok = False
                break
        
        if ok:
            selected.append(v)
            selected_norms_sq.append(n_sq)
            max_norm_sq = new_max
    
    if len(selected) == 0:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    vec = [0] * D
                    vec[i], vec[j] = si, sj
                    vectors.append(vec)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
