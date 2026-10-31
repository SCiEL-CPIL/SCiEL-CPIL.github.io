import numpy as np
import itertools

D = 11

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance is at least
    the maximum vector norm.
    
    Strategy: Use all weight-4 vectors in {-1, 0, 1}^11 (i.e., vectors with exactly four
    non-zero entries, each ±1). There are C(11,4) * 2^4 = 330 * 16 = 5280 such vectors.
    All have norm squared = 4, so norm = 2.
    
    For any two distinct such vectors u, v:
    - If they share 0 positions: dot = 0, distance squared = 8 ≥ 4 ✓
    - If they share 1 position: dot = ±1, distance squared = 4 - 2(±1) = 4 ∓ 2, so 2 or 6.
      Minimum is 2 < 4. ✗ Wait, this is a problem.
    
    Let me re-check: distance squared = ||u||² + ||v||² - 2 u·v = 4 + 4 - 2u·v = 8 - 2u·v.
    Requirement: 8 - 2u·v ≥ 4 → u·v ≤ 2.
    
    So we need u·v ≤ 2 for all pairs.
    
    For weight-4 vectors:
    - Share 0: dot = 0 ≤ 2 ✓
    - Share 1: dot = ±1 ≤ 2 ✓
    - Share 2: dot = ±2 (if same signs on both shared) or 0 (if opposite signs). Max is 2 ≤ 2 ✓
    - Share 3: dot = ±3 or ±1. If +1+1+1=3 > 2 ✗
    - Share 4: same vector, excluded (distinct vectors).
    
    So weight-4 vectors with 3 shared positions can have dot = 3, which violates the constraint.
    
    Therefore, we cannot take ALL weight-4 vectors. We need a subset where no two share 3 positions
    with all same signs, or more generally, where u·v ≤ 2 for all pairs.
    
    The conflict condition: u·v ≥ 3. For weight-4 vectors, this means they share exactly 3 positions
    and all 3 shared signs are the same (giving dot = 3), or they share 3 positions with 2 same and 1 opposite
    (dot = 1, fine), etc. Actually:
    - Share 3, all same signs: dot = 3 (conflict)
    - Share 3, 2 same 1 opposite: dot = 1 (ok)
    - Share 3, 1 same 2 opposite: dot = -1 (ok)
    - Share 3, all opposite: dot = -3 (ok)
    
    So the only conflicts among weight-4 vectors are pairs that share exactly 3 positions with the
    same sign pattern on those 3 positions.
    
    This is a maximum independent set problem on a specific graph. The current program does this
    but with a greedy approach.
    
    Alternatively, we can use a different construction. Let me think about the Leech lattice or
    other known constructions.
    
    Actually, the kissing number in dimension 11 is known to be at least 550. The current program
    scores 1.99 with the seed, and inspirations score 2.57.
    
    Let me try a different approach: use the known optimal or near-optimal construction.
    
    For dimension 11, a good construction comes from the Leech lattice projected, or from
    coding theory. The shortened binary Goppa code or similar.
    
    Another idea: use vectors from the D11 lattice or B11 lattice.
    
    Let me try: take all vectors of the form where we pick 4 coordinates, assign signs, but
    with a parity constraint to avoid the 3-overlap conflicts.
    
    Actually, let me just use a more effective search. The graph of weight-4 vectors where
    edges represent conflicts (dot ≥ 3) has 5280 nodes. The degree of each node: how many other
    weight-4 vectors share 3 positions with the same sign pattern?
    
    For a fixed vector with 4 non-zero positions, the number of vectors sharing exactly 3 of those
    positions with the same signs: choose which 1 position to differ on (4 choices), then the new
    4th position can be any of the remaining 7 positions (7 choices), and the sign of the new position
    is free (2 choices). But we also need the 3 shared positions to have the same signs. So for each
    choice of which of the 4 positions to drop, we pick a new position from the other 7, and the sign
    is free. That's 4 * 7 * 2 = 56 conflicts per vector.
    
    So the graph is relatively sparse (degree 56 out of 5279). A good MIS algorithm should find
    a large independent set.
    
    Let me use a better heuristic: local search or a more careful greedy.
    """
    
    # Generate all weight-4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    vectors = np.array(v4_candidates, dtype=np.int64)
    N = len(vectors)
    
    # Compute dot products to find conflicts (dot >= 3)
    dots = vectors @ vectors.T
    conflicts = dots >= 3
    np.fill_diagonal(conflicts, False)
    
    # Use a simple but effective greedy: process nodes in order of increasing degree
    degrees = conflicts.sum(axis=1)
    
    # Try multiple random orderings and keep the best
    best_size = 0
    best_mask = np.zeros(N, dtype=bool)
    
    import time
    start = time.time()
    deadline = start + 110
    
    # First, try degree-based greedy
    order = np.argsort(degrees)
    selected = np.zeros(N, dtype=bool)
    excluded = np.zeros(N, dtype=bool)
    for v in order:
        if excluded[v]:
            continue
        selected[v] = True
        excluded[conflicts[v]] = True
    count = selected.sum()
    if count > best_size:
        best_size = count
        best_mask = selected.copy()
    
    # Now try random restarts
    rng = np.random.default_rng(42)
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        random_order = rng.permutation(N)
        selected = np.zeros(N, dtype=bool)
        excluded = np.zeros(N, dtype=bool)
        for v in random_order:
            if excluded[v]:
                continue
            selected[v] = True
            excluded[conflicts[v]] = True
        count = selected.sum()
        if count > best_size:
            best_size = count
            best_mask = selected.copy()
    
    result = vectors[best_mask]
    return result
