# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    their maximum squared norm is strictly less than their minimum pairwise squared distance.

    Strategy:
      1. Start with the set of all weight-2 vectors (squared norm 4), i.e., vectors with
         exactly two ±1 entries and zeros elsewhere. There are 11*10 = 110 such vectors.
         For any two distinct weight-2 vectors u, v:
           - If they share no nonzero coordinate: ||u-v||^2 = 4+4 = 8
           - If they share exactly one nonzero coordinate: ||u-v||^2 = 4+4-2*(±1) = 4 or 8
         The minimum possible distance is 4 (when they share a coordinate with same sign).
         Since max squared norm = 4, we need min pair dist > 4, i.e., >= 5.
         So we need to select a subset where no two vectors share a coordinate with the same sign.
         This is equivalent to: for each coordinate i and sign s in {+1,-1}, at most one
         selected vector has s at position i.
         Actually, let's reconsider: if u and v share coordinate i with the same sign, then
         (u-v)_i = 0, and they differ at other coordinates. The squared distance is:
         sum over all coords of (u_j - v_j)^2.
         If they share coordinate i with same sign: (u_i - v_i)^2 = 0.
         They each have one other nonzero coordinate. If those are different coords:
         distance^2 = 0 + 1 + 1 = 2? No wait.
         
         Let me recompute carefully. Weight-2 vectors have exactly two nonzero entries, each ±1.
         u = (…, +1 at i, +1 at j, …)
         v = (…, +1 at i, -1 at k, …) where k != j
         Then u - v has: 0 at i, +1 at j, +1 at k (since -(-1)=+1), so ||u-v||^2 = 2.
         That's < 4, so invalid.
         
         So weight-2 vectors alone don't form a valid set unless we ensure no two share a coordinate
         with the same sign AND no two share a coordinate at all (to avoid distance 2).
         Actually if they share coordinate i with opposite signs: (u_i - v_i)^2 = (1-(-1))^2 = 4.
         Plus the other two nonzero coords contribute 1+1=2. Total = 6. That's > 4. Good.
         If they share no coordinates: distance^2 = 4+4=8. Good.
         If they share coordinate i with same sign: distance^2 = 0 + 1 + 1 = 2. Bad.
         
         So the constraint is: no two selected vectors may share a coordinate with the same sign.
         Equivalently, for each (i, s) pair, at most one vector in our set has value s at coordinate i.
         There are 11*2 = 22 such (i,s) pairs, and each weight-2 vector uses exactly 2 of them.
         So we can have at most 22/2 = 11 vectors? No wait, that's not right either.
         Each vector uses 2 (i,s) pairs. We have 22 (i,s) pairs total. If each is used at most once,
         max vectors = 22/2 = 11. But we already know 110 weight-2 vectors exist, and the kissing
         number in 11D is much larger.
         
         I think I'm confusing myself. Let me restart with a cleaner approach.
         
    Better strategy: Use the known construction. The kissing number in 11 dimensions is at least
    the number of vectors in the Leech lattice or similar. But for integer vectors, a simpler
    approach: use all vectors of squared norm 4 (weight-2) but only those that are pairwise
    at distance >= 5. As computed, two weight-2 vectors have distance^2 in {2, 4, 6, 8}.
    We need distance^2 > 4 (since max norm^2 = 4), so we need distance^2 >= 5, i.e., 6 or 8.
    Distance^2 = 2 when they share a coord with same sign.
    Distance^2 = 4 when they share a coord with opposite signs? Let me recheck.
    
    u has +1 at i, +1 at j. v has -1 at i, +1 at k.
    u-v: at i: 1-(-1)=2, at j: 1-0=1, at k: 0-1=-1. ||u-v||^2 = 4+1+1=6. OK.
    
    u has +1 at i, +1 at j. v has +1 at i, +1 at k.
    u-v: at i: 0, at j: 1, at k: -1. ||u-v||^2 = 0+1+1=2. Bad.
    
    So the rule: two weight-2 vectors are compatible iff they do NOT share a coordinate with the same sign.
    This is a graph problem. We want a maximum clique in the compatibility graph, or equivalently,
    a maximum independent set in the conflict graph where edges connect incompatible pairs.
    
    Given time constraints, let's use a greedy approach with a good ordering, then try to improve.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    d = 11
    
    # Generate all weight-2 vectors: choose 2 coordinates from 11, assign signs.
    # Total: C(11,2) * 4 = 55 * 4 = 220 vectors.
    # Wait, C(11,2) = 55, and for each pair of coords, 4 sign combinations. So 220 vectors.
    # But earlier I said 110. Let me recount: 11*10/2 = 55 pairs, times 4 = 220. Yes, 220.
    
    # For each vector, record which (coord, sign) pairs it uses.
    # Two vectors conflict if they share a (coord, sign) pair.
    
    # Build list of all 220 vectors as tuples of (coord, sign) pairs.
    vectors = []
    for i in range(d):
        for j in range(i+1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    # Vector has si at coord i, sj at coord j
                    vectors.append(((i, si), (j, sj)))
    
    n_vecs = len(vectors)  # 220
    
    # Build conflict graph: two vectors conflict if they share a (coord, sign) pair.
    # For each (coord, sign), find all vectors that use it.
    from collections import defaultdict
    coord_sign_to_vecs = defaultdict(list)
    for idx, vec in enumerate(vectors):
        for cs in vec:
            coord_sign_to_vecs[cs].append(idx)
    
    # Conflict adjacency: for each pair of vectors that share a (coord,sign), they conflict.
    # Build adjacency list.
    adj = [set() for _ in range(n_vecs)]
    for cs, vec_list in coord_sign_to_vecs.items():
        for a in range(len(vec_list)):
            for b in range(a+1, len(vec_list)):
                u, v = vec_list[a], vec_list[b]
                adj[u].add(v)
                adj[v].add(u)
    
    # Now find a large independent set in this conflict graph.
    # Greedy: order by degree (ascending), pick vertices with no conflicts among chosen.
    # Better: use a greedy coloring-based approach or just greedy with good ordering.
    
    # Sort by degree ascending
    degrees = [len(adj[i]) for i in range(n_vecs)]
    order = sorted(range(n_vecs), key=lambda x: degrees[x])
    
    selected = []
    selected_set = set()
    for idx in order:
        if time.time() > deadline:
            break
        # Check if idx conflicts with any selected
        compatible = True
        for sel in selected:
            if idx in adj[sel]:
                compatible = False
                break
        if compatible:
            selected.append(idx)
            selected_set.add(idx)
    
    # Try to improve: for each unselected vertex, check if it can be added
    # and if so, add it. Repeat until no improvement.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n_vecs):
            if idx in selected_set:
                continue
            compatible = True
            for sel in selected:
                if idx in adj[sel]:
                    compatible = False
                    break
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
                improved = True
                if time.time() > deadline:
                    break
    
    # Convert selected indices to point array
    points = np.zeros((len(selected), d), dtype=np.int64)
    for row, idx in enumerate(selected):
        for (coord, sign) in vectors[idx]:
            points[row, coord] = sign
    
    # Verify the result
    if len(points) < 2:
        # Fallback: return the trivial 2-point solution
        points = np.array([np.ones(d), -1 * np.ones(d)], dtype=np.int64)
    else:
        # Check validity
        norms_sq = np.sum(points ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        min_pair_dist_sq = np.inf
        k = len(points)
        for i in range(k):
            for j in range(i+1, k):
                diff = points[i] - points[j]
                d_sq = np.sum(diff ** 2)
                if d_sq < min_pair_dist_sq:
                    min_pair_dist_sq = d_sq
        if min_pair_dist_sq <= max_norm_sq:
            # Invalid, fallback
            points = np.array([np.ones(d), -1 * np.ones(d)], dtype=np.int64)
    
    points = np.around(points).astype(np.int64)
    return points


# EVOLVE-BLOCK-END
