import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # Strategy: Work at support level.
    # For each support S (subset of {0,...,D-1}), we can choose a set of sign patterns.
    # The constraint: for all pairs of vectors (from any two supports), dist^2 >= max_norm_sq.
    # 
    # Key: if all vectors have the same weight w, then max_norm_sq = w.
    # For same support S: two sign patterns with Hamming distance k have dist^2 = 2k.
    #   Need 2k >= w, so k >= ceil(w/2).
    # For different supports S, T: dist^2 = w + w - 2*(dot product) = 2w - 2*(dot).
    #   Need 2w - 2*(dot) >= w, so dot <= w/2.
    #   dot = sum over S∩T of (sign products). |dot| <= |S∩T|.
    #   So need |S∩T| <= floor(w/2) for guaranteed compatibility, OR
    #   if |S∩T| > floor(w/2), need specific sign choices.
    #
    # For w=2: need dot <= 1. |S∩T| <= 1 always works. |S∩T|=2 means same support.
    #   All C(11,2)=55 supports, each with 4 sign patterns (all valid since min dist=2).
    #   Total: 55*4 = 220. This is the seed.
    #
    # For w=3: need dot <= 1. |S∩T| <= 1 works. |S∩T|=2: dot can be -2,-1,0,1,2.
    #   Need dot <= 1, so not all sign pairs work.
    #   Per support: sign patterns must differ in >= 2 positions (min dist^2=4>=3? No, 2*2=4>=3 OK).
    #   Actually for w=3: need 2k >= 3, so k >= 2. Code of length 3, min dist 2: max 4 (even parity).
    #
    # For w=4: need dot <= 2. |S∩T| <= 2 works. |S∩T|=3: need dot<=2.
    #   Per support: need 2k >= 4, k >= 2. Code length 4, min dist 2: max 8 (even parity) or...
    #   Actually all 16 sign patterns: min dist is 2 (differ in 1 pos). 2 < 4. Conflict!
    #   Need k >= 2. Max code of length 4, min dist 2: 8 (even parity).
    #
    # Let me try a mixed approach: use multiple weights.
    # Actually, let me try the weighted greedy MIS approach on supports.
    #
    # New approach: Consider ALL possible supports of size 1-4.
    # For each support, compute the max number of sign patterns that can be included
    # (subject to intra-support constraints).
    # Then build a conflict graph between supports (two supports conflict if NO sign
    # assignment can make them all mutually compatible).
    # Use weighted greedy MIS to select supports, then assign optimal sign patterns.
    #
    # Actually, let me try a simpler but effective approach:
    # Use weight-2 supports (all 55, 4 each = 220) PLUS try to add weight-3 supports
    # that don't conflict.
    #
    # Wait - if I mix weights, max_norm_sq = max weight used.
    # If I use weights 2 and 3: max_norm_sq = 3.
    # Then weight-2 pairs need dist^2 >= 3. But min dist^2 between weight-2 vectors is 2.
    # So I'd need to remove many weight-2 vectors. Probably not worth it.
    #
    # Let me try: pure weight-3 with greedy support selection.
    # C(11,3) = 165 supports. Each can have up to 4 vectors.
    # Conflict: two supports S,T with |S∩T|=2 may conflict depending on signs.
    # But we can choose signs to avoid conflict if possible.
    #
    # Two weight-3 vectors u (support S) and v (support T), |S∩T|=2:
    # Let S∩T = {a,b}, S\T = {c}, T\S = {d}.
    # u = (s_a, s_b, s_c, 0, 0, 0, 0, 0, 0, 0, 0) [in appropriate positions]
    # v = (t_a, t_b, 0, t_d, 0, 0, 0, 0, 0, 0, 0)
    # u·v = s_a*t_a + s_b*t_b
    # dist^2 = 3+3-2*(s_a*t_a + s_b*t_b) = 6 - 2*(s_a*t_a + s_b*t_b)
    # Need >= 3: s_a*t_a + s_b*t_b <= 1.5, i.e., <= 1.
    # s_a*t_a + s_b*t_b can be -2, -1, 0, 1, 2.
    # Conflict only when s_a*t_a + s_b*t_b = 2, i.e., s_a=t_a AND s_b=t_b.
    # So for a given pair of supports with |S∩T|=2, we need:
    # For ALL u in sign_set(S), v in sign_set(T): not (s_a=t_a AND s_b=t_b).
    # This means: the projections of sign_set(S) and sign_set(T) onto {a,b} must be
    # "disjoint" in the sense that no pair matches on both coordinates.
    #
    # If sign_set(S) is even-parity on {a,b,c}: {(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)}
    # Projections onto {a,b}: {(1,1),(1,-1),(-1,1),(-1,-1)} - all 4 possibilities!
    # So ANY sign_set(T) with 4 even-parity patterns will have a conflict.
    #
    # What if we use smaller sign sets? If sign_set(S) has only 2 patterns with different
    # projections, e.g., {(1,1,1),(1,-1,-1)}, projections: {(1,1),(1,-1)}.
    # Then sign_set(T) must avoid having (1,1) or (1,-1) on {a,b}.
    # So sign_set(T) can only use patterns with s_a=-1: {(-1,1,1),(-1,-1,-1)} if even parity.
    # That's 2 patterns.
    #
    # This is getting complex. Let me try a computational approach.
    #
    # APPROACH: Weighted greedy MIS on supports for weight-3.
    # 1. Generate all C(11,3)=165 supports.
    # 2. For each support, weight = 4 (max sign patterns).
    # 3. Build conflict graph: S~T if |S∩T|=2 (potential conflict).
    #    (|S∩T|=0 or 1: always compatible. |S∩T|=3: same support.)
    # 4. Use weighted greedy: repeatedly pick the support with highest weight/(1+degree),
    #    remove it and its neighbors.
    # 5. For selected supports, assign sign patterns greedily.
    # 6. Also include all weight-2 vectors that don't conflict? No, mixing weights is bad.
    #
    # Actually, let me just try to get a good set of weight-3 supports and see the count.
    # If I get ~40 supports * 4 = 160, that's less than 220.
    # If I get ~60 supports * 4 = 240, that beats 220!
    #
    # Let me also consider: what about using weight-2 AND some weight-3?
    # If max_norm_sq = 3, I need to prune weight-2 vectors that conflict.
    # Two weight-2 vectors with dist^2 = 2 conflict (need >= 3).
    # e_i+e_j and e_i+e_k (sharing i, same sign): dist^2 = 2. Conflict.
    # How many such pairs? For each i, the 20 vectors e_i±e_j (j≠i).
    # Among these, pairs with same sign on i: 10 pairs (e_i+e_j vs e_i+e_k for j<k).
    # Total conflicting pairs: 11 * 10 = 110 pairs.
    # To remove all conflicts, I'd need to remove at least... this is a vertex cover problem.
    # Probably need to remove ~110 vectors, leaving ~110 weight-2 + some weight-3.
    # Not clearly better.
    #
    # Let me focus on pure weight-3 with good support selection.
    #
    # Alternative: what about weight-4?
    # C(11,4) = 330 supports. Each can have up to 8 sign patterns (even parity, min dist 2, need >= 4? No).
    # For w=4: need 2k >= 4, k >= 2. Even parity code length 4: 8 codewords, min dist 2. OK.
    # Conflict: |S∩T| <= 2 always works (dot <= 2, need <= 2). |S∩T|=3: need dot <= 2.
    #   dot can be -3,-2,-1,0,1,2,3. Conflict when dot=3.
    #   dot=3 means all 3 shared coords have same sign product = 1.
    # |S∩T|=4: same support.
    #
    # For weight-4, supports with |S∩T|<=2 are always compatible.
    # Number of 4-subsets of [11] with pairwise intersection <= 2:
    # This is a 4-(11,4,2) packing. Each pair is in at most 2 blocks? No, intersection <= 2 means
    # each pair of elements is in at most... let me think differently.
    # Each block has C(4,2)=6 pairs. Total pairs: C(11,2)=55. If each pair in at most 1 block: max 55/6 ≈ 9.
    # But intersection <= 2 means two blocks can share at most 2 elements, i.e., at most 1 pair.
    # So it IS a packing: each pair in at most 1 block. Max blocks: floor(55/6) = 9.
    # 9 * 8 = 72. Much less than 220.
    #
    # What about intersection <= 3 (allowing some conflicts to be resolved by sign choice)?
    # This is harder. Let me try a different approach.
    #
    # NEW IDEA: Use a greedy approach directly on individual vectors.
    # Generate candidate vectors (all weight-2 and weight-3 with entries ±1).
    # Weight-2: 220 vectors. Weight-3: 165*8 = 1320 vectors. Total: 1540.
    # Build conflict graph. Find large independent set.
    # But 1540 nodes with potentially dense conflicts - MIS is NP-hard.
    # Use greedy: repeatedly pick vector with fewest conflicts among remaining.
    #
    # Actually, let me try the weighted greedy on supports for weight-3,
    # and see if I can get enough supports.
    #
    # Let me just implement it and see.

    # Generate all weight-3 supports
    supports = list(itertools.combinations(range(D), 3))
    n_sup = len(supports)
    
    # Build adjacency: S~T if |S∩T| == 2
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i+1, n_sup):
            inter = len(set(supports[i]) & set(supports[j]))
            if inter == 2:
                adj[i].add(j)
                adj[j].add(i)
    
    # Weighted greedy MIS: pick node with max weight/(1+degree), remove it and neighbors
    # Weight for each support = 4 (max sign patterns)
    weights = [4] * n_sup
    in_graph = [True] * n_sup
    selected = []
    
    while True:
        if time.time() > deadline - 10.0:
            break
        best = -1
        best_score = -1
        for i in range(n_sup):
            if not in_graph[i]:
                continue
            deg = len(adj[i])
            # Count only neighbors still in graph
            active_deg = sum(1 for j in adj[i] if in_graph[j])
            score = weights[i] / (1 + active_deg)
            if score > best_score:
                best_score = score
                best = i
        if best < 0:
            break
        selected.append(best)
        in_graph[best] = False
        for j in adj[best]:
            in_graph[j] = False
    
    # Now assign sign patterns to selected supports
    # For each support, use even-parity sign patterns (4 of them)
    # But need to check cross-support compatibility
    # For supports with |S∩T|<=1: always OK
    # For supports with |S∩T|=2: need to verify
    
    # Since we selected an independent set in the conflict graph (|S∩T|=2 means conflict),
    # all selected supports have |S∩T| <= 1. So all sign patterns work!
    
    all_vectors = []
    for s in selected:
        sup = supports[s]
        # All 8 sign patterns? No, need min dist within support >= 3.
        # 2k >= 3 means k >= 2. So sign patterns must differ in >= 2 positions.
        # Even parity: 4 patterns. Or we can use all 8 and check.
        # Let's use even parity (sum of signs = 0 mod 2, i.e., even number of -1s)
        for bits in range(8):
            signs = [1 if (bits >> b) & 1 else -1 for b in range(3)]
            # Check parity: even number of -1s
            n_neg = signs.count(-1)
            if n_neg % 2 != 0:
                continue
            v = np.zeros(D, dtype=np.int8)
            for b, idx in enumerate(sup):
                v[idx] = signs[b]
            all_vectors.append(v)
    
    vec_arr = np.array(all_vectors, dtype=np.int8) if all_vectors else np.array([], dtype=np.int8).reshape(0, D)
    count_w3 = len(vec_arr)
    
    # Now try to add weight-2 vectors that don't conflict
    # If we have weight-3 vectors, max_norm_sq = 3.
    # Weight-2 vector u conflicts with weight-3 vector v if dist^2 < 3.
    # dist^2 = 2 + 3 - 2*(u·v) = 5 - 2*(u·v). Need >= 3: u·v <= 1.
    # u·v for weight-2 u and weight-3 v:
    # If |supp(u) ∩ supp(v)| = 0: u·v = 0. OK.
    # If |supp(u) ∩ supp(v)| = 1: u·v = ±1. OK (<=1).
    # If |supp(u) ∩ supp(v)| = 2: u·v = sum of 2 sign products = -2,-1,0,1,2.
    #   Conflict when u·v = 2.
    
    # So a weight-2 vector conflicts with a weight-3 vector only when they share both
    # support indices AND the signs match on both.
    
    # Let's try to add weight-2 vectors greedily.
    w2_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = si
                    v[j] = sj
                    w2_candidates.append(v)
    
    w2_arr = np.array(w2_candidates, dtype=np.int8)
    n_w2 = len(w2_arr)
    
    # For each weight-2 candidate, check if it conflicts with any existing weight-3 vector
    # Conflict: dist^2 < 3, i.e., u·v > 1 (where u is w2, v is w3)
    # Since u·v is integer, conflict when u·v >= 2.
    
    added_w2 = []
    if count_w3 > 0:
        # Vectorized check
        # For each w2 candidate, compute max dot with any w3 vector
        # If max dot <= 1, it's safe
        for idx in range(n_w2):
            if time.time() > deadline - 5.0:
                break
            u = w2_arr[idx]
            dots = vec_arr @ u  # dot products with all w3 vectors
            if np.max(dots) <= 1:
                added_w2.append(u)
            # Also check: does u conflict with already-added w2 vectors?
            # Two w2 vectors: dist^2 = 4 - 2*(u·u'). Need >= 3: u·u' <= 0.5, i.e., u·u' <= 0.
            # So two w2 vectors conflict if their dot product >= 1.
            # u·u' for weight-2 vectors: -2,-1,0,1,2.
            # Conflict when u·u' >= 1.
            # Check against already added w2:
            conflict_with_added = False
            for v in added_w2:
                if u @ v >= 1:
                    conflict_with_added = True
                    break
            if not conflict_with_added:
                added_w2.append(u)
            # Wait, I'm appending twice. Let me fix.
    
    # Let me redo the w2 addition more carefully
    added_w2 = []
    if count_w3 > 0:
        for idx in range(n_w2):
            if time.time() > deadline - 5.0:
                break
            u = w2_arr[idx]
            # Check against all w3 vectors
            dots = vec_arr @ u
            if np.max(dots) > 1:
                continue
            # Check against already added w2 vectors
            conflict = False
            for v in added_w2:
                if int(u @ v) >= 1:
                    conflict = True
                    break
            if not conflict:
                added_w2.append(u)
    else:
        # No w3 vectors, add all w2 (the seed)
        added_w2 = list(w2_arr)
    
    # Combine
    if added_w2:
        w2_final = np.array(added_w2, dtype=np.int8)
    else:
        w2_final = np.array([], dtype=np.int8).reshape(0, D)
    
    if count_w3 > 0 and len(w2_final) > 0:
        result = np.vstack([vec_arr, w2_final])
    elif count_w3 > 0:
        result = vec_arr
    else:
        result = w2_final
    
    # Verify
    if len(result) > 1:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # Check pairwise distances
        # For large arrays, do it in batches
        n = len(result)
        ok = True
        for i in range(0, n, 100):
            if time.time() > deadline - 2.0:
                break
            batch = result[i:i+100]
            for j in range(i+1, n):
                if time.time() > deadline - 2.0:
                    ok = False
                    break
                d2 = np.sum((batch - result[j])**2, axis=1)
                if np.min(d2) < max_norm_sq:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            # Fallback to seed
            result = w2_arr
    
    return result
