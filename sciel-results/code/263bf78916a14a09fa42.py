"""Kissing number in dimension 11: structural reformulation as MIS on support graph + SA."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    D = 11

    # The key structural insight: for weight-4 vectors (norm^2 = 4),
    # the maximum norm squared is 4. Two weight-4 vectors v, w satisfy
    # ||v - w||^2 >= 4 iff v·w <= (4+4-4)/2 = 2.
    # For two weight-4 vectors, v·w = sum over intersection of supports of (sign products).
    # If |intersection| = 3, then v·w can be -3,-1,1,3. It's <= 2 iff not 3.
    # v·w = 3 iff all three shared coordinates have matching signs.
    # If |intersection| = 4, v·w can be -4,-2,0,2,4. It's <= 2 iff not 4.
    # v·w = 4 iff all four shared coordinates have matching signs.
    # If |intersection| <= 2, v·w <= 2 always holds (max |v·w| = 2).
    
    # So the conflict structure among weight-4 vectors depends on:
    # 1. Whether supports intersect in 3 or 4 elements
    # 2. The sign patterns on the intersection
    
    # However, a cleaner structural decomposition:
    # Group weight-4 vectors by their support (4-element subset of {0,...,10}).
    # There are C(11,4) = 330 supports.
    # For each support, there are 16 sign patterns.
    # Two vectors with the SAME support: v·w = sum of 4 sign products.
    #   v·w = 4 iff same signs, 2 iff 3 match, 0 iff 2 match, -2 iff 1 match, -4 iff 0 match.
    #   Conflict (v·w > 2) only when v·w = 4, i.e., identical vectors.
    #   So within a support group, only identical vectors conflict. We can pick at most 1 per support... 
    #   wait, no. We want to pick multiple vectors. If v·w = 2, that's OK (<= 2).
    #   Actually v·w = 2 means ||v-w||^2 = 4+4-4 = 4 >= 4. OK.
    #   v·w = 3 is impossible for weight-4 vectors with same support (sum of 4 ±1 products).
    #   So within a support, ANY two distinct sign patterns are compatible!
    #   We can pick all 16 vectors from a support.
    
    # Two vectors with DIFFERENT supports:
    # Let S1, S2 be supports, |S1 ∩ S2| = k.
    # v·w = sum over S1∩S2 of (v_i * w_i). Max |v·w| = k.
    # If k <= 2: always compatible (|v·w| <= 2).
    # If k = 3: v·w ∈ {-3,-1,1,3}. Conflict iff v·w = 3 (all three signs match).
    # If k = 4: same support, handled above.
    
    # So conflicts between different supports occur ONLY when |S1 ∩ S2| = 3 AND
    # the sign patterns agree on the 3 shared coordinates.
    
    # This means: if we fix a set of supports, and for each support choose sign patterns,
    # the compatibility depends on sign agreement on shared coordinates.
    
    # Key structural simplification: If we pick a set of supports that are pairwise
    # disjoint in their 3-element intersections... no, that's not quite right.
    
    # Alternative approach: For each support, we can pick ALL 16 sign patterns.
    # The question is which supports to include. If we include support S1 and S2 with
    # |S1 ∩ S2| = 3, then we need to check: for every sign pattern a on S1 and b on S2,
    # is a·b <= 2 on the intersection? No! We need EVERY pair (v from S1 group, w from S2 group)
    # to be compatible. But if there exist sign patterns that conflict, we can't include both groups fully.
    
    # Wait, re-reading: we need the MINIMUM pairwise distance >= max norm. So EVERY pair must be compatible.
    # If we include all 16 vectors from support S1 and all 16 from support S2 where |S1∩S2|=3,
    # then there exist pairs with v·w = 3 on the intersection, which violates the constraint.
    # So we CANNOT include full groups from two supports that intersect in 3.
    
    # Therefore, the structural problem reduces to:
    # Choose a set of supports such that no two chosen supports intersect in exactly 3 elements.
    # (Supports that are identical are the same support; supports intersecting in 0,1,2,4 are fine.)
    # Note: two distinct 4-subsets of an 11-set can intersect in 0,1,2,3,4.
    # Intersect in 4 means they're the same. So distinct supports intersect in 0,1,2,or 3.
    
    # So: find a maximum set of 4-subsets of {0,...,10} such that no two intersect in exactly 3.
    # This is a combinatorial design problem!
    
    # A 4-subset S and another 4-subset T with |S∩T|=3 means T = (S\{a}) ∪ {b} for some a∈S, b∉S.
    # So T is obtained from S by swapping one element.
    
    # The conflict graph on 330 supports: edge between S and T if |S∩T|=3.
    # This is the "3-intersection graph" of 4-subsets.
    # We need the maximum independent set in this graph.
    
    # Then, for each chosen support, we include all 16 sign patterns.
    # Total vectors = 16 * |chosen supports|.
    
    # Additionally, we can potentially add weight-3, weight-2, weight-1 vectors too,
    # but let's first maximize the weight-4 contribution.
    
    # Generate all 4-subsets
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330
    
    # Build conflict graph: edge if |intersection| == 3
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i + 1, n_sup):
            s_j = set(supports[j])
            if len(s_i & s_j) == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Now solve MIS on this 330-node graph using simulated annealing
    # State: S = set of chosen support indices
    # Move: pick v not in S. N = neighbors of v in S. New S = S - N + {v}.
    # Delta = 1 - |N|.
    
    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        return current_set
    
    best_set = greedy_is()
    best_size = len(best_set)
    
    # Simulated annealing
    np.random.seed(42)
    random.seed(42)
    
    S = best_set.copy()
    current_size = len(S)
    T = 1.0
    cooling = 0.9995
    
    # Precompute adjacency as lists for faster iteration
    adj_list = [list(s) for s in adj]
    
    step = 0
    max_steps = 200000
    while step < max_steps and time.time() < deadline - 20.0:
        step += 1
        v = random.randint(0, n_sup - 1)
        
        if v in S:
            # Try removing v
            if random.random() < 0.1:
                S.discard(v)
                current_size -= 1
            continue
        
        # v not in S
        N = adj[v] & S
        n_N = len(N)
        
        if n_N == 0:
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            delta = 1 - n_N
            if delta >= 0:
                S -= N
                S.add(v)
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                # delta < 0
                if T > 0.01:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S -= N
                        S.add(v)
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S.copy()
        
        T = max(0.01, T * cooling)
    
    # Also try multiple restarts of greedy + SA
    for restart in range(20):
        if time.time() > deadline - 15.0:
            break
        np.random.seed(42 + restart)
        random.seed(42 + restart)
        
        # Randomized greedy
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        
        S = current_set.copy()
        current_size = len(S)
        T = 1.0
        cooling = 0.9995
        
        for step2 in range(50000):
            if time.time() > deadline - 15.0:
                break
            v = random.randint(0, n_sup - 1)
            
            if v in S:
                if random.random() < 0.1:
                    S.discard(v)
                    current_size -= 1
                continue
            
            N = adj[v] & S
            n_N = len(N)
            
            if n_N == 0:
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                delta = 1 - n_N
                if delta >= 0:
                    S -= N
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    if T > 0.01:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S -= N
                            S.add(v)
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()
            
            T = max(0.01, T * cooling)
    
    # Construct vectors from best_set: all 16 sign patterns for each chosen support
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
    
    # Now try to add weight-3, weight-2, weight-1 vectors that are compatible with all
    # Generate weight-3 vectors
    weight3_candidates = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            weight3_candidates.append(v)
    
    # Generate weight-2 vectors
    weight2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            weight2_candidates.append(v)
    
    # Generate weight-1 vectors
    weight1_candidates = []
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            weight1_candidates.append(v)
    
    # Check compatibility of each additional candidate with all existing vectors
    # A candidate c is compatible with vector v if ||c - v||^2 >= max_norm_sq
    # max_norm_sq = 4 (from weight-4 vectors)
    # ||c - v||^2 = ||c||^2 + ||v||^2 - 2*c·v
    # For c weight-3, v weight-4: 3 + 4 - 2*c·v >= 4 => c·v <= 1.5 => c·v <= 1
    # For c weight-2, v weight-4: 2 + 4 - 2*c·v >= 4 => c·v <= 1
    # For c weight-1, v weight-4: 1 + 4 - 2*c·v >= 4 => c·v <= 0.5 => c·v <= 0
    # Also need compatibility among the added vectors themselves.
    
    # This is getting complex. Let's just try to greedily add compatible vectors.
    existing = np.array(vectors, dtype=np.int32)
    if len(existing) > 0:
        existing_norms_sq = np.sum(existing ** 2, axis=1)
        max_norm_sq = int(existing_norms_sq.max())
        
        # Try adding weight-3 vectors
        for w3 in weight3_candidates:
            if time.time() > deadline - 5.0:
                break
            w3_arr = np.array(w3, dtype=np.int32)
            w3_norm_sq = int(np.sum(w3_arr ** 2))
            # Check against all existing
            dots = existing @ w3_arr
            min_dist_sq = w3_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w3)
                existing = np.vstack([existing, w3_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())
        
        # Try adding weight-2 vectors
        for w2 in weight2_candidates:
            if time.time() > deadline - 3.0:
                break
            w2_arr = np.array(w2, dtype=np.int32)
            w2_norm_sq = int(np.sum(w2_arr ** 2))
            dots = existing @ w2_arr
            min_dist_sq = w2_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w2)
                existing = np.vstack([existing, w2_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())
        
        # Try adding weight-1 vectors
        for w1 in weight1_candidates:
            if time.time() > deadline - 2.0:
                break
            w1_arr = np.array(w1, dtype=np.int32)
            w1_norm_sq = int(np.sum(w1_arr ** 2))
            dots = existing @ w1_arr
            min_dist_sq = w1_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w1)
                existing = np.vstack([existing, w1_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())
    
    if not vectors:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
    
    result = np.array(vectors, dtype=np.int64)
    return result
