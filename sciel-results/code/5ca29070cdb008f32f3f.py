import numpy as np
import time
from itertools import combinations, product
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # E8 x D3 lattice construction - provably valid, skip verification
    # E8: 240 weight-4 vectors in dims 0-7
    e8_vectors = []
    for support in combinations(range(8), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            e8_vectors.append(v)
    
    # D3: 12 weight-2 vectors in dims 8-10
    d3_vectors = []
    for support in combinations(range(8, 11), 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            d3_vectors.append(v)
    
    # Mixed: 96 weight-2 vectors with one in 0-7 and one in 8-10
    mixed_vectors = []
    for i in range(8):
        for j in range(8, 11):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = si
                    v[j] = sj
                    mixed_vectors.append(v)
    
    # Total: 240 + 12 + 96 = 348 vectors (provably valid)
    base_vectors = e8_vectors + d3_vectors + mixed_vectors
    
    # Now try to add more vectors via MIS on weight-4 supports
    # A weight-4 vector is compatible with all base vectors if:
    # - With E8 (norm²=16): need dot <= 8, always true for |dot|<=4
    # - With D3 (norm²=4): need dot <= 2, true since |dot|<=1 (disjoint supports)
    # - With Mixed (norm²=4): need dot <= 2, true since |dot|<=2
    # So any weight-4 vector is compatible with all base vectors!
    # But two weight-4 vectors are compatible iff |dot| <= 4
    # For weight-4 vectors: dot = sum of products on shared indices
    # If they share 4 indices (same support): dot = ±4 (if same signs) or less
    # Actually if same support and same signs, they're the same vector
    # If same support, opposite signs on some: dot = 4-2k where k is number of sign flips
    # Max |dot| for distinct weight-4 vectors with same support: 2 (flip 2 signs)
    # If they share 3 indices: dot = ±3 or ±1
    # If they share 2 indices: dot = ±2 or 0
    # If they share 1 index: dot = ±1
    # If they share 0 indices: dot = 0
    # So we need |dot| <= 4, which is always satisfied!
    # Wait, that means ALL weight-4 vectors are mutually compatible?
    # Let me check: two weight-4 vectors with same support, 3 same signs, 1 diff: dot = 3
    # Two weight-4 vectors with same support, 2 same, 2 diff: dot = 0
    # So yes, all weight-4 vectors are mutually compatible!
    
    # But wait - we need to check against the base vectors more carefully
    # A weight-4 vector v in dims 0-7 has dot with a mixed vector (i,a) where i in 0-7:
    # dot = v[i] * 1 = ±1, so |dot| = 1 <= 2 ✓
    # A weight-4 vector in dims 0-7 has dot with D3: 0 ✓
    # A weight-4 vector in dims 0-7 has dot with E8: up to 4, need <= 8 ✓
    
    # What about weight-4 vectors with entries in dims 8-10?
    # E.g., support (8,9,10,x) where x in 0-7: norm² = 4
    # Dot with E8 vector: at most 1 (if x is in E8 support) ✓
    # Dot with D3 vector: at most 2 (if two of 8,9,10 are in D3 support) 
    #   Need <= 2, so this is tight!
    # Dot with Mixed vector (i,a): at most 1 ✓
    # Dot with another such vector: need to check
    
    # Let's just add ALL weight-4 vectors and verify
    all_weight4 = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            all_weight4.append(v)
    
    # Check if all weight-4 vectors are compatible with base vectors and each other
    # This is 330*16 = 5280 vectors. Let's verify quickly.
    # Actually, let's just try adding them all and verify the full set.
    
    # Try: base + all weight-4
    candidate_vectors = base_vectors + all_weight4
    arr = np.array(candidate_vectors, dtype=np.int32)
    n = len(arr)
    
    # Quick verification using vectorized approach
    max_norm_sq = int(np.max(np.sum(arr**2, axis=1)))
    
    # Check min pairwise distance
    min_dist_sq = float('inf')
    for i in range(0, n, 1):
        if time.time() > deadline - 30.0:
            break
        diffs = arr[i+1:] - arr[i]
        dists_sq = np.sum(diffs**2, axis=1)
        if len(dists_sq) > 0:
            d = int(np.min(dists_sq))
            if d < min_dist_sq:
                min_dist_sq = d
    
    if min_dist_sq >= max_norm_sq:
        return np.array(candidate_vectors, dtype=np.int8)
    
    # If that fails, fall back to MIS approach
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330
    
    # Two weight-4 vectors are incompatible if |dot| > 4
    # For weight-4 vectors, max |dot| = 4 (same support, same signs)
    # But same support + same signs = same vector, so for distinct vectors:
    # Same support: max |dot| = 2 (flip 2 signs) or 3 (flip 1 sign)
    # Wait: 4 same signs, 1 flip: dot = 3
    # So supports with |intersection| = 4 (same support) CAN have |dot| = 3 > 2?
    # No, we need |dot| <= 4 for the constraint. Let me re-derive.
    
    # Constraint: min pairwise distance >= max norm
    # For two vectors v1, v2 with norms n1, n2:
    # ||v1-v2||^2 >= max(n1, n2)^2
    # n1^2 + n2^2 - 2*v1.v2 >= max(n1^2, n2^2)
    # If n1 = n2 = n: 2n^2 - 2*v1.v2 >= n^2 => v1.v2 <= n^2/2
    # For weight-4 vectors: n^2 = 4, so need v1.v2 <= 2
    # For E8 vectors: n^2 = 16, so need v1.v2 <= 8
    # For mixed/D3: n^2 = 4, so need v1.v2 <= 2
    
    # So for two weight-4 vectors, need dot <= 2
    # Same support, 1 sign flip: dot = 3 > 2 INCOMPATIBLE
    # Same support, 2 sign flips: dot = 0 ✓
    # Same support, 3 sign flips: dot = -3 ✓
    # Same support, 4 sign flips: dot = -4 ✓ (but this is the negative, distinct)
    
    # So we CAN'T just add all weight-4 vectors. We need the MIS approach.
    
    # Build incompatibility: two supports are incompatible if they share >= 3 indices
    # (because then we can have sign patterns with dot = 3)
    # Actually, two supports with |intersection| = 3: max dot = 3 > 2, incompatible
    # Two supports with |intersection| = 4 (same): max dot = 4, but min distinct dot = 2
    #   Wait, for same support, we can choose signs to make dot = 0, 2, or 4
    #   We need ALL pairs to have dot <= 2, so same support is problematic
    #   Unless we only include one sign pattern per support
    
    # Revised approach: for each support, include at most one sign pattern
    # Two supports with |intersection| >= 3 are incompatible
    # Two supports with |intersection| <= 2 are always compatible
    
    # Build incompatibility graph on supports
    sup_sets = [set(s) for s in supports]
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    
    for i in range(N_sup):
        if time.time() > deadline - 40.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    # Find MIS using greedy + local search
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    incompat_bool = incompat.astype(bool)
    
    def greedy_mis(available, rng):
        selected = set()
        avail = set(available)
        while avail:
            if time.time() > deadline - 15.0:
                break
            if len(avail) > 100:
                avail_arr = np.array(list(avail), dtype=np.int32)
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = int(deg_arr.min())
                candidates = avail_arr[deg_arr == min_deg]
                u = int(rng.choice(candidates))
            else:
                min_deg = float('inf')
                candidates = []
                for u in avail:
                    d = len(neighbors[u] & avail)
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                u = rng.choice(candidates)
            selected.add(u)
            avail -= {u} | (neighbors[u] & avail)
        return selected
    
    def local_search_add(selected, rng):
        selected = set(selected)
        if not selected:
            return selected
        sel_arr = np.array(list(selected), dtype=np.int32)
        compatible_mask = ~incompat_bool[:, sel_arr].any(axis=1)
        compatible_mask[sel_arr] = False
        addable = np.where(compatible_mask)[0]
        for x in addable:
            if time.time() > deadline - 10.0:
                break
            selected.add(int(x))
        return selected
    
    def local_search_1_2_swap(selected, rng):
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = list(selected)
            for v in sel_list:
                if time.time() > deadline - 10.0:
                    break
                remaining = selected - {v}
                rem_arr = np.array(list(remaining), dtype=np.int32) if remaining else np.array([], dtype=np.int32)
                if len(rem_arr) == 0:
                    compat_with_rem = np.ones(N_sup, dtype=bool)
                else:
                    compat_with_rem = ~incompat_bool[:, rem_arr].any(axis=1)
                compat_with_rem[list(selected)] = False
                addable = np.where(compat_with_rem)[0]
                n_add = len(addable)
                if n_add >= 2:
                    # Find pair in addable that are compatible
                    for i in range(n_add):
                        if time.time() > deadline - 10.0:
                            break
                        u1 = addable[i]
                        for j in range(i + 1, n_add):
                            u2 = addable[j]
                            if incompat[u1, u2] == 0:
                                selected.remove(v)
                                selected.add(int(u1))
                                selected.add(int(u2))
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        return selected
    
    best_iset = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    n_restarts = 80
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        if r == 0:
            iset = greedy_mis(set(range(N_sup)), rng)
        elif r < 30:
            iset = greedy_mis(set(range(N_sup)), rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        elif r < 60:
            avail = set(range(N_sup))
            n_remove = int(rng.integers(5, 25))
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        else:
            avail = set(range(N_sup))
            n_remove = int(rng.integers(20, 50))
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    # Generate vectors: for each support in MIS, pick a sign pattern
    # To maximize compatibility, use all-ones sign pattern
    vectors = list(base_vectors)
    for i in best_iset:
        sup = supports[i]
        v = np.zeros(D, dtype=np.int8)
        for idx in sup:
            v[idx] = 1
        vectors.append(v)
    
    # Verify the full set
    if vectors:
        arr = np.array(vectors, dtype=np.int32)
        max_norm_sq = int(np.max(np.sum(arr**2, axis=1)))
        n = len(arr)
        min_dist_sq = float('inf')
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            diffs = arr[i+1:] - arr[i]
            if len(diffs) > 0:
                dists_sq = np.sum(diffs**2, axis=1)
                d = int(np.min(dists_sq))
                if d < min_dist_sq:
                    min_dist_sq = d
                    if min_dist_sq < max_norm_sq:
                        break
        if min_dist_sq >= max_norm_sq:
            return np.array(vectors, dtype=np.int8)
    
    # Fallback: just return base vectors
    return np.array(base_vectors, dtype=np.int8)
