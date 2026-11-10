import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points where min pairwise
    squared distance >= max squared norm. Uses conflict graph + greedy independent
    set on vectors of norm 1 and norm 2 (22 + 220 = 242 candidates).
    """
    d = 11
    start = time.time()
    deadline = start + 575  # safety margin

    # Build candidate set: all vectors with exactly 1 or 2 non-zero entries, each ±1
    # Norm-1: 22 vectors (±e_i)
    # Norm-2: 220 vectors (±e_i ± e_j)
    candidates = []
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        v[i] = -1
        candidates.append(v.copy())
    for i, j in itertools.combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)

    # Norms: norm-1 vectors have norm²=1, norm-2 vectors have norm²=2
    norms_sq = np.sum(candidates * candidates, axis=1)

    # For the condition min_dist² >= max_norm²:
    # If we include any norm-1 vector, max_norm² >= 1, need all dist² >= 1.
    # If we only use norm-2 vectors, max_norm² = 2, need all dist² >= 2.
    #
    # Key: two vectors conflict if their squared distance < required threshold.
    # But the threshold depends on which vectors are in the set.
    #
    # Simplification: if we restrict to norm-2 vectors only, we need dist² >= 2 for all pairs.
    # Two norm-2 vectors u,v: ||u-v||² = 4 - 2*u·v. Need 4-2*u·v >= 2, i.e. u·v <= 1.
    # u·v for two norm-2 vectors: each has two non-zero ±1 entries.
    # u·v can be -2,-1,0,1,2.
    # Conflict when u·v > 1, i.e. u·v = 2, meaning u = v.
    # Wait, u·v=2 only if u=v (since both have norm²=2 and dot=2 implies equal).
    # So ALL distinct norm-2 vectors have u·v <= 1, hence dist² >= 2.
    # This means ALL 220 norm-2 vectors are mutually valid!
    #
    # Now can we add norm-1 vectors? If we add a norm-1 vector, max_norm² becomes 2 (since norm-2
    # vectors are present). Need norm-1 vector at dist² >= 2 from all norm-2 vectors.
    # norm-1 vector e_i: dist² to norm-2 vector with entries at positions a,b:
    #   If i in {a,b}: dist² = 4 - 2*e_i·v = 4 - 2*(±1) = 2 or 6.
    #     If e_i·v = 1: dist² = 2. OK (2>=2).
    #     If e_i·v = -1: dist² = 6. OK.
    #   If i not in {a,b}: dist² = 1 + 2 = 3. OK.
    # So e_i is at dist² >= 2 from ALL norm-2 vectors!
    # Similarly -e_i.
    #
    # But also need norm-1 vectors among themselves at dist² >= 2.
    # e_i and e_j (i≠j): dist² = 2. OK.
    # e_i and -e_i: dist² = 4. OK.
    #
    # So ALL 242 vectors (22 + 220) are mutually valid with max_norm²=2!
    #
    # Let me verify this directly.

    # Verify: check all pairs
    def verify(points):
        k = len(points)
        if k < 2:
            return False
        max_norm = max(np.sum(p*p) for p in points)
        min_dist = float('inf')
        for i in range(k):
            for j in range(i+1, k):
                diff = points[i] - points[j]
                dist2 = np.sum(diff*diff)
                if dist2 < min_dist:
                    min_dist = dist2
        return min_dist >= max_norm

    # Direct verification that all 242 work
    if verify(candidates):
        points = np.around(candidates).astype(np.int64)
        return points

    # If the direct verification fails (shouldn't), fall back to conflict graph approach
    # Build conflict: two vectors conflict if dist² < 2 (the norm-2 threshold)
    # Actually, let's just do a greedy with the correct threshold.

    # Conflict graph: u and v conflict if ||u-v||² < 2 (since if max norm is 2, need dist>=2)
    # But if we drop all norm-2 vectors, max norm could be 1, need dist>=1.
    # For simplicity, build conflict for threshold 2 (safe for any subset containing norm-2 vectors)

    # Compute pairwise squared distances
    # n=242, so n²/2 = ~29k pairs, very fast
    diff_mat = candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    dist_sq = np.sum(diff_mat * diff_mat, axis=2)
    np.fill_diagonal(dist_sq, np.inf)

    # Conflict: dist_sq < 2 (need dist² >= 2 if max_norm² = 2)
    conflict = dist_sq < 2

    # Greedy: pick vertices in order of fewest conflicts first
    degrees = np.sum(conflict, axis=1)

    best_set = []
    best_k = 0

    # Multiple random orders
    rng = np.random.default_rng(123)
    for trial in range(200):
        if time.time() > deadline:
            break
        if trial == 0:
            order = np.argsort(degrees)
        else:
            # Perturb: random shuffle
            order = rng.permutation(n)

        available = np.ones(n, dtype=bool)
        selected = []

        for idx in order:
            if not available[idx]:
                continue
            # Check if adding idx keeps all selected points at dist >= 2 from idx
            # Since conflict already encodes dist<2, check conflicts with selected
            sel_arr = np.array(selected) if selected else np.empty(0)
            if len(sel_arr) == 0:
                selected.append(idx)
                available[idx] = False
                # Remove neighbors
                nbrs = np.where(conflict[idx] & available)[0]
                available[nbrs] = False
            else:
                # All selected should not conflict with idx
                # Since we removed neighbors of each selected when we picked them,
                # if idx is still available, it doesn't conflict with any selected.
                selected.append(idx)
                available[idx] = False
                nbrs = np.where(conflict[idx] & available)[0]
                available[nbrs] = False

        k = len(selected)
        if k > best_k:
            best_k = k
            best_set = list(selected)

        if best_k >= n:
            break

    if best_set:
        points = candidates[best_set]
        points = np.around(points).astype(np.int64)
    else:
        # Fallback: all norm-2 vectors
        points = candidates[22:]  # norm-2 vectors start at index 22
        points = np.around(points).astype(np.int64)

    return points
