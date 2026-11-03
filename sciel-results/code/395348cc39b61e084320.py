"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit

def construct():
    t0 = time.perf_counter()
    best = None
    best_k = 0

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    def update_best(arr):
        nonlocal best, best_k
        if arr is None:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    There are exactly C(11,2)*2^2 = 220 such vectors, and they satisfy
    #    min pairwise distance (sqrt(2) for shared coordinate) == max norm (sqrt(2)).
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    update_best(base_arr)

    # ------------------------------------------------------------------
    # 2) Better approach: exploit sign symmetry.
    #    For vectors in the m=2 shell, we can identify each pair {v, -v} by its
    #    absolute support. Two vectors are "conflicting" (distance < max_norm)
    #    iff their inner product > 0 (for same-norm vectors, dist^2 = 2m - 2<u,v>,
    #    so dist >= sqrt(m) iff <u,v> <= m/2).
    #    
    #    For m=2: conflict iff <u,v> > 1, i.e., <u,v> >= 2.
    #    But <u,v> = 2 only if u = v (same vector). <u,v> = 1 if they share one
    #    coordinate with same sign. <u,v> = 0 if orthogonal or opposite signs.
    #    <u,v> = -1 if they share one coordinate with opposite sign.
    #    <u,v> = -2 if u = -v.
    #    
    #    So for m=2, the only conflict is u = v itself. All 220 vectors are
    #    mutually non-conflicting! That's why the baseline works.
    #    
    #    Now let's try to add vectors from other shells. The key constraint is:
    #    min pairwise distance >= max norm. If we mix shells with different norms,
    #    we need dist(u,v) >= max(||u||, ||v||) for all pairs.
    #    
    #    Strategy: Start with the m=2 shell (220 vectors, all norm sqrt(2)).
    #    Try to add vectors from m=3 shell (norm sqrt(3)). For a vector w with
    #    ||w||^2 = 3 and a vector v with ||v||^2 = 2, we need:
    #    ||w - v||^2 >= 3 (since max norm is sqrt(3))
    #    ||w-v||^2 = 3 + 2 - 2<w,v> >= 3  =>  <w,v> <= 1
    #    
    #    So we need <w,v> <= 1 for all v in the m=2 shell.
    #    Also for two m=3 vectors w1, w2: ||w1-w2||^2 >= 3 => <w1,w2> <= 0.
    #    
    #    Let's enumerate m=3 vectors and find a large subset satisfying these.
    # ------------------------------------------------------------------

    def gen_m3_vectors():
        """Generate all integer vectors in R^11 with L2^2 = 3."""
        out = []
        for support in itertools.combinations(range(D), 3):
            for signs in itertools.product([1, -1], repeat=3):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(support, signs):
                    v[idx] = s
                out.append(v)
        return np.array(out, dtype=np.int64)

    def gen_m2_vectors():
        """Generate all integer vectors in R^11 with L2^2 = 2."""
        out = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    out.append(v)
        return np.array(out, dtype=np.int64)

    # Get m=3 vectors
    m3_vecs = gen_m3_vectors()  # C(11,3)*8 = 165*8 = 1320 vectors
    m2_vecs = base_arr  # 220 vectors

    # For each m=3 vector, compute max inner product with any m=2 vector
    # <w,v> for w in m3, v in m2
    # We need max_v <w,v> <= 1
    # 
    # <w,v> = sum of w_i * v_i over coordinates where both nonzero
    # w has 3 nonzero entries (all +/-1), v has 2 nonzero entries (all +/-1)
    # So <w,v> can be: 2 (if v's support subset of w's support and signs match),
    #                  1 (if one coordinate shared, signs match),
    #                  0 (no shared coords, or one match one mismatch),
    #                  -1 (one shared, signs mismatch),
    #                  -2 (v's support subset of w's support, signs opposite)
    #
    # max <w,v> <= 1 means: no v in m2 has both coordinates in w's support with matching signs.
    # i.e., for w with support {a,b,c} and signs (sa,sb,sc), there should be no v
    #    with support {a,b} and signs (sa,sb), or {a,c} and (sa,sc), or {b,c} and (sb,sc).
    #
    # But wait - v can have ANY signs. So for w with support {a,b,c}, the m=2 vectors
    # with support {a,b} are: (sa,sb), (sa,-sb), (-sa,sb), (-sa,-sb).
    # <w,v> for v=(sa,sb) is sa*sa + sb*sb = 2. This is > 1, so conflict!
    #
    # So actually, for ANY m=3 vector w with support {a,b,c}, the m=2 vector
    # v with support {a,b} and matching signs gives <w,v> = 2 > 1. Conflict!
    #
    # This means NO m=3 vector can coexist with the full m=2 shell!
    # 
    # Let me reconsider. The condition is dist >= max_norm. If we have vectors
    # with different norms, the max norm is the largest. So if we add an m=3 vector,
    # max norm becomes sqrt(3). Then ALL pairs must have distance >= sqrt(3).
    # For two m=2 vectors: dist^2 = 4 - 2<u,v>. Need 4-2<u,v> >= 3, i.e., <u,v> <= 0.5,
    # so <u,v> <= 0. But many m=2 pairs have <u,v> = 1 (share one coord, same sign).
    # Those would have dist^2 = 4-2 = 2 < 3. Conflict!
    #
    # So mixing shells is very restrictive. Let's stick with pure shells.
    #
    # For a pure m=3 shell: need <u,v> <= 0 for all pairs (dist^2 = 6-2<u,v> >= 3).
    # So we need a code where all pairwise inner products <= 0.
    # This is a "spherical code" type problem.
    #
    # Let's try a greedy approach for m=3 with constraint <u,v> <= 0.

    def greedy_m3():
        """Greedy: select m=3 vectors with pairwise inner product <= 0."""
        n = len(m3_vecs)
        # Compute Gram matrix
        G = m3_vecs @ m3_vecs.T  # shape (1320, 1320)
        
        chosen = []
        chosen_set = set()
        remaining = set(range(n))
        
        # Start with vector 0
        chosen.append(0)
        chosen_set.add(0)
        remaining.discard(0)
        
        while remaining and not is_deadline():
            # Find candidate with minimum max inner product with chosen, subject to max_ip <= 0
            best_c = -1
            best_score = 999
            for c in remaining:
                max_ip = 0
                for j in chosen:
                    ip = G[c, j]
                    if ip > 0:
                        max_ip = 999
                        break
                    if ip < max_ip:
                        max_ip = ip
                if max_ip <= 0 and max_ip < best_score:
                    best_score = max_ip
                    best_c = c
            if best_c == -1:
                break
            chosen.append(best_c)
            chosen_set.add(best_c)
            remaining.discard(best_c)
        
        return m3_vecs[chosen] if chosen else np.zeros((0, D), dtype=np.int64)

    try:
        m3_sub = greedy_m3()
        update_best(m3_sub)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) Try m=4 shell: vectors with L2^2 = 4.
    #    Types: (2,0,...,0) [11*2=22], (1,1,1,1,0,...,0) [C(11,4)*16=5280]
    #    For pure m=4: need <u,v> <= 2 for all pairs (dist^2 = 8-2<u,v> >= 4).
    #    Greedy approach.
    # ------------------------------------------------------------------
    def gen_m4_vectors():
        out = []
        # Type 1: single coordinate with value 2 or -2
        for i in range(D):
            for s in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s
                out.append(v)
        # Type 2: four coordinates with value 1 or -1
        for support in itertools.combinations(range(D), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(support, signs):
                    v[idx] = s
                out.append(v)
        return np.array(out, dtype=np.int64)

    def greedy_m4():
        """Greedy: select m=4 vectors with pairwise inner product <= 2."""
        m4_vecs = gen_m4_vectors()
        n = len(m4_vecs)
        if n > 6000:  # too many for full Gram
            return np.zeros((0, D), dtype=np.int64)
        
        G = m4_vecs @ m4_vecs.T
        
        chosen = []
        chosen_set = set()
        remaining = set(range(n))
        
        chosen.append(0)
        chosen_set.add(0)
        remaining.discard(0)
        
        while remaining and not is_deadline():
            best_c = -1
            best_score = 999
            for c in remaining:
                max_ip = -999
                for j in chosen:
                    ip = G[c, j]
                    if ip > 2:
                        max_ip = 999
                        break
                    if ip > max_ip:
                        max_ip = ip
                if max_ip <= 2 and max_ip < best_score:
                    best_score = max_ip
                    best_c = c
            if best_c == -1:
                break
            chosen.append(best_c)
            chosen_set.add(best_c)
            remaining.discard(best_c)
        
        return m4_vecs[chosen] if chosen else np.zeros((0, D), dtype=np.int64)

    try:
        m4_sub = greedy_m4()
        update_best(m4_sub)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) Try m=1 shell: vectors with L2^2 = 1, i.e., unit basis vectors with signs.
    #    22 vectors. Need <u,v> <= 0 for all pairs. But <e_i, e_j> = 0 for i != j,
    #    and <e_i, -e_i> = -1. So all 22 work! But 22 < 220, so not better.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 5) Safety: return the best found.
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
