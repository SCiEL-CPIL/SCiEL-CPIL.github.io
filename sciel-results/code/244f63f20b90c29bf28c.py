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
    # 2) Efficient generation of all vectors with norm^2 = m using combinatorics
    # ------------------------------------------------------------------
    def gen_all_vectors_m(m):
        """Generate all integer vectors in R^D with L2^2 = m.
        For m=2: C(D,2)*4 + D*2 (one coord is ±2) = 220+22=242... wait no.
        Actually for m=2: either two coords are ±1, or one coord is ±2.
        Let's be precise:
        m=2: patterns: (1,1,0,...) with 2 nonzero ±1 coords, or (2,0,...) with one ±2 coord.
        m=3: (1,1,1,0,...) three nonzero ±1, or (1,1,0,...) no wait that's m=2.
        For m=3: only (1,1,1,0,...) since 3 = 1+1+1, can't have a 2 (4>3).
        m=4: (2,0,...) one ±2, (1,1,1,1,0,...) four ±1, (1,1,0,...) no that's m=2.
              Actually: 4 = 4 (one coord ±2), or 1+1+1+1 (four coords ±1), or 1+3? No, 3 not a square.
              So m=4: one coord ±2 (D*2=22) or four coords ±1 (C(D,4)*16=330*16=5280). Total=5302.
        m=5: (1,1,1,1,1,0,...) five ±1 (C(11,5)*32=462*32=14784), or (1,2,0,...) one ±1 and one ±2 (C(11,2)*4=66*4=264). Total=15048.
        """
        if m < 1:
            return []
        
        results = []
        
        # Generate all compositions of m as sum of squares
        # Squares available: 1, 4, 9 (since D=11, max coord value with m<=5 is 2)
        
        if m == 2:
            # Two ±1 coords
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
            # One ±2 coord
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        
        elif m == 3:
            # Three ±1 coords
            for combo in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
        
        elif m == 4:
            # Four ±1 coords
            for combo in itertools.combinations(range(D), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            # One ±2 coord
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        
        elif m == 5:
            # Five ±1 coords
            for combo in itertools.combinations(range(D), 5):
                for signs in itertools.product([1, -1], repeat=5):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            # One ±1 and one ±2 coord
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (2, -2):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
        
        elif m == 6:
            # Six ±1 coords
            for combo in itertools.combinations(range(D), 6):
                for signs in itertools.product([1, -1], repeat=6):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            # One ±2 and two ±1 coords
            for combo in itertools.combinations(range(D), 3):
                for signs2 in (2, -2):
                    for signs1 in itertools.product([1, -1], repeat=2):
                        v = np.zeros(D, dtype=np.int64)
                        v[combo[0]] = signs2
                        v[combo[1]] = signs1[0]
                        v[combo[2]] = signs1[1]
                        results.append(v)
        
        return results

    # ------------------------------------------------------------------
    # 3) Fast greedy with multiple orderings
    # ------------------------------------------------------------------
    def greedy_fast(V, max_ip_allowed, order=None):
        """Greedy selection with given order. V is (n, D) int array.
        Returns indices of selected vectors."""
        n = V.shape[0]
        if n == 0:
            return []
        
        if order is None:
            order = list(range(n))
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        chosen = []
        chosen_set = set()
        max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
        
        for idx in order:
            if is_deadline():
                break
            if idx in chosen_set:
                continue
            if max_ip_with_chosen[idx] <= max_ip_allowed:
                chosen.append(idx)
                chosen_set.add(idx)
                max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
                max_ip_with_chosen[chosen_set.__iter__().__next__() if False else list(chosen_set)] = -999999
                # More efficient: just set the chosen ones
                for c in chosen:
                    max_ip_with_chosen[c] = -999999
        
        return chosen

    def greedy_with_order(V, max_ip_allowed, order):
        """Greedy selection with given order. Returns (indices, V_sub)."""
        n = V.shape[0]
        if n == 0:
            return [], np.zeros((0, D), dtype=np.int64)
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        chosen = []
        chosen_mask = np.zeros(n, dtype=bool)
        max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
        
        for idx in order:
            if is_deadline():
                break
            if chosen_mask[idx]:
                continue
            if max_ip_with_chosen[idx] <= max_ip_allowed:
                chosen.append(idx)
                chosen_mask[idx] = True
                max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
                max_ip_with_chosen[chosen_mask] = -999999
        
        if not chosen:
            return [], np.zeros((0, D), dtype=np.int64)
        return chosen, V[chosen]

    # ------------------------------------------------------------------
    # 4) Multi-order greedy: try several random permutations
    # ------------------------------------------------------------------
    def multi_order_greedy(V, max_ip_allowed, n_trials=20, time_limit=20.0):
        """Try greedy with multiple random orderings, keep best."""
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        best_chosen = []
        best_len = 0
        
        rng = np.random.RandomState(42)
        
        for trial in range(n_trials):
            if is_deadline() or elapsed() > time_limit:
                break
            
            if trial == 0:
                order = list(range(n))
            else:
                order = rng.permutation(n).tolist()
            
            chosen_mask = np.zeros(n, dtype=bool)
            max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
            chosen = []
            
            for idx in order:
                if is_deadline() or elapsed() > time_limit:
                    break
                if chosen_mask[idx]:
                    continue
                if max_ip_with_chosen[idx] <= max_ip_allowed:
                    chosen.append(idx)
                    chosen_mask[idx] = True
                    max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
                    max_ip_with_chosen[chosen_mask] = -999999
            
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_chosen = chosen
        
        if not best_chosen:
            return np.zeros((0, D), dtype=np.int64)
        return V[best_chosen]

    # ------------------------------------------------------------------
    # 5) Mixed-norm construction: combine vectors from different shells
    # ------------------------------------------------------------------
    def mixed_greedy(all_vectors, norms, time_limit=30.0):
        """all_vectors: list of (vector, norm_sq) pairs.
        Greedily add vectors maintaining min_dist >= max_norm."""
        n = len(all_vectors)
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        
        V = np.array([v for v, m in all_vectors], dtype=np.int64)
        M = np.array([m for v, m in all_vectors], dtype=np.int64)
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        # For vectors i and j with norms^2 = M[i], M[j]:
        # dist^2 = M[i] + M[j] - 2*G[i,j]
        # We need dist^2 >= max(M[i], M[j])
        # i.e., M[i] + M[j] - 2*G[i,j] >= max(M[i], M[j])
        # i.e., min(M[i], M[j]) - 2*G[i,j] >= 0
        # i.e., G[i,j] <= min(M[i], M[j]) // 2
        
        # For a set S, the constraint for adding vector v is:
        # For all u in S: G[v,u] <= min(M[v], M[u]) // 2
        # And the new max_norm is max(max_old_norm, M[v])
        # And for all pairs in S: already satisfied
        
        # Actually we need to check: after adding v, for all pairs (u1, u2) in S∪{v}:
        # dist(u1,u2)^2 >= max(M[u1], M[u2])
        # Pairs in S are already ok. Pairs involving v need checking.
        
        max_norm_sq = 0
        chosen = []
        chosen_mask = np.zeros(n, dtype=bool)
        
        # Precompute: for each pair, the max IP allowed
        # ip_allowed[i,j] = min(M[i], M[j]) // 2
        
        # We'll track for each candidate the max violation with chosen set
        # violation[i] = max over j in chosen of (G[i,j] - min(M[i], M[j])//2)
        # We can add i if violation[i] <= 0
        
        ip_allowed = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i+1, n):
                val = min(M[i], M[j]) // 2
                ip_allowed[i,j] = val
                ip_allowed[j,i] = val
            ip_allowed[i,i] = -999999
        
        # Track max IP with chosen for each candidate
        # Actually we need: G[i,j] <= ip_allowed[i,j] for all j in chosen
        # Equivalent to: max over j in chosen of (G[i,j] - ip_allowed[i,j]) <= 0
        
        # Let's track: for each i, max_violation[i] = max over j in chosen of (G[i,j] - ip_allowed[i,j])
        max_violation = np.full(n, -999999, dtype=np.int64)
        
        # Sort candidates by norm (try larger norms first? or smaller?)
        # Let's try both: first by increasing norm, then by decreasing norm
        order1 = np.argsort(M)  # increasing norm
        order2 = np.argsort(-M)  # decreasing norm
        
        for order in [order1, order2]:
            if is_deadline() or elapsed() > time_limit:
                break
            
            chosen = []
            chosen_mask = np.zeros(n, dtype=bool)
            max_violation = np.full(n, -999999, dtype=np.int64)
            current_max_norm = 0
            
            for idx in order:
                if is_deadline() or elapsed() > time_limit:
                    break
                if chosen_mask[idx]:
                    continue
                if max_violation[idx] <= 0:
                    chosen.append(idx)
                    chosen_mask[idx] = True
                    current_max_norm = max(current_max_norm, M[idx])
                    # Update max_violation for all others
                    # For each k not chosen: max_violation[k] = max(max_violation[k], G[k,idx] - ip_allowed[k,idx])
                    max_violation = np.maximum(max_violation, G[:, idx] - ip_allowed[:, idx])
                    max_violation[chosen_mask] = -999999
            
            if len(chosen) > best_k or (best is None and len(chosen) > 0):
                if len(chosen) > best_k:
                    update_best(V[chosen])
        
        return V[chosen] if chosen else np.zeros((0, D), dtype=np.int64)

    # ------------------------------------------------------------------
    # 6) Try each shell separately with multi-order greedy
    # ------------------------------------------------------------------
    for m in [2, 3, 4, 5, 6]:
        if is_deadline():
            break
        
        t_start = elapsed()
        try:
            vecs = gen_all_vectors_m(m)
            if len(vecs) == 0:
                continue
            
            V = np.array(vecs, dtype=np.int64)
            n = V.shape[0]
            max_ip_allowed = m // 2
            
            # Compute Gram once
            G = V.astype(np.int64) @ V.astype(np.int64).T
            G = np.ascontiguousarray(G)
            
            best_chosen = []
            best_len = 0
            
            # Try multiple random orderings
            rng = np.random.RandomState(42 + m)
            n_trials = min(30, max(5, 60 // (n // 1000)))  # more trials for smaller sets
            
            for trial in range(n_trials):
                if is_deadline() or elapsed() > t_start + 40:
                    break
                
                if trial == 0:
                    order = list(range(n))
                elif trial == 1:
                    order = list(range(n-1, -1, -1))
                else:
                    order = rng.permutation(n).tolist()
                
                chosen_mask = np.zeros(n, dtype=bool)
                max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
                chosen = []
                
                for idx in order:
                    if is_deadline() or elapsed() > t_start + 40:
                        break
                    if chosen_mask[idx]:
                        continue
                    if max_ip_with_chosen[idx] <= max_ip_allowed:
                        chosen.append(idx)
                        chosen_mask[idx] = True
                        max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
                        max_ip_with_chosen[chosen_mask] = -999999
                
                if len(chosen) > best_len:
                    best_len = len(chosen)
                    best_chosen = chosen
            
            if best_chosen:
                update_best(V[best_chosen])
                
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 7) Mixed-norm: combine vectors from m=2,3,4,5 shells
    # ------------------------------------------------------------------
    if not is_deadline():
        try:
            all_vecs = []
            for m in [2, 3, 4, 5]:
                vecs = gen_all_vectors_m(m)
                for v in vecs:
                    all_vecs.append((v, m))
            
            if len(all_vecs) > 0:
                result = mixed_greedy(all_vecs, [m for v, m in all_vecs], time_limit=30.0)
                update_best(result)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 8) Safety: return best found
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
