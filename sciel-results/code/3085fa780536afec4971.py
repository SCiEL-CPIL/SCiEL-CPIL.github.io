"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 145.0  # hard wall-clock safety margin before the 150 s limit

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
        """Generate all integer vectors in R^D with L2^2 = m."""
        if m < 1:
            return []
        
        results = []
        
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
    # 3) Weighted greedy: pick vectors with lowest conflict first
    # ------------------------------------------------------------------
    def weighted_greedy(V, max_ip_allowed, time_limit=20.0):
        """Weighted greedy selection. At each step, pick the vector with the 
        lowest 'conflict score' among those that can be added."""
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        chosen = []
        chosen_mask = np.zeros(n, dtype=bool)
        max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
        
        t_start = now()
        
        while len(chosen) < n:
            if is_deadline() or (now() - t_start) > time_limit:
                break
            
            # Find candidates that can be added
            candidates = np.where((~chosen_mask) & (max_ip_with_chosen <= max_ip_allowed))[0]
            if len(candidates) == 0:
                break
            
            # Compute conflict score for each candidate:
            # Number of already-chosen vectors that have high IP with this candidate
            # Or: max IP with chosen (lower is better)
            # Let's use: sum of max(0, G[c, j] - max_ip_allowed) for j in chosen
            # Actually, let's just use the max IP with chosen as the score
            # Lower max IP = less conflict = better to pick first
            
            scores = max_ip_with_chosen[candidates]
            
            # Pick the one with lowest score (least conflict)
            best_local = np.argmin(scores)
            idx = candidates[best_local]
            
            chosen.append(idx)
            chosen_mask[idx] = True
            max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
            max_ip_with_chosen[chosen_mask] = -999999
        
        if not chosen:
            return np.zeros((0, D), dtype=np.int64)
        return V[chosen]

    # ------------------------------------------------------------------
    # 4) Multi-order greedy with random restarts
    # ------------------------------------------------------------------
    def multi_order_greedy(V, max_ip_allowed, n_trials=50, time_limit=30.0):
        """Try greedy with multiple random orderings, keep best."""
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        best_chosen = []
        best_len = 0
        
        rng = np.random.RandomState(42)
        
        t_start = now()
        
        for trial in range(n_trials):
            if is_deadline() or (now() - t_start) > time_limit:
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
                if is_deadline() or (now() - t_start) > time_limit:
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
    # 5) Local search: try to improve a solution by swapping
    # ------------------------------------------------------------------
    def local_search_improve(V_sub, max_ip_allowed, time_limit=10.0):
        """Try to improve the solution by removing one vector and adding another."""
        n_sub = V_sub.shape[0]
        if n_sub == 0:
            return V_sub
        
        # Compute Gram for the current solution
        G_sub = V_sub.astype(np.int64) @ V_sub.astype(np.int64).T
        G_sub = np.ascontiguousarray(G_sub)
        
        # For each vector in the solution, compute how many others it conflicts with
        # Conflict: G[i,j] > max_ip_allowed
        conflict_count = np.sum(G_sub > max_ip_allowed, axis=1)
        
        # Try removing the vector with highest conflict count and see if we can add more
        t_start = now()
        best_solution = V_sub.copy()
        best_len = n_sub
        
        # Generate all candidate vectors (we need the full set)
        # This is expensive, so we'll skip for now and just do simple swaps within the set
        
        return best_solution

    # ------------------------------------------------------------------
    # 6) Mixed-norm construction: combine vectors from different shells
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
        # i.e., min(M[i], M[j]) - 2*G[i,j] >= 0
        # i.e., G[i,j] <= min(M[i], M[j]) // 2
        
        # Track max violation for each candidate
        # violation[i] = max over j in chosen of (G[i,j] - min(M[i], M[j])//2)
        
        # Precompute ip_allowed
        ip_allowed = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i+1, n):
                val = min(M[i], M[j]) // 2
                ip_allowed[i,j] = val
                ip_allowed[j,i] = val
            ip_allowed[i,i] = -999999
        
        max_violation = np.full(n, -999999, dtype=np.int64)
        chosen = []
        chosen_mask = np.zeros(n, dtype=bool)
        
        # Sort by norm (increasing first)
        order = np.argsort(M)
        
        t_start = now()
        
        for idx in order:
            if is_deadline() or (now() - t_start) > time_limit:
                break
            if chosen_mask[idx]:
                continue
            if max_violation[idx] <= 0:
                chosen.append(idx)
                chosen_mask[idx] = True
                max_violation = np.maximum(max_violation, G[:, idx] - ip_allowed[:, idx])
                max_violation[chosen_mask] = -999999
        
        if not chosen:
            return np.zeros((0, D), dtype=np.int64)
        return V[chosen]

    # ------------------------------------------------------------------
    # 7) Try each shell separately with multi-order greedy
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
            n_trials = min(50, max(10, 100 // (n // 1000)))
            
            for trial in range(n_trials):
                if is_deadline() or elapsed() > t_start + 30:
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
                    if is_deadline() or elapsed() > t_start + 30:
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
    # 8) Weighted greedy on each shell
    # ------------------------------------------------------------------
    for m in [2, 3, 4, 5]:
        if is_deadline():
            break
        
        try:
            vecs = gen_all_vectors_m(m)
            if len(vecs) == 0:
                continue
            
            V = np.array(vecs, dtype=np.int64)
            max_ip_allowed = m // 2
            
            result = weighted_greedy(V, max_ip_allowed, time_limit=15.0)
            update_best(result)
            
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 9) Mixed-norm: combine vectors from m=2,3,4,5 shells
    # ------------------------------------------------------------------
    if not is_deadline():
        try:
            all_vecs = []
            for m in [2, 3, 4, 5]:
                vecs = gen_all_vectors_m(m)
                for v in vecs:
                    all_vecs.append((v, m))
            
            if len(all_vecs) > 0:
                result = mixed_greedy(all_vecs, [m for v, m in all_vecs], time_limit=25.0)
                update_best(result)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 10) Safety: return best found
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
