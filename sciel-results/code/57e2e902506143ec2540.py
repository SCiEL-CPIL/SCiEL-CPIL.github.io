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

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    best = None
    best_k = 0

    def update_best(arr):
        nonlocal best, best_k
        if arr is None or arr.shape[0] == 0:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    There are exactly C(11,2)*2^2 = 220 such vectors.
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
    # 3) Multi-order greedy: try several random permutations
    # ------------------------------------------------------------------
    def multi_order_greedy(V, max_ip_allowed, n_trials=30, time_limit=40.0):
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
            elif trial == 1:
                order = list(range(n-1, -1, -1))
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
    # 4) CSP backtracking with forward checking and most-constrained-variable
    #    heuristic. This explores the solution space more systematically
    #    than greedy, escaping local optima.
    # ------------------------------------------------------------------
    def csp_backtracking(V, max_ip_allowed, time_limit=30.0):
        """CSP backtracking search for maximum independent set.
        V: (n, D) int array, all vectors have same norm^2.
        max_ip_allowed: max inner product allowed between any two selected vectors.
        Returns best set of indices found within time_limit.
        """
        n = V.shape[0]
        if n == 0:
            return []
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        # Precompute: for each vector i, the set of vectors j (j != i) that are
        # "conflicting" with i, i.e., G[i,j] > max_ip_allowed
        # conflict_mask[i] is a boolean array of length n
        conflict_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            conflict_mask[i] = G[i] > max_ip_allowed
            conflict_mask[i, i] = False  # no self-conflict
        
        # Precompute degrees: number of conflicts for each vector
        degrees = conflict_mask.sum(axis=1)
        
        # We'll use a recursive backtracking with forward checking.
        # State: chosen (list of indices), available (set of indices not yet decided
        #        and not conflicting with chosen)
        # Heuristic: pick the available vector with highest degree (most constrained)
        
        best_found = []
        
        # Use iterative deepening or just a single DFS with time limit
        # For efficiency, we'll do a DFS that tries to maximize the set size
        
        # Available initially is all vectors
        available = set(range(n))
        chosen = []
        
        # Order candidates by degree (descending) for initial exploration
        # We'll use a stack-based DFS with backtracking
        
        def dfs(available, chosen, depth):
            nonlocal best_found
            
            if is_deadline():
                return
            
            if len(chosen) > len(best_found):
                best_found = chosen[:]
            
            if not available:
                return
            
            # Forward checking: if no available vector, we're done
            # Pick the most constrained available vector (highest degree among available)
            # But we also want to try adding it first (greedy preference)
            
            # For efficiency, if we have many available vectors, we can prune:
            # If len(available) + len(chosen) <= len(best_found), prune
            if len(available) + len(chosen) <= len(best_found):
                return
            
            # Pick the vector with highest degree in available
            # This is the "most-constrained-variable" heuristic
            best_idx = None
            best_deg = -1
            for idx in available:
                deg = degrees[idx]
                if deg > best_deg:
                    best_deg = deg
                    best_idx = idx
            
            if best_idx is None:
                return
            
            # Option 1: Include best_idx
            new_chosen = chosen + [best_idx]
            new_available = set()
            for idx in available:
                if idx != best_idx and not conflict_mask[best_idx, idx]:
                    new_available.add(idx)
            
            dfs(new_available, new_chosen, depth + 1)
            
            if is_deadline():
                return
            
            # Option 2: Exclude best_idx
            available_without = available - {best_idx}
            dfs(available_without, chosen, depth + 1)
        
        # Start DFS
        dfs(available, chosen, 0)
        
        return best_found

    # ------------------------------------------------------------------
    # 5) Local search improvement: swap-based refinement
    # ------------------------------------------------------------------
    def local_search_improve(V, max_ip_allowed, time_limit=20.0):
        """Improve a set using 1-swap moves.
        V: (n, D) int array.
        Returns improved set of indices.
        """
        n = V.shape[0]
        if n == 0:
            return []
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        # Start with a greedy solution
        rng = np.random.RandomState(123)
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
        
        if not chosen:
            return []
        
        chosen_set = set(chosen)
        current_size = len(chosen)
        
        # Try 1-swap moves: remove one chosen, add one unchosen
        improved = True
        while improved and not is_deadline() and elapsed() <= time_limit:
            improved = False
            # Recompute max_ip_with_chosen for current chosen set
            max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
            for c in chosen:
                max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[c])
            max_ip_with_chosen[chosen_mask] = -999999
            
            # For each chosen vector c, try removing it and adding the best unchosen
            for c in list(chosen):
                if is_deadline() or elapsed() > time_limit:
                    break
                
                # Temporarily remove c
                chosen_set.remove(c)
                chosen_mask[c] = False
                chosen.remove(c)
                
                # Recompute max_ip_with_chosen without c
                max_ip_without_c = np.full(n, -999999, dtype=np.int64)
                for c2 in chosen:
                    max_ip_without_c = np.maximum(max_ip_without_c, G[c2])
                max_ip_without_c[chosen_mask] = -999999
                
                # Find the best unchosen vector to add
                candidates = []
                for idx in range(n):
                    if not chosen_mask[idx] and max_ip_without_c[idx] <= max_ip_allowed:
                        # Count how many other unchosen vectors this would conflict with
                        # (i.e., how many unchosen vectors have G[idx, j] > max_ip_allowed)
                        conflict_count = 0
                        for j in range(n):
                            if not chosen_mask[j] and j != idx and G[idx, j] > max_ip_allowed:
                                conflict_count += 1
                        candidates.append((conflict_count, idx))
                
                if candidates:
                    # Pick the one with fewest conflicts
                    candidates.sort()
                    best_idx = candidates[0][1]
                    
                    # Add it
                    chosen.append(best_idx)
                    chosen_mask[best_idx] = True
                    chosen_set.add(best_idx)
                    
                    if len(chosen) > current_size:
                        current_size = len(chosen)
                        improved = True
                    else:
                        # Same size, keep the swap if it reduces future conflicts
                        # For simplicity, keep it
                        improved = True
                else:
                    # Can't add anything, put c back
                    chosen.append(c)
                    chosen_mask[c] = True
                    chosen_set.add(c)
        
        return chosen

    # ------------------------------------------------------------------
    # 6) Try each shell separately with both greedy and CSP
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
            
            # Multi-order greedy
            result_greedy = multi_order_greedy(V, max_ip_allowed, n_trials=30, time_limit=20.0)
            if result_greedy.shape[0] > 0:
                update_best(result_greedy)
            
            if is_deadline():
                continue
            
            # CSP backtracking (only for smaller sets to avoid explosion)
            if n <= 2000:
                t_csp = elapsed()
                csp_result = csp_backtracking(V, max_ip_allowed, time_limit=20.0)
                if csp_result and len(csp_result) > best_k:
                    update_best(V[csp_result])
            
            if is_deadline():
                continue
            
            # Local search improvement
            if n <= 5000:
                ls_result = local_search_improve(V, max_ip_allowed, time_limit=15.0)
                if ls_result and len(ls_result) > best_k:
                    update_best(V[ls_result])
                    
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 7) Safety: return best found
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
