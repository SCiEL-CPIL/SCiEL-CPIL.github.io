"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

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
    # 2) MODEL-AND-OPTIMIZE: try other fixed-norm^2 shells and a small
    #    mixed-norm extension, but only when we can finish within the budget.
    #    For a fixed-norm^2 = m shell, the condition "min distance >= max norm"
    #    is equivalent to a constant-weight code with max inner product
    #    <= floor(m/2) (since dist^2 = 2m - 2 u.v).
    #    Greedy (order by a random sign-flip heuristic) plus a couple of
    #    deterministic orderings is all we attempt; we stop well before the deadline.
    # ------------------------------------------------------------------

    def gen_weight_vectors(m, max_terms, hard_stop=30.0):
        """Yield integer vectors in R^D with L2^2 exactly = m and <= max_terms
        nonzero coordinates, values in {1,-1}, generated quickly.
        Used only for small m (3,4,5) where the set is tiny-to-moderate."""
        if m < 1:
            return []
        terms = np.zeros(D, dtype=np.int64)
        out = []
        # recursive enumeration of integer vectors with squared norm m
        # restrict support to <= max_terms for speed
        def rec(pos, sq_remaining, used_terms):
            if is_deadline() or elapsed() > hard_stop:
                return
            if sq_remaining == 0:
                out.append(terms.copy())
                return
            if pos == D:
                return
            if used_terms >= max_terms:
                # can only fill zeros here
                if sq_remaining == 0:
                    out.append(terms.copy())
                return
            for x in (-1, 0, 1):
                ns = sq_remaining - x * x
                if ns < 0:
                    continue
                # prune: we must be able to reach exactly 0 with at most (D-pos) terms of 1 or 0
                terms[pos] = x
                rec(pos + 1, ns, used_terms + (1 if x != 0 else 0))
            terms[pos] = 0

        rec(0, m, 0)
        return out

    # We'll try m = 3 and m = 4 (small shells) and a bounded greedy for each.
    def greedy_subset(vectors, m):
        """Given integer vectors all of L2^2 == m, pick a maximal (greedy)
        subset where pairwise distance^2 >= m, i.e., inner product <= floor(m/2)."""
        max_ip = m // 2  # floor
        V = np.asarray(vectors, dtype=np.int64)
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        # Compute Gram in one go
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        # Greedy with vectorized max computation
        chosen = [0]
        chosen_mask = np.zeros(n, dtype=bool)
        chosen_mask[0] = True
        # Track max inner product with chosen for each candidate
        max_ip_with_chosen = G[0].copy()  # max ip with just vector 0
        max_ip_with_chosen[0] = -999999  # exclude self
        while not is_deadline():
            # Find candidates with max_ip_with_chosen <= max_ip
            valid = max_ip_with_chosen <= max_ip
            valid[chosen_mask] = False  # exclude already chosen
            if not np.any(valid):
                break
            # Among valid, pick the one with minimum max_ip_with_chosen
            # (tie-break: pick first)
            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                break
            # Pick the one with smallest max_ip_with_chosen
            scores = max_ip_with_chosen[valid_indices]
            min_score = np.min(scores)
            best_c = valid_indices[np.argmin(scores)]
            chosen.append(int(best_c))
            chosen_mask[best_c] = True
            # Update max_ip_with_chosen: for each candidate, update with G[best_c]
            max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[best_c])
            max_ip_with_chosen[chosen_mask] = -999999  # mark chosen
        if not chosen:
            return np.zeros((0, D), dtype=np.int64)
        return V[chosen]

    def local_search_improve(vectors, m, max_iters=1000, time_limit=5.0):
        """
        Start with a greedy solution and try to improve it via local search.
        We try to swap out one vector and swap in another to improve the
        'minimum margin' (max inner product).
        """
        if elapsed() > time_limit or is_deadline():
            return None
        
        V = np.asarray(vectors, dtype=np.int64)
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        max_ip_allowed = m // 2
        
        # Get initial greedy solution
        init_sol = greedy_subset(vectors, m)
        if init_sol.shape[0] == 0:
            return init_sol
            
        # Map from vector value to index? No, indices are unique in V.
        # But greedy_subset returns V[chosen], so we need the indices.
        # Let's refactor greedy_subset to return indices or re-derive.
        # Actually, let's just run a fresh greedy that returns indices.
        
        def greedy_indices():
            chosen = [0]
            chosen_mask = np.zeros(n, dtype=bool)
            chosen_mask[0] = True
            max_ip_with_chosen = G[0].copy()
            max_ip_with_chosen[0] = -999999
            while not is_deadline() and elapsed() < time_limit:
                valid = max_ip_with_chosen <= max_ip_allowed
                valid[chosen_mask] = False
                if not np.any(valid):
                    break
                valid_indices = np.where(valid)[0]
                scores = max_ip_with_chosen[valid_indices]
                best_c = valid_indices[np.argmin(scores)]
                chosen.append(int(best_c))
                chosen_mask[best_c] = True
                max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[best_c])
                max_ip_with_chosen[chosen_mask] = -999999
            return chosen

        chosen_idx = greedy_indices()
        if not chosen_idx:
            return np.zeros((0, D), dtype=np.int64)
        
        chosen_set = set(chosen_idx)
        
        # Local search: try to replace one vector with another to reduce max IP
        # Or just try to add more? Greedy is maximal, so we can't add.
        # We can try to swap.
        # Objective: minimize the maximum inner product within the set.
        
        def max_ip_of_set(indices):
            if len(indices) < 2:
                return -999999
            idx_arr = np.array(indices)
            sub_G = G[np.ix_(idx_arr, idx_arr)]
            # Mask diagonal
            np.fill_diagonal(sub_G, -999999)
            return np.max(sub_G)
        
        current_max_ip = max_ip_of_set(chosen_idx)
        
        for _ in range(max_iters):
            if is_deadline() or elapsed() > time_limit:
                break
            
            improved = False
            # Try swapping each chosen vector with each non-chosen vector
            # This is O(k * (n-k)) which can be large.
            # k is small (maybe 50-100), n is 5000. 50*5000 = 250k checks.
            # Each check involves updating the max IP.
            
            # Optimization: Precompute max IP contribution of each chosen vector?
            # Let's just do a simple pass.
            
            chosen_list = list(chosen_idx)
            non_chosen = [i for i in range(n) if i not in chosen_set]
            
            for i_out in range(len(chosen_list)):
                v_out = chosen_list[i_out]
                
                # Calculate max IP if we remove v_out
                # This is expensive to do from scratch every time.
                # Approximation: If v_out is the "bottleneck", removing it helps.
                
                # Let's try a simpler heuristic:
                # Find the pair with max IP. Try to replace one of them with a vector
                # that has lower IP with the rest of the set.
                
                # Get the set as array
                idx_arr = np.array(chosen_idx)
                sub_G = G[np.ix_(idx_arr, idx_arr)]
                np.fill_diagonal(sub_G, -999999)
                
                # Find max IP pair
                max_val = np.max(sub_G)
                if max_val <= max_ip_allowed:
                    # Valid, but can we improve?
                    # If max_val is close to limit, try to improve.
                    pass
                else:
                    # Invalid? Greedy ensures validity. So max_val should be <= max_ip_allowed.
                    # Wait, greedy_subset ensures pairwise IP <= max_ip.
                    # So we are looking to improve the margin.
                    
                    # Find argmax
                    max_idx = np.unravel_index(np.argmax(sub_G), sub_G.shape)
                    i1, i2 = max_idx
                    v1 = idx_arr[i1]
                    v2 = idx_arr[i2]
                    
                    # Try to replace v1 or v2 with a better vector
                    # Better vector: one that has lower max IP with the rest of the set (excluding the one being removed)
                    
                    for v_removal in [v1, v2]:
                        # Rest of the set
                        rest = [v for v in chosen_idx if v != v_removal]
                        rest_arr = np.array(rest)
                        
                        # For each candidate in non_chosen, compute max IP with rest
                        # We want min of (max IP with rest)
                        # And it must be <= max_ip_allowed
                        
                        best_cand = -1
                        best_score = 999999
                        
                        # Vectorized: G[non_chosen][:, rest_arr]
                        # Max over rest for each candidate
                        if len(non_chosen) > 0 and len(rest_arr) > 0:
                            ips = G[np.ix_(non_chosen, rest_arr)]
                            max_ips = np.max(ips, axis=1)
                            
                            # Valid candidates: max_ips <= max_ip_allowed
                            valid_cands = np.where(max_ips <= max_ip_allowed)[0]
                            if len(valid_cands) > 0:
                                # Pick the one with min max_ip
                                min_score = np.min(max_ips[valid_cands])
                                if min_score < best_score:
                                    best_score = min_score
                                    # Find which candidate it is
                                    # valid_cands are indices into non_chosen
                                    idx_in_non_chosen = valid_cands[np.argmin(max_ips[valid_cands])]
                                    best_cand = non_chosen[idx_in_non_chosen]
                        
                        if best_cand != -1:
                            # Perform swap
                            chosen_idx.remove(v_removal)
                            chosen_idx.append(best_cand)
                            chosen_set.remove(v_removal)
                            chosen_set.add(best_cand)
                            non_chosen.remove(best_cand)
                            # non_chosen.append(v_removal) # v_removal becomes available
                            non_chosen.append(v_removal)
                            improved = True
                            break
                    if improved:
                        break
                        
            if not improved:
                break
                
        return V[chosen_idx]

    # --- m = 3 (weight-3) shell ---
    try:
        vecs3 = gen_weight_vectors(3, max_terms=3, hard_stop=10.0)
        if len(vecs3) > 0 and not is_deadline():
            # Try local search if time permits
            if elapsed() < 10:
                sub3 = local_search_improve(vecs3, 3, max_iters=500, time_limit=5.0)
            else:
                sub3 = greedy_subset(vecs3, 3)
            update_best(sub3)
    except Exception:
        pass

    # --- m = 4: weight-4 vectors (11 choose 4)*2^4 = 330*16=5280, plus single-2 vectors (11*2=22) = 5302 total
    # With the optimized greedy, we can handle this larger set.
    try:
        vecs4 = gen_weight_vectors(4, max_terms=4, hard_stop=15.0)
        if len(vecs4) > 0 and not is_deadline():
            # Local search on m=4 might be too slow for 5000 vectors if not careful
            # But we have ~100s left.
            if elapsed() < 50:
                sub4 = local_search_improve(vecs4, 4, max_iters=1000, time_limit=30.0)
            else:
                sub4 = greedy_subset(vecs4, 4)
            update_best(sub4)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) Safety: if we somehow exceeded the deadline, just return the baseline.
    #    Otherwise return the best found.
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
