"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 140.0  # hard wall-clock safety margin before the 150 s limit


def construct():
    t0 = time.perf_counter()

    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

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
    best = base_arr
    best_k = base_arr.shape[0]

    def update_best(arr):
        nonlocal best, best_k
        if arr is None or arr.shape[0] == 0:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 2) Efficient vector generation for small shells
    # ------------------------------------------------------------------

    def gen_shell_vectors(m):
        """Generate all integer vectors in R^D with L2^2 exactly = m, values in {-1,0,1}."""
        out = []
        for support in itertools.combinations(range(D), m):
            # For each support of size m, assign ±1 to each
            for signs in itertools.product([-1, 1], repeat=m):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(support, signs):
                    v[idx] = s
                out.append(v)
        return np.array(out, dtype=np.int64) if out else np.zeros((0, D), dtype=np.int64)

    # Pre-generate shells
    shell2 = base_arr  # m=2
    shell3 = None
    shell4 = None

    try:
        if not is_deadline():
            shell3 = gen_shell_vectors(3)
    except Exception:
        pass

    try:
        if not is_deadline():
            shell4 = gen_shell_vectors(4)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) Efficient greedy with multiple random restarts
    # ------------------------------------------------------------------

    def greedy_subset(vectors, m, max_restarts=5, seed=42):
        """Given integer vectors all of L2^2 == m, pick a maximal subset where
        pairwise distance^2 >= m, i.e., inner product <= floor(m/2)."""
        max_ip = m // 2
        V = np.asarray(vectors, dtype=np.int64)
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)

        # Compute Gram
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)

        best_result = None
        best_count = 0

        rng = np.random.RandomState(seed)

        for restart in range(max_restarts):
            if is_deadline():
                break

            # Random ordering for this restart
            if restart == 0:
                order = np.arange(n)
            else:
                order = rng.permutation(n)

            chosen = []
            chosen_mask = np.zeros(n, dtype=bool)
            # Track max inner product with chosen for each candidate
            max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
            valid_candidates = set(range(n))

            for idx in order:
                if is_deadline():
                    break
                if chosen_mask[idx]:
                    continue
                # Check if idx can be added
                if max_ip_with_chosen[idx] <= max_ip:
                    chosen.append(int(idx))
                    chosen_mask[idx] = True
                    valid_candidates.discard(idx)
                    # Update max_ip_with_chosen for all remaining candidates
                    for j in valid_candidates:
                        ip = G[idx, j]
                        if ip > max_ip_with_chosen[j]:
                            max_ip_with_chosen[j] = ip
                            if max_ip_with_chosen[j] > max_ip:
                                valid_candidates.discard(j)

            if len(chosen) > best_count:
                best_count = len(chosen)
                best_result = V[chosen]

        if best_result is None:
            return np.zeros((0, D), dtype=np.int64)
        return best_result

    # Try m=3 shell
    try:
        if shell3 is not None and len(shell3) > 0 and not is_deadline():
            sub3 = greedy_subset(shell3, 3, max_restarts=8)
            update_best(sub3)
    except Exception:
        pass

    # Try m=4 shell
    try:
        if shell4 is not None and len(shell4) > 0 and not is_deadline():
            sub4 = greedy_subset(shell4, 4, max_restarts=8)
            update_best(sub4)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) Mixed-norm greedy: combine vectors from different shells
    # ------------------------------------------------------------------

    def mixed_greedy(shells, max_restarts=5, seed=42):
        """Greedy subset from mixed shells.
        shells: list of (array, norm2) tuples
        """
        # Concatenate all vectors
        all_vecs = []
        all_norms = []
        for vecs, norm2 in shells:
            if vecs is not None and len(vecs) > 0:
                all_vecs.append(vecs)
                all_norms.append(norm2)
        if not all_vecs:
            return np.zeros((0, D), dtype=np.int64)

        V = np.vstack(all_vecs)
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)

        norm_arr = np.concatenate([np.full(len(v), n2) for v, n2 in zip(all_vecs, all_norms)])

        # Compute Gram
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)

        def max_allowed_ip(m1, m2):
            return int(min(m1, m2) / 2)

        best_result = None
        best_count = 0
        rng = np.random.RandomState(seed)

        for restart in range(max_restarts):
            if is_deadline():
                break

            if restart == 0:
                order = np.arange(n)
            else:
                order = rng.permutation(n)

            chosen = []
            chosen_mask = np.zeros(n, dtype=bool)
            # For each candidate j, track the minimum allowed IP over all chosen i
            min_allowed = np.array([max_allowed_ip(norm_arr[0], norm_arr[j]) for j in range(n)], dtype=np.int64)
            min_allowed[0] = -999999
            # Track max actual IP with chosen
            max_ip_with_chosen = G[0].copy()
            max_ip_with_chosen[0] = -999999
            valid_candidates = set(range(1, n))

            # First pick index 0
            chosen.append(0)
            chosen_mask[0] = True

            for idx in order[1:]:
                if is_deadline():
                    break
                if chosen_mask[idx] or idx not in valid_candidates:
                    continue
                if max_ip_with_chosen[idx] <= min_allowed[idx]:
                    chosen.append(int(idx))
                    chosen_mask[idx] = True
                    valid_candidates.discard(idx)
                    # Update for remaining candidates
                    for j in valid_candidates:
                        ip = G[idx, j]
                        if ip > max_ip_with_chosen[j]:
                            max_ip_with_chosen[j] = ip
                        allowed = max_allowed_ip(norm_arr[idx], norm_arr[j])
                        if allowed < min_allowed[j]:
                            min_allowed[j] = allowed
                        if max_ip_with_chosen[j] > min_allowed[j]:
                            valid_candidates.discard(j)

            if len(chosen) > best_count:
                best_count = len(chosen)
                best_result = V[chosen]

        if best_result is None:
            return np.zeros((0, D), dtype=np.int64)
        return best_result

    # Try mixed m=2 and m=3
    try:
        if shell3 is not None and len(shell3) > 0 and not is_deadline():
            mixed_23 = mixed_greedy([(shell2, 2), (shell3, 3)], max_restarts=10)
            update_best(mixed_23)
    except Exception:
        pass

    # Try mixed m=2 and m=4
    try:
        if shell4 is not None and len(shell4) > 0 and not is_deadline():
            mixed_24 = mixed_greedy([(shell2, 2), (shell4, 4)], max_restarts=10)
            update_best(mixed_24)
    except Exception:
        pass

    # Try mixed m=2, m=3, m=4
    try:
        if shell3 is not None and shell4 is not None and len(shell3) > 0 and len(shell4) > 0 and not is_deadline():
            mixed_234 = mixed_greedy([(shell2, 2), (shell3, 3), (shell4, 4)], max_restarts=8)
            update_best(mixed_234)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5) Local search: try to improve the best solution by adding more vectors
    # ------------------------------------------------------------------

    def local_search_improve(solution, all_vectors, max_iterations=500):
        """Try to add more vectors to the solution or swap to improve."""
        if solution is None or len(solution) == 0:
            return solution

        V_sol = np.asarray(solution, dtype=np.int64)
        k = V_sol.shape[0]
        n_all = all_vectors.shape[0]

        # Compute norms of solution vectors
        norms_sol = np.sum(V_sol ** 2, axis=1)
        max_norm2 = np.max(norms_sol)

        # Compute Gram of solution with all vectors
        G_all = all_vectors.astype(np.int64) @ V_sol.astype(np.int64).T

        # For each candidate, check if it can be added
        # Condition: for all i in solution, dist^2(candidate, V_sol[i]) >= max_norm2
        # dist^2 = ||c||^2 + ||V_sol[i]||^2 - 2*c.V_sol[i] >= max_norm2
        # c.V_sol[i] <= (||c||^2 + ||V_sol[i]||^2 - max_norm2) / 2

        added = list(range(k))
        improved = False

        for _ in range(max_iterations):
            if is_deadline():
                break

            # Find candidates that can be added
            best_add = -1
            best_score = -1

            for j in range(n_all):
                if j in added:
                    continue
                c = all_vectors[j]
                c_norm2 = np.sum(c ** 2)
                # Check against all current solution vectors
                can_add = True
                min_margin = float('inf')
                for i in range(len(added)):
                    sol_i = V_sol[added[i]]
                    sol_norm2 = norms_sol[added[i]]
                    ip = int(np.dot(c, sol_i))
                    # Need: c_norm2 + sol_norm2 - 2*ip >= max(max_norm2, c_norm2, sol_norm2)
                    # Actually the condition is dist >= max(||c||, ||sol_i||)
                    # dist^2 = c_norm2 + sol_norm2 - 2*ip >= max(c_norm2, sol_norm2)
                    # So: ip <= (c_norm2 + sol_norm2 - max(c_norm2, sol_norm2)) / 2 = min(c_norm2, sol_norm2) / 2
                    allowed_ip = min(c_norm2, sol_norm2) // 2
                    if ip > allowed_ip:
                        can_add = False
                        break
                    margin = allowed_ip - ip
                    if margin < min_margin:
                        min_margin = margin

                if can_add:
                    score = -min_margin  # prefer vectors with most margin
                    if score > best_score:
                        best_score = score
                        best_add = j

            if best_add < 0:
                break

            # Add the vector
            added.append(best_add)
            V_sol = np.vstack([V_sol, all_vectors[best_add:best_add+1]])
            norms_sol = np.append(norms_sol, np.sum(all_vectors[best_add] ** 2))
            max_norm2 = np.max(norms_sol)
            # Recompute G_all for the new solution
            G_all = all_vectors.astype(np.int64) @ V_sol.astype(np.int64).T
            improved = True

        if not improved:
            return solution
        return V_sol

    # Try to improve the best solution
    try:
        if not is_deadline():
            # Use all vectors from shells 2, 3, 4
            all_vecs_list = [shell2]
            if shell3 is not None and len(shell3) > 0:
                all_vecs_list.append(shell3)
            if shell4 is not None and len(shell4) > 0:
                all_vecs_list.append(shell4)
            all_vecs = np.vstack(all_vecs_list)

            improved = local_search_improve(best, all_vecs, max_iterations=200)
            update_best(improved)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 6) Safety: return the best found.
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
