import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses a combination of:
    1. Fast bitmask-based local search (1-add, 1-swap, 2-swap)
    2. Multi-start randomized greedy initialization
    3. Simulated annealing-style kicks to escape local optima
    4. Time-bounded search with safety margin
    """
    m = 205
    deadline = time.time() + 580  # Safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)
    n_forbid = len(forbidden_list)

    # For each element i, compute bitmask of elements that conflict with i
    # conflict_mask[i] has bit j set if (i-j) mod m is in forbidden
    # Using Python integers as bitmasks (arbitrary precision)
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden_list:
            j = (i - f) % m
            mask |= (1 << j)
            j2 = (i + f) % m
            mask |= (1 << j2)
        # Remove self
        mask &= ~(1 << i)
        conflict_mask[i] = mask

    # Precompute: for each element, list of elements it conflicts with (for faster iteration)
    conflict_list = []
    for i in range(m):
        lst = []
        mask = conflict_mask[i]
        for j in range(m):
            if mask & (1 << j):
                lst.append(j)
        conflict_list.append(lst)

    def count_conflicts_in_set(S, x):
        """Count how many elements in S conflict with x. S is a set of ints."""
        cnt = 0
        mask = conflict_mask[x]
        for s in S:
            if mask & (1 << s):
                cnt += 1
        return cnt

    def is_valid_set(S):
        """Check if set S has no forbidden pairs."""
        for a in S:
            mask = conflict_mask[a]
            for b in S:
                if a < b and (mask & (1 << b)):
                    return False
        return True

    def greedy_init(order):
        """Greedy: process candidates in given order, add if no conflict with selected."""
        S = set()
        for v in order:
            if count_conflicts_in_set(S, v) == 0:
                S.add(v)
        return S

    def local_search(S, time_limit=None, max_iters=100000):
        """
        Improve set S via:
        - 1-add: add an element if it doesn't conflict
        - 1-swap: remove one, add one (if net improvement or enables future adds)
        - 2-swap: remove two, add two (if net improvement)
        Uses bitmask for fast conflict checking.
        """
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        start_time = time.time()
        iters = 0
        no_improve_count = 0
        max_no_improve = 200

        while iters < max_iters:
            iters += 1
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            if time.time() > deadline:
                break

            improved = False

            # Try 1-add
            for cand in range(m):
                if cand not in S and count_conflicts_in_set(S, cand) == 0:
                    S.add(cand)
                    improved = True
                    break

            if improved:
                no_improve_count = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue

            # Try 1-swap: remove one, add one
            s_list = list(S)
            random.shuffle(s_list)
            swap_found = False
            for rem in s_list:
                S_without = S - {rem}
                # Find best addition
                best_cand = -1
                best_score = -1
                for cand in range(m):
                    if cand not in S_without and count_conflicts_in_set(S_without, cand) == 0:
                        # Score: prefer candidates that allow more future additions
                        S_temp = S_without | {cand}
                        # Count how many more we could add
                        future_adds = 0
                        for c2 in range(m):
                            if c2 not in S_temp and count_conflicts_in_set(S_temp, c2) == 0:
                                future_adds += 1
                        score = future_adds
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                if best_cand >= 0:
                    S = S_without | {best_cand}
                    swap_found = True
                    break

            if swap_found:
                no_improve_count = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue

            # Try 2-swap: remove two, add two
            if len(S) >= 2:
                s_list2 = list(S)
                random.shuffle(s_list2)
                two_swap_found = False
                for i in range(min(len(s_list2), 20)):
                    for j in range(i + 1, min(len(s_list2), 20)):
                        rem1, rem2 = s_list2[i], s_list2[j]
                        S_without = S - {rem1, rem2}
                        # Find best pair to add
                        best_pair = None
                        best_score = -1
                        for cand1 in range(m):
                            if cand1 not in S_without and count_conflicts_in_set(S_without, cand1) == 0:
                                for cand2 in range(m):
                                    if cand2 != cand1 and cand2 not in S_without and \
                                       count_conflicts_in_set(S_without, cand2) == 0 and \
                                       not (conflict_mask[cand1] & (1 << cand2)):
                                        S_temp = S_without | {cand1, cand2}
                                        future_adds = 0
                                        for c2 in range(m):
                                            if c2 not in S_temp and count_conflicts_in_set(S_temp, c2) == 0:
                                                future_adds += 1
                                        score = future_adds
                                        if score > best_score:
                                            best_score = score
                                            best_pair = (cand1, cand2)
                        if best_pair is not None:
                            S = S_without | set(best_pair)
                            two_swap_found = True
                            break
                    if two_swap_found:
                        break

                if two_swap_found:
                    no_improve_count = 0
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    continue

            # No improvement: kick (remove random elements and re-greedy)
            no_improve_count += 1
            if no_improve_count >= max_no_improve or random.random() < 0.1:
                # Kick: remove k random elements
                k = random.choice([1, 2, 3, 4, 5])
                if len(S) < k:
                    k = max(1, len(S) - 1)
                if k > 0 and len(S) > 0:
                    to_remove = random.sample(list(S), min(k, len(S)))
                    S = S - set(to_remove)
                    # Re-greedy fill
                    S_list = list(S)
                    random.shuffle(S_list)
                    for cand in range(m):
                        if cand not in S and count_conflicts_in_set(S, cand) == 0:
                            S.add(cand)
                    no_improve_count = 0
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)

        return best_S

    # Main search
    best_global = set()
    best_global_size = 0

    # Strategy 1: Deterministic greedy
    S_det = set()
    for v in range(m):
        if count_conflicts_in_set(S_det, v) == 0:
            S_det.add(v)
    S_det = local_search(S_det, time_limit=30.0)
    if len(S_det) > best_global_size:
        best_global = S_det
        best_global_size = len(S_det)

    # Strategy 2: Multi-start randomized greedy + local search
    rng = random.Random(42)
    n_starts = 500
    for trial in range(n_starts):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        S_init = greedy_init(order)
        # Time budget per start
        time_left = deadline - time.time()
        if time_left < 5:
            break
        ls_time = min(2.0, time_left * 0.8 / max(1, n_starts - trial))
        S_best = local_search(S_init, time_limit=ls_time)
        if len(S_best) > best_global_size:
            best_global_size = len(S_best)
            best_global = S_best

    # Strategy 3: Perturb best solution and re-search
    n_perturb = 200
    for trial in range(n_perturb):
        if time.time() > deadline:
            break
        S = set(best_global)
        if len(S) < 2:
            break
        # Remove 1-3 random elements
        n_remove = min(len(S), rng.randint(1, 3))
        to_remove = rng.sample(list(S), n_remove)
        S = S - set(to_remove)
        # Re-greedy fill
        for cand in range(m):
            if cand not in S and count_conflicts_in_set(S, cand) == 0:
                S.add(cand)
        time_left = deadline - time.time()
        if time_left < 5:
            break
        ls_time = min(1.5, time_left * 0.8 / max(1, n_perturb - trial))
        S_best = local_search(S, time_limit=ls_time)
        if len(S_best) > best_global_size:
            best_global_size = len(S_best)
            best_global = S_best

    # Strategy 4: Random restarts with small seeds
    n_seed = 100
    for trial in range(n_seed):
        if time.time() > deadline:
            break
        # Start with a small random valid seed
        seed_size = rng.randint(1, 4)
        seed = set()
        candidates = list(range(m))
        rng.shuffle(candidates)
        for c in candidates:
            if len(seed) >= seed_size:
                break
            if count_conflicts_in_set(seed, c) == 0:
                seed.add(c)
        # Greedy fill from seed
        S = set(seed)
        for cand in range(m):
            if cand not in S and count_conflicts_in_set(S, cand) == 0:
                S.add(cand)
        time_left = deadline - time.time()
        if time_left < 5:
            break
        ls_time = min(1.0, time_left * 0.8 / max(1, n_seed - trial))
        S_best = local_search(S, time_limit=ls_time)
        if len(S_best) > best_global_size:
            best_global_size = len(S_best)
            best_global = S_best

    # Final polish: run local search on best solution with more time
    time_left = deadline - time.time()
    if time_left > 10:
        S_final = local_search(best_global, time_limit=time_left - 5)
        if len(S_final) > best_global_size:
            best_global = S_final
            best_global_size = len(S_final)

    result = np.array(sorted(best_global), dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
