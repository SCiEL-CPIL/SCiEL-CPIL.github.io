import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set, S_list):
        """Check if x can be added to S without violating sum-free property."""
        for a in S_list:
            if x + a in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def get_conflicts_vectorized(S_list):
        """Count conflicts using numpy vectorization for speed."""
        n = len(S_list)
        if n == 0:
            return {}
        arr = np.array(S_list, dtype=np.int64)
        # Compute all pair sums (upper triangle including diagonal)
        i_idx, j_idx = np.triu_indices(n, k=0)
        sums = arr[i_idx] + arr[j_idx]
        # Check which sums are in S
        S_set = set(S_list)
        conflict_mask = np.array([s in S_set for s in sums])
        if not conflict_mask.any():
            return {}
        ci = i_idx[conflict_mask]
        cj = j_idx[conflict_mask]
        cs = sums[conflict_mask]
        # Count conflicts per element
        conflicts = {}
        for idx in range(len(ci)):
            a_idx = ci[idx]
            b_idx = cj[idx]
            s_val = cs[idx]
            conflicts[S_list[a_idx]] = conflicts.get(S_list[a_idx], 0) + 1
            if b_idx != a_idx:
                conflicts[S_list[b_idx]] = conflicts.get(S_list[b_idx], 0) + 1
            conflicts[s_val] = conflicts.get(s_val, 0) + 1
        return conflicts

    def hill_climb_add_back(S_init, V, time_limit):
        """Hill climbing: repeatedly try to add back removed elements."""
        S = set(S_init)
        remaining = [x for x in V if x not in S]
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_remaining = []
            random.shuffle(remaining)
            S_list = list(S)
            S_set = S

            for x in remaining:
                if time.time() > start_time + time_limit:
                    new_remaining.append(x)
                    continue
                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_list.append(x)
                    improved = True
                else:
                    new_remaining.append(x)

            if improved:
                remaining = new_remaining

        return S

    def diverse_init_local_search(V, time_limit, num_starts=6):
        """
        Try multiple diverse initializations and local search from each.
        Primary strategy: start from odds (guaranteed sum-free), then add back.
        """
        best_S = []
        best_size = 0
        n = len(V)

        # Define diverse initializations
        inits = []

        # 1. Odds only (guaranteed sum-free - primary strategy)
        odds = [x for x in V if x % 2 != 0]
        if odds:
            inits.append(odds)

        # 2. Positives only
        pos = [x for x in V if x > 0]
        if pos:
            inits.append(pos)

        # 3. Negatives only
        neg = [x for x in V if x < 0]
        if neg:
            inits.append(neg)

        # 4. High absolute value elements
        sorted_by_abs = sorted(V, key=lambda x: abs(x), reverse=True)
        inits.append(sorted_by_abs[:n//2])

        # 5. Random sample
        rng = random.Random()
        inits.append(rng.sample(V, n//2))

        # 6. Another random sample
        inits.append(rng.sample(V, n//3))

        # Time per start (more time for the parity-based starts which are most promising)
        # Allocate more time to odds-based init since it's guaranteed sum-free
        total_time = time_limit
        # Give 40% to odds, 15% to each of the other 5
        time_weights = [0.40, 0.15, 0.15, 0.10, 0.10, 0.10]
        if len(inits) < 6:
            # Adjust weights
            time_weights = [w / sum(time_weights[:len(inits)]) for w in time_weights[:len(inits)]]

        for idx, init in enumerate(inits):
            if time.time() > start_time + time_limit - 1:
                break

            t_alloc = total_time * time_weights[idx] if idx < len(time_weights) else total_time / len(inits)

            # First make the init sum-free via greedy removal (only needed for non-odds inits)
            S_init = set(init)
            if not is_sum_free(S_init):
                while not is_sum_free(S_init):
                    if time.time() > start_time + time_limit - 1:
                        break
                    conflicts = get_conflicts_vectorized(list(S_init))
                    if not conflicts:
                        break
                    max_deg = max(conflicts.values())
                    candidates = [x for x, c in conflicts.items() if c == max_deg]
                    x_to_remove = random.choice(candidates)
                    S_init.discard(x_to_remove)

            # Now do local search add-back
            S_result = hill_climb_add_back(S_init, V, start_time + time_limit - 1)

            if len(S_result) > best_size:
                best_size = len(S_result)
                best_S = list(S_result)

        if not best_S:
            # Fallback: odds
            best_S = [x for x in V if x % 2 != 0]

        return best_S

    results = []

    # Time allocation: give more time to larger instances
    # Instance 1 (60): fast
    # Instance 2 (120): medium
    # Instance 3 (240): slow
    time_alloc = {60: 10.0, 120: 25.0, 240: 70.0}

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)

        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        if remaining_time < 3:
            # Emergency: just return odds
            odds = [x for x in V if x % 2 != 0]
            results.append(odds)
            continue

        t_limit = min(time_alloc.get(size, 15.0), remaining_time - 3)

        # Run diverse init local search
        best_S = diverse_init_local_search(V, t_limit, num_starts=6)

        # Final verification
        if not is_sum_free(best_S):
            # Fallback to odds
            best_S = [x for x in V if x % 2 != 0]

        results.append(sorted(best_S))

    return results
