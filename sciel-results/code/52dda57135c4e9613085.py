import time
import random

def construct():
    start = time.time()
    DEADLINE = 140.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def can_add(x, Sset, S_list):
        """Check if x can be added to S maintaining sum-freeness."""
        if x + x in Sset:
            return False
        for a in S_list:
            if x + a in Sset:
                return False
        return True

    def repair_sum_free(S, time_limit, start_time):
        """Remove elements until S is sum-free, removing the most-conflicting element each time."""
        S = list(set(S))
        Sset = set(S)
        while not is_sum_free(S):
            if time.time() - start_time > time_limit:
                break
            n = len(S)
            if n == 0:
                break
            # Count conflicts per element using index map
            conflicts = [0] * n
            s_to_idx = {v: i for i, v in enumerate(S)}
            for i in range(n):
                a = S[i]
                for j in range(i, n):
                    b = S[j]
                    s = a + b
                    if s in s_to_idx:
                        k = s_to_idx[s]
                        conflicts[i] += 1
                        if j != i:
                            conflicts[j] += 1
                        conflicts[k] += 1
            max_conf = max(conflicts)
            if max_conf == 0:
                break
            idx = conflicts.index(max_conf)
            S.pop(idx)
            Sset = set(S)
        return S

    def greedy_add(S_base, V, time_limit, start_time, order_fn=None):
        """Greedily add elements from V \\ S_base to S_base, maintaining sum-freeness."""
        S = list(set(S_base))
        Sset = set(S)
        remaining = [x for x in V if x not in Sset]
        if order_fn:
            remaining = order_fn(remaining)
        for x in remaining:
            if time.time() - start_time > time_limit:
                break
            if can_add(x, Sset, S):
                S.append(x)
                Sset.add(x)
        return S

    def local_search(S, V, time_limit, start_time, rng):
        """Try swap operations: remove one element, greedily re-add others."""
        S = list(S)
        Sset = set(S)
        best_size = len(S)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            n = len(S)
            # Try removing each element and see if we can add more
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                S_removed = S[:i] + S[i+1:]
                S_removed_set = set(S_removed)
                # Try to greedily add from remaining
                S_new = list(S_removed)
                S_new_set = set(S_removed)
                for x in V:
                    if x in S_new_set:
                        continue
                    if can_add(x, S_new_set, S_new):
                        S_new.append(x)
                        S_new_set.add(x)
                if len(S_new) > best_size:
                    best_size = len(S_new)
                    S = S_new
                    Sset = set(S)
                    improved = True
                    break
        return S

    def direct_odds(V):
        return [x for x in V if x % 2 != 0]

    def direct_upper_magnitude(V):
        if not V:
            return []
        max_abs = max(abs(x) for x in V)
        threshold = max_abs / 2.0
        return [x for x in V if abs(x) > threshold]

    def direct_upper_third(V):
        if not V:
            return []
        max_abs = max(abs(x) for x in V)
        threshold = max_abs * 2.0 / 3.0
        return [x for x in V if abs(x) > threshold]

    def direct_modular(V, M):
        if not V or M <= 0:
            return []
        min_v = min(V)
        lower = M // 3 + 1
        upper = (2 * M) // 3
        result = []
        for x in V:
            r = (x - min_v) % M
            if lower <= r <= upper:
                result.append(x)
        return result

    def direct_positive(V):
        return [x for x in V if x > 0]

    def direct_negative(V):
        return [x for x in V if x < 0]

    def solve_instance(V, time_limit, start_time):
        """Solve one instance, return the best sum-free subset."""
        max_abs = max(abs(x) for x in V) if V else 0

        # Generate direct structural candidates
        candidates = [
            direct_odds(V),
            direct_upper_magnitude(V),
            direct_upper_third(V),
            direct_positive(V),
            direct_negative(V),
        ]
        for M in [max(3, max_abs), max(3, len(V)), max(3, len(V)) * 2, max(3, max_abs // 2)]:
            if M > 0:
                candidates.append(direct_modular(V, M))

        best_set = set()
        best_size = 0

        # Process each candidate
        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            cand_set = set(cand)
            if not is_sum_free(cand_set):
                cand_list = repair_sum_free(list(cand_set), 2.0, time.time())
                cand_set = set(cand_list)

            if len(cand_set) > best_size:
                best_size = len(cand_set)
                best_set = set(cand_set)

            # Greedy expand with different orderings
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 1.0:
                orderings = [
                    lambda lst: sorted(lst, key=lambda x: abs(x), reverse=True),
                    lambda lst: sorted(lst, key=lambda x: x),
                    lambda lst: sorted(lst, key=lambda x: -x),
                    lambda lst: sorted(lst, key=lambda x: abs(x)),
                ]
                for order_fn in orderings:
                    if time.time() - start_time > time_limit:
                        break
                    expanded = greedy_add(cand_set, V, time_limit, start_time, order_fn)
                    if is_sum_free(expanded) and len(expanded) > best_size:
                        best_size = len(expanded)
                        best_set = set(expanded)

        # Local search on the best found
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5.0 and best_size > 0:
            rng = random.Random(42)
            improved = local_search(list(best_set), V, time_limit, start_time, rng)
            if is_sum_free(improved) and len(improved) > best_size:
                best_size = len(improved)
                best_set = set(improved)

        # Final safety
        if not is_sum_free(best_set):
            best_set = set(direct_odds(V))
            if not is_sum_free(best_set):
                best_set = set()
                best_list = list(best_set)
                for x in V:
                    if can_add(x, set(best_list), best_list):
                        best_list.append(x)
                best_set = set(best_list)

        return sorted(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining = DEADLINE - elapsed
        if remaining < 10:
            # Fallback: just odds
            results.append(direct_odds(V))
            continue
        results.append(solve_instance(V, remaining - 3.0, start))

    return results
