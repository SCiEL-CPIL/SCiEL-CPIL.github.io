# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0  # Safety margin before 90s kill
    N = 512
    MAXC = 48
    rng = random.Random(98765)

    def _deadline_exceeded():
        return time.time() >= DEADLINE

    def _is_mean_free(s):
        """
        Exact check: no element a in s equals the arithmetic mean of any subset
        of size >= 2 from s \ {a}.
        Uses integer bitset DP for subset sums of exact cardinalities.
        Returns True if valid.
        """
        n = len(s)
        if n < 3:
            return True
        for i in range(n):
            a = s[i]
            # dp[k] is a bitset: bit v set iff sum v is achievable with exactly k elements
            # from s excluding index i.
            dp = [0] * (n + 1)
            dp[0] = 1  # sum 0 with 0 elements
            for j in range(n):
                if j == i:
                    continue
                v = s[j]
                # Iterate k descending to avoid using v more than once
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k - 1] << v)
            # Check if k*a is achievable for any k in [2, n-1]
            for k in range(2, n):
                if (dp[k] >> (k * a)) & 1:
                    return False
        return True

    def _fast_validate_pair(s):
        """
        Quick necessary-condition check: no element is the mean of any 2-element subset.
        This is much faster than full validation and catches most violations early.
        """
        s_set = set(s)
        for a in s:
            for b in s:
                if b == a:
                    continue
                c = 2 * a - b
                if c in s_set and c != a and c != b:
                    return False
        return True

    def _greedy_min_outdegree(candidates, current, current_set):
        """
        Greedy: repeatedly pick the candidate with the fewest 2-conflicts
        with the current set (and among themselves), adding it if valid.
        This is a structural change: instead of random/greedy-by-value,
        we use a min-conflict heuristic.
        """
        result = list(current)
        result_set = set(current_set)
        remaining = [x for x in candidates if x not in result_set]

        while len(result) < MAXC and remaining and not _deadline_exceeded():
            best_x = None
            best_conflicts = None

            # For each remaining candidate, count how many 2-conflicts it creates
            for x in remaining:
                conflicts = 0
                for a in result:
                    # x is mean of pair (a, b): 2x = a + b => b = 2x - a
                    b = 2 * x - a
                    if b in result_set and b != a and b != x:
                        conflicts += 1
                    # a is mean of pair (b, x): 2a = b + x => b = 2a - x
                    b2 = 2 * a - x
                    if b2 in result_set and b2 != a and b2 != x:
                        conflicts += 1
                if best_conflicts is None or conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_x = x
                    if conflicts == 0:
                        break

            if best_x is None:
                break

            # Verify the addition is truly 2-valid
            x = best_x
            valid = True
            for a in result:
                b = 2 * x - a
                if b in result_set and b != a and b != x:
                    valid = False
                    break
                b2 = 2 * a - x
                if b2 in result_set and b2 != a and b2 != x:
                    valid = False
                    break

            if valid:
                result.append(x)
                result_set.add(x)
                remaining.remove(x)
            else:
                # This shouldn't happen if conflict count was 0, but be safe
                remaining.remove(x)

        return result, result_set

    def _local_search_add(s, s_set):
        """Try to add any element from 1..N that isn't in s, maintaining validity."""
        improved = True
        while improved and not _deadline_exceeded():
            improved = False
            for x in range(1, N + 1):
                if _deadline_exceeded():
                    break
                if x in s_set:
                    continue
                if len(s) >= MAXC:
                    break
                # Quick 2-conflict check
                ok = True
                for a in s:
                    b = 2 * x - a
                    if b in s_set and b != a and b != x:
                        ok = False
                        break
                    b2 = 2 * a - x
                    if b2 in s_set and b2 != a and b2 != x:
                        ok = False
                        break
                if not ok:
                    continue
                # Full validation
                test_s = s + [x]
                if _is_mean_free(test_s):
                    s = test_s
                    s_set.add(x)
                    improved = True
                    break
        return s, s_set

    def _local_search_swap(s, s_set):
        """Try to replace one element with another to maintain/increase size."""
        improved = True
        while improved and not _deadline_exceeded():
            improved = False
            for i in range(len(s)):
                if _deadline_exceeded():
                    break
                old = s[i]
                for x in range(1, N + 1):
                    if _deadline_exceeded():
                        break
                    if x in s_set:
                        continue
                    # Remove old, add x
                    new_s = [v for j, v in enumerate(s) if j != i] + [x]
                    new_s.sort()
                    if len(new_s) <= len(s):
                        continue
                    # Quick check
                    if not _fast_validate_pair(new_s):
                        continue
                    if _is_mean_free(new_s):
                        s = new_s
                        s_set = set(s)
                        improved = True
                        break
                if improved:
                    break
        return s, s_set

    # Phase 1: Start with a good base using Cantor-like numbers (powers of 2)
    # Powers of 2 are known to be mean-free for small sizes.
    base = []
    val = 1
    while val <= N and len(base) < MAXC:
        base.append(val)
        val *= 2
    # base = [1, 2, 4, 8, 16, 32, 64, 128, 256] -> 9 elements

    # Phase 2: Greedy min-outdegree expansion
    best, best_set = _greedy_min_outdegree(range(1, N + 1), base, set(base))
    best.sort()

    # Phase 3: Validate and repair if needed
    if not _is_mean_free(best):
        # Remove violators
        while not _is_mean_free(best) and not _deadline_exceeded():
            # Find a violator
            n = len(best)
            removed = False
            for i in range(n):
                a = best[i]
                dp = [0] * (n + 1)
                dp[0] = 1
                for j in range(n):
                    if j == i:
                        continue
                    v = best[j]
                    for k in range(n, 0, -1):
                        dp[k] |= (dp[k - 1] << v)
                for k in range(2, n):
                    if (dp[k] >> (k * a)) & 1:
                        best.pop(i)
                        best_set.discard(a)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break

    # Phase 4: Local search - add elements
    best, best_set = _local_search_add(best, best_set)

    # Phase 5: Local search - swaps
    if not _deadline_exceeded():
        best, best_set = _local_search_swap(best, best_set)

    # Final validation
    if not _is_mean_free(best):
        # Emergency repair
        while not _is_mean_free(best) and not _deadline_exceeded():
            n = len(best)
            removed = False
            for i in range(n):
                a = best[i]
                dp = [0] * (n + 1)
                dp[0] = 1
                for j in range(n):
                    if j == i:
                        continue
                    v = best[j]
                    for k in range(n, 0, -1):
                        dp[k] |= (dp[k - 1] << v)
                for k in range(2, n):
                    if (dp[k] >> (k * a)) & 1:
                        best.pop(i)
                        best_set.discard(a)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break

    return best

# EVOLVE-BLOCK-END
