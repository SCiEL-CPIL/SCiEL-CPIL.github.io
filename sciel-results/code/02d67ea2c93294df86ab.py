import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = L * i // __import__('math').gcd(L, i)

    # V[i] = L // i
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    def _subset_sum_possible(vals, target):
        """True iff some subset of vals (integers) sums to target. Meet-in-the-middle."""
        k = len(vals)
        if k == 0:
            return target == 0
        if k <= 1:
            return any(v == target for v in vals)
        if sum(vals) < target:
            return False
        if target < 0:
            return False

        half = k // 2
        left = vals[:half]
        right = vals[half:]

        # Build left subset sums
        left_sums = {0}
        for v in left:
            left_sums = left_sums | {s + v for s in left_sums}

        # Build right subset sums
        right_sums = {0}
        for v in right:
            right_sums = right_sums | {s + v for s in right_sums}

        # Check if any combination works
        if len(left_sums) <= len(right_sums):
            for ls in left_sums:
                if (target - ls) in right_sums:
                    return True
        else:
            for rs in right_sums:
                if (target - rs) in left_sums:
                    return True
        return False

    def _check_validity_incremental(current_set, new_elem):
        """Check if adding new_elem to current_set preserves validity."""
        new_recip = V[new_elem]
        current_recips = [V[v] for v in current_set]

        # Check 1: new element's reciprocal vs. current set
        if _subset_sum_possible(current_recips, new_recip):
            return False

        # Check 2: each existing element's reciprocal vs. (others + new)
        for a in current_set:
            a_recip = V[a]
            others = [V[v] for v in current_set if v != a]
            others.append(new_recip)
            if _subset_sum_possible(others, a_recip):
                return False

        return True

    def is_valid(S):
        """Exact reciprocal-subsum-free check for a nonempty set S of ints."""
        if not S:
            return False
        for a in S:
            others = [V[v] for v in S if v != a]
            if _subset_sum_possible(others, V[a]):
                return False
        return True

    def _greedy_add(best, all_elems):
        """Try to add elements greedily to best."""
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for v in all_elems:
                if time.time() > deadline:
                    break
                if v in best:
                    continue
                if _check_validity_incremental(best, v):
                    best = best + [v]
                    improved = True
                    break
        return best

    def _local_search(best):
        """Local search: try remove-1-add-1 and remove-1-add-2 swaps."""
        all_elems = set(range(1, 25))
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            best_set = set(best)

            # Try remove-1-add-1
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j in range(1, 25):
                    if time.time() > deadline:
                        break
                    if j in best_set:
                        continue
                    candidate = [x for x in best if x != best[i]] + [j]
                    if len(candidate) > len(best) and is_valid(candidate):
                        best = candidate
                        best_set = set(best)
                        improved = True
                        break
                if improved:
                    break

            if improved or time.time() > deadline:
                continue

            # Try remove-1-add-2
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j, k in combinations(range(1, 25), 2):
                    if time.time() > deadline:
                        break
                    if j in best_set or k in best_set:
                        continue
                    candidate = [x for x in best if x != best[i]] + [j, k]
                    if len(candidate) > len(best) and is_valid(candidate):
                        best = candidate
                        best_set = set(best)
                        improved = True
                        break
                if improved:
                    break

        return best

    def _branch_and_bound(best, all_elems):
        """Branch-and-bound DFS to find exact maximum within time budget."""
        best = list(best)
        best_size = len(best)
        best_set = set(best)

        # Sort elements: try larger elements first (smaller reciprocals, less likely to conflict)
        order = sorted(all_elems, key=lambda x: -x)

        def dfs(current_set, idx, current_size):
            nonlocal best, best_size, best_set

            if time.time() > deadline:
                return

            # Prune: if we can't beat best, stop
            if current_size + (len(order) - idx) <= best_size:
                return

            if idx >= len(order):
                return

            # Try including order[idx]
            elem = order[idx]
            if elem not in best_set:
                if _check_validity_incremental(current_set, elem):
                    new_set = current_set + [elem]
                    dfs(new_set, idx + 1, current_size + 1)

            # Try excluding order[idx]
            dfs(current_set, idx + 1, current_size)

        dfs(best, 0, len(best))
        return best

    # Start with the known good seed: top half {13,...,24}
    best = list(range(13, 25))
    best_size = len(best)
    all_elems = list(range(1, 25))

    # Phase 1: Greedy improvement
    best = _greedy_add(best, all_elems)
    best_size = len(best)

    # Phase 2: Local search
    if time.time() < deadline:
        best = _local_search(best)
        best_size = len(best)

    # Phase 3: Branch-and-bound if time remains
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 5:  # Only do B&B if we have enough time
            best = _branch_and_bound(best, all_elems)
            best_size = len(best)

    # Final validation
    if not is_valid(best):
        # Fallback to known good
        best = list(range(13, 25))

    return list(best)
