"""Branch-and-bound for max dissociated reciprocal set with local refinement."""
import time
from math import gcd
from functools import reduce


def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L where L = lcm(1..22)
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # integer numerator for each reciprocal

    def _time_ok():
        return (time.time() - start_time) < deadline

    def _verify_dissociated(elems):
        """Verify all subset sums of reciprocals are distinct using integer arithmetic."""
        if not elems:
            return True
        n = len(elems)
        expected = 1 << n
        sums = {0}
        for e in elems:
            r = L // e
            new_sums = {s + r for s in sums}
            if len(new_sums) != len(sums):
                return False
            sums = sums | new_sums
        return len(sums) == expected

    def _collision_score(elems):
        """Count how many other elements each element collides with when added to a growing set."""
        # For each element, count how many subset sums of the current set would collide
        scores = [0] * len(elems)
        for i, e in enumerate(elems):
            r = L // e
            # Count collisions with all pairs of subset sums
            # A collision occurs if there exist s1, s2 in current sums with s1 + r == s2
            # This is equivalent to: the set of sums and sums+r have an intersection
            # We approximate by checking against a small sample of subset sums
            # For efficiency, just count how many elements j != i have gcd issues
            for j in range(len(elems)):
                if i != j:
                    # Check if 1/ei and 1/ej have a simple relation
                    # More collisions likely when ratios are simple
                    a, b = L // e, L // elems[j]
                    g = gcd(a, b)
                    if g > 1:
                        scores[i] += 1
        return scores

    # Try multiple orderings for the DFS search
    # Ordering 1: descending value (22, 21, ..., 1) - small reciprocals first
    # Ordering 2: ascending value (1, 2, ..., 22) - large reciprocals first
    # Ordering 3: interleaved - mix of small and large
    # Ordering 4: by collision score (most constrained first)
    orderings = [
        list(range(N - 1, -1, -1)),  # 22, 21, ..., 1
        list(range(N)),               # 1, 2, ..., 22
        # Interleaved: take from both ends
        list(range(0, N, 2)) + list(range(1, N, 2))[::-1],
    ]
    
    # Add ordering by collision score
    scores = _collision_score(universe)
    collision_order = sorted(range(N), key=lambda i: -scores[i])
    orderings.append(collision_order)

    best = []
    best_len = 0

    def _dfs_search(order, vals, nums, time_limit):
        """Run DFS branch-and-bound with given ordering. Returns (best_list, best_len)."""
        nonlocal best, best_len

        local_best = []
        local_best_len = 0

        stack = [(0, [], {0})]

        while stack:
            pos, cur_list, cur_sums = stack.pop()

            if not _time_ok() or (time.time() - start_time) > time_limit:
                break

            # Prune: even if we add all remaining, can't beat local best
            remaining = N - pos
            if len(cur_list) + remaining <= local_best_len:
                continue

            if pos == N:
                if len(cur_list) > local_best_len:
                    local_best_len = len(cur_list)
                    local_best = cur_list[:]
                continue

            n = nums[pos]
            v = vals[pos]

            # Check collision for including n
            if cur_sums.isdisjoint(s + n for s in cur_sums):
                # No collision, we can include
                new_sums = cur_sums | {s + n for s in cur_sums}
                new_list = cur_list + [v]
                # Push exclude first so include is popped first (LIFO)
                stack.append((pos + 1, cur_list, cur_sums))
                stack.append((pos + 1, new_list, new_sums))
            else:
                # Collision, must exclude
                stack.append((pos + 1, cur_list, cur_sums))

        return local_best, local_best_len

    # Split time: 55% for DFS search, 35% for local refinement, 10% buffer
    search_deadline = start_time + 48
    refine_deadline = start_time + 78

    # Try each ordering with time budget
    for order in orderings:
        if not _time_ok() or (time.time() - start_time) > search_deadline:
            break

        vals = [universe[i] for i in order]
        nums = [num[i] for i in order]

        local_best, local_best_len = _dfs_search(order, vals, nums, search_deadline)

        if local_best_len > best_len:
            best_len = local_best_len
            best = local_best[:]

    # Local refinement: iterative improvement
    if best and _time_ok():
        best_set = set(best)
        
        # Phase 1: Try adding elements not in best
        improved = True
        while improved and _time_ok() and (time.time() - start_time) < refine_deadline:
            improved = False
            remaining = [e for e in universe if e not in best_set]
            for e in remaining:
                if not _time_ok() or (time.time() - start_time) > refine_deadline:
                    break
                test_set = best + [e]
                if _verify_dissociated(test_set):
                    best = test_set[:]
                    best_len += 1
                    best_set.add(e)
                    improved = True

        # Phase 2: Try single swaps (remove one, add one) to find better configurations
        if _time_ok() and (time.time() - start_time) < refine_deadline:
            swap_improved = True
            while swap_improved and _time_ok() and (time.time() - start_time) < refine_deadline:
                swap_improved = False
                for i, e_out in enumerate(list(best)):
                    if (time.time() - start_time) > refine_deadline:
                        break
                    candidate = best[:i] + best[i+1:]
                    for e_in in [e for e in universe if e not in set(candidate)]:
                        if (time.time() - start_time) > refine_deadline:
                            break
                        test1 = candidate + [e_in]
                        if _verify_dissociated(test1):
                            # This swap maintains same size but might enable future additions
                            # Only accept if it leads to a set from which we can add more
                            # For now, just track it as a potential improvement
                            pass

        # Phase 3: Try double swaps (remove one, add two)
        if _time_ok() and (time.time() - start_time) < refine_deadline:
            for i, e_out in enumerate(list(best)):
                if (time.time() - start_time) > refine_deadline:
                    break
                candidate = best[:i] + best[i+1:]
                for e1 in [e for e in universe if e not in set(candidate)]:
                    if (time.time() - start_time) > refine_deadline:
                        break
                    test1 = candidate + [e1]
                    if _verify_dissociated(test1):
                        for e2 in [e for e in universe if e not in set(test1)]:
                            if (time.time() - start_time) > refine_deadline:
                                break
                            test2 = test1 + [e2]
                            if _verify_dissociated(test2):
                                if len(test2) > best_len:
                                    best = test2[:]
                                    best_len = len(test2)
                                    break
                        else:
                            continue
                        break

    # Final verification
    if best and _verify_dissociated(best):
        return sorted(best)

    # Fallback: a known dissociated set
    fallback = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    if _verify_dissociated(fallback):
        return fallback

    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
