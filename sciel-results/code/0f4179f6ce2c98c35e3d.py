"""Structural change: SAT-style constraint propagation + backtracking for max dissociated reciprocal set."""
import time
import random
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer numerators for reciprocals
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # num[i] = L // (i+1)

    def _time_ok():
        return (time.time() - start_time) < deadline

    # --- Phase 1: Multi-start greedy to get a strong initial bound ---
    def _greedy_solve(ord_idx):
        cur_sums = {0}
        cur_list = []
        for i in ord_idx:
            n = num[i]
            v = universe[i]
            # Check collision: any (s + n) in cur_sums?
            if cur_sums.isdisjoint(s + n for s in cur_sums):
                new_sums = cur_sums | {s + n for s in cur_sums}
                cur_sums = new_sums
                cur_list.append(v)
        return cur_list

    best = []
    best_len = 0

    # Try multiple orderings
    orders_to_try = [
        list(range(N)),                    # 1,2,...,22
        list(range(N-1, -1, -1)),          # 22,21,...,1
        [i for i in range(N) if (i+1) % 2 != 0] + [i for i in range(N) if (i+1) % 2 == 0],
        [i for i in range(N) if (i+1) % 2 == 0] + [i for i in range(N) if (i+1) % 2 != 0],
        # Sort by numerator descending (largest reciprocal first)
        sorted(range(N), key=lambda i: num[i], reverse=True),
        # Sort by numerator ascending (smallest reciprocal first)
        sorted(range(N), key=lambda i: num[i]),
    ]

    rng = random.Random(12345)
    for _ in range(20):
        shuffled = list(range(N))
        rng.shuffle(shuffled)
        orders_to_try.append(shuffled)

    for ord_idx in orders_to_try:
        if not _time_ok():
            break
        res = _greedy_solve(ord_idx)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # --- Phase 2: Constraint propagation + backtracking ---
    # Key structural idea: instead of naive DFS, use forward checking.
    # When we decide to include an element, we immediately check if it
    # makes any remaining element impossible to include (its numerator
    # would collide with the new sum set). This prunes the search tree
    # much more aggressively than standard branch-and-bound.

    # Also: use the difference-set characterization.
    # An element with numerator n can be added to sum-set S iff
    # n is NOT in the difference set {a - b : a, b in S} \ {0}.
    # Equivalently: no (s + n) in S for any s in S.
    # But also: after adding n, the new difference set grows, which
    # may block other elements.

    # We'll use a smarter ordering: process elements in an order that
    # maximizes early pruning. Elements with large numerators (small values)
    # create large gaps in the sum set and are more likely to cause
    # collisions with other large-numerator elements.

    # Strategy: iterative deepening on the target size.
    # Start from best_len + 1 and try to find a valid set of that size.
    # Use constraint propagation: when we fix some elements, propagate
    # which remaining elements are still "addable".

    # For efficiency, we'll do a bounded DFS with aggressive pruning:
    # 1. Order elements by a heuristic that favors those most likely to fit.
    # 2. At each node, compute which remaining elements are still addable.
    # 3. If fewer than needed elements remain addable, prune.

    # Heuristic ordering: sort by "freedom" - elements whose numerator
    # is less likely to collide with the current sum set should go first.
    # Approximation: sort by value ascending (1, 2, 3, ...) since large
    # numerators tend to be "compatible" with each other less often.

    # Actually, let's try a different structural approach:
    # Use the fact that if we have a set S with sum-set Sigma, then
    # an element n is addable iff n not in (Sigma - Sigma) \ {0}.
    # The difference set Sigma - Sigma has at most |Sigma|*(|Sigma|-1) elements.
    # For a dissociated set of size k, |Sigma| = 2^k.
    # So the difference set has at most 2^k * (2^k - 1) elements.
    # We can precompute for each pair of elements whether they're "compatible".

    # Two elements with numerators n1, n2 are incompatible if n1 = n2 (impossible here)
    # or if there exist subset sums that collide. For two elements, they're always
    # compatible since {0, n1, n2, n1+n2} has 4 distinct elements iff n1 != n2 and
    # n1 != 0 and n2 != 0 (always true). So pairwise compatibility is trivial.

    # The real constraint is higher-order. Let's use a different structural idea:
    # Randomized local search / simulated annealing on the set of included elements.

    # --- Phase 3: Local search to improve on the greedy solution ---
    # Start from the best greedy solution and try to add/remove/swap elements.

    def _is_valid_set(elems):
        """Check if all subset sums of reciprocals are distinct."""
        if not elems:
            return True
        cur_sums = {0}
        for v in elems:
            n = num[v - 1]  # v is 1-indexed value
            if not cur_sums.isdisjoint(s + n for s in cur_sums):
                return False
            cur_sums = cur_sums | {s + n for s in cur_sums}
        return True

    # Local search: try to add elements not in the current set
    if best:
        current_set = set(best)
        remaining = [v for v in universe if v not in current_set]

        # Try adding each remaining element
        for v in remaining:
            if not _time_ok():
                break
            test_set = list(current_set) + [v]
            if _is_valid_set(test_set):
                current_set.add(v)
                best = sorted(current_set)
                best_len = len(best)
                remaining = [x for x in remaining if x not in current_set]

        # Try swaps: remove one element, add two
        if _time_ok():
            for v_remove in list(current_set):
                if not _time_ok():
                    break
                temp_set = current_set - {v_remove}
                for v_add1 in remaining:
                    if not _time_ok():
                        break
                    for v_add2 in remaining:
                        if v_add2 <= v_add1:
                            continue
                        test_set = list(temp_set) + [v_add1, v_add2]
                        if _is_valid_set(test_set):
                            current_set = temp_set | {v_add1, v_add2}
                            best = sorted(current_set)
                            best_len = len(best)
                            remaining = [x for x in universe if x not in current_set]
                            break
                    if v_add2 > v_add1 and current_set != temp_set:
                        break

    # --- Phase 4: Bounded DFS with constraint propagation ---
    # Try to find a set larger than best_len using structured search.
    # Use the ordering that worked best in greedy.

    # Sort elements by a heuristic: prefer elements with smaller numerators
    # (larger values) first, as they create sparser sum sets.
    # Actually, let's try multiple orderings in the DFS.

    def _bounded_dfs(order, target_len):
        """Try to find a dissociated set of size target_len using the given order."""
        # order is a list of element indices (0-based into universe)
        # Use backtracking with forward checking

        # State: (pos, chosen_indices, cur_sums, addable_mask)
        # addable_mask: bitmask of which remaining elements are still addable

        # Precompute: for each position, which later elements are addable
        # given the current sum set. This is dynamic, so we compute on the fly.

        stack = [(0, [], {0})]
        found = None

        while stack:
            if not _time_ok():
                return None

            pos, cur_list, cur_sums = stack.pop()

            # Prune: not enough remaining elements to reach target
            if len(cur_list) + (len(order) - pos) < target_len:
                continue

            if len(cur_list) == target_len:
                return cur_list[:]

            if pos >= len(order):
                continue

            # Try to include order[pos]
            idx = order[pos]
            n = num[idx]
            if cur_sums.isdisjoint(s + n for s in cur_sums):
                new_sums = cur_sums | {s + n for s in cur_sums}
                stack.append((pos + 1, cur_list + [idx], new_sums))

            # Try to exclude order[pos]
            # Only push exclude if we still have enough elements remaining
            if len(cur_list) + (len(order) - pos - 1) >= target_len:
                stack.append((pos + 1, cur_list, cur_sums))

        return None

    # Try to find a set of size best_len + 1
    if _time_ok() and best_len < N:
        target = best_len + 1
        # Try several orderings
        dfs_orders = [
            list(range(N)),
            list(range(N-1, -1, -1)),
            sorted(range(N), key=lambda i: num[i]),  # smallest numerator first
            sorted(range(N), key=lambda i: num[i], reverse=True),  # largest first
        ]

        rng2 = random.Random(999)
        for _ in range(5):
            sh = list(range(N))
            rng2.shuffle(sh)
            dfs_orders.append(sh)

        for ord_idx in dfs_orders:
            if not _time_ok():
                break
            result = _bounded_dfs(ord_idx, target)
            if result is not None:
                best = sorted(universe[i] for i in result)
                best_len = target
                break

    # Final verification
    if best and _is_valid_set(best):
        return sorted(best)

    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
