import time
import math
from math import gcd

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 140.0
_N = 512
_LIMIT = 511


def _score(lst):
    return sum(1.0 / (_N - a) for a in lst)


def _precompute_coprime_masks():
    """For each a in 1..511, precompute a bitmask of all b in 1..511 with gcd(a,b)==1."""
    # We'll use a list of Python ints as bitmasks (511 bits each)
    masks = [0] * (_LIMIT + 1)
    for a in range(1, _LIMIT + 1):
        m = 0
        for b in range(1, _LIMIT + 1):
            if gcd(a, b) == 1:
                m |= (1 << b)
        masks[a] = m
    return masks


def _branch_and_bound(masks, deadline):
    """
    Exact branch-and-bound: try all candidates in descending value order.
    At each node, decide include or exclude.
    Upper bound = current_score + sum of remaining values (relaxation ignoring coprimality).
    """
    # Candidates sorted by descending value 1/(512-a), i.e., descending a
    candidates = list(range(_LIMIT, 0, -1))
    n = len(candidates)
    # Precompute suffix sums of values for upper bound
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + 1.0 / (_N - candidates[i])

    best_score = 0.0
    best_set = []
    # Track which indices are chosen (for reconstruction)
    chosen = []

    # Use a recursive DFS with pruning
    # To avoid deep recursion issues, we use iterative approach with explicit stack
    # But 511 depth is fine for Python recursion if we set limit
    import sys
    sys.setrecursionlimit(2000)

    def dfs(idx, cur_score, chosen_mask, chosen_list):
        nonlocal best_score, best_set

        if time.time() > deadline:
            return

        if idx == n:
            if cur_score > best_score:
                best_score = cur_score
                best_set = list(chosen_list)
            return

        # Upper bound: even if we take all remaining, what's the max?
        ub = cur_score + suffix[idx]
        if ub <= best_score + 1e-15:
            return

        a = candidates[idx]
        bit = 1 << a

        # Option 1: Include a if coprime to all chosen
        if (chosen_mask & ~masks[a]) == 0:
            # a is coprime to all in chosen_mask
            chosen_mask |= bit
            chosen_list.append(a)
            dfs(idx + 1, cur_score + 1.0 / (_N - a), chosen_mask, chosen_list)
            chosen_list.pop()
            chosen_mask ^= bit

        # Option 2: Exclude a
        dfs(idx + 1, cur_score, chosen_mask, chosen_list)

    dfs(0, 0.0, 0, [])
    return best_set, best_score


def _branch_and_bound_with_ordering(masks, deadline):
    """
    Improved B&B: use a smarter ordering and tighter bounds.
    Instead of pure descending a, we can try a few orderings.
    Also, we can use a greedy solution as initial lower bound.
    """
    # First get a good initial solution via greedy
    greedy_set = []
    greedy_mask = 0
    for a in range(_LIMIT, 0, -1):
        if (greedy_mask & ~masks[a]) == 0:
            greedy_set.append(a)
            greedy_mask |= (1 << a)
    greedy_score = _score(greedy_set)

    # Now do B&B with this as initial best
    candidates = list(range(_LIMIT, 0, -1))
    n = len(candidates)
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + 1.0 / (_N - candidates[i])

    best_score = greedy_score
    best_set = list(greedy_set)

    import sys
    sys.setrecursionlimit(2000)

    def dfs(idx, cur_score, chosen_mask, chosen_list):
        nonlocal best_score, best_set

        if time.time() > deadline:
            return

        if idx == n:
            if cur_score > best_score:
                best_score = cur_score
                best_set = list(chosen_list)
            return

        ub = cur_score + suffix[idx]
        if ub <= best_score + 1e-15:
            return

        a = candidates[idx]
        bit = 1 << a

        # Try include first (since high-value items are more likely beneficial)
        if (chosen_mask & ~masks[a]) == 0:
            chosen_mask |= bit
            chosen_list.append(a)
            dfs(idx + 1, cur_score + 1.0 / (_N - a), chosen_mask, chosen_list)
            chosen_list.pop()
            chosen_mask ^= bit

        # Try exclude
        dfs(idx + 1, cur_score, chosen_mask, chosen_list)

    dfs(0, 0.0, 0, [])
    return best_set, best_score


def _local_search_improve(start, masks, deadline):
    """
    Local search: try to swap out low-value elements for high-value coprime ones.
    Use bitmask for fast checks.
    """
    cur = list(start)
    cur_score = _score(cur)
    cur_mask = 0
    for a in cur:
        cur_mask |= (1 << a)

    best_score = cur_score
    best_set = list(cur)

    while time.time() < deadline:
        improved = False
        # Try adding any coprime candidate not in set
        for a in range(_LIMIT, 0, -1):
            bit = 1 << a
            if (cur_mask & bit) != 0:
                continue
            if (cur_mask & ~masks[a]) == 0:
                cur.append(a)
                cur_mask |= bit
                cur_score += 1.0 / (_N - a)
                improved = True
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = list(cur)

        # Try 1-for-1 swaps: remove one, add one or more
        for i in range(len(cur)):
            a_out = cur[i]
            bit_out = 1 << a_out
            new_mask = cur_mask ^ bit_out
            # Try to add best available
            for a_in in range(_LIMIT, 0, -1):
                bit_in = 1 << a_in
                if (new_mask & bit_in) != 0:
                    continue
                if (new_mask & ~masks[a_in]) == 0:
                    gain = 1.0 / (_N - a_in) - 1.0 / (_N - a_out)
                    if gain > 1e-15:
                        cur[i] = a_in
                        cur_mask = new_mask | bit_in
                        cur_score += gain
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_set = list(cur)
                        break

        if not improved:
            break

    return best_set, best_score


def construct():
    deadline = _DEADLINE - 5.0  # safety margin

    # Precompute coprime masks
    masks = _precompute_coprime_masks()

    # Phase 1: Greedy initial solution
    greedy_set = []
    greedy_mask = 0
    for a in range(_LIMIT, 0, -1):
        if (greedy_mask & ~masks[a]) == 0:
            greedy_set.append(a)
            greedy_mask |= (1 << a)
    greedy_score = _score(greedy_set)

    best_set = list(greedy_set)
    best_score = greedy_score

    # Phase 2: Local search to improve
    ls_set, ls_score = _local_search_improve(best_set, masks, deadline - 30.0)
    if ls_score > best_score:
        best_set, best_score = ls_set, ls_score

    # Phase 3: Branch and bound (exact if time allows)
    if time.time() < deadline - 10.0:
        bb_set, bb_score = _branch_and_bound_with_ordering(masks, deadline)
        if bb_score > best_score:
            best_set, best_score = bb_set, bb_score

    return list(best_set)

# EVOLVE-BLOCK-END
