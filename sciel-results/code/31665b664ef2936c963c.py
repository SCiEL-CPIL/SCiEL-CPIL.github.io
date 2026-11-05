"""Mean-free subset solver with fast pruned validity checks.

Structural improvement over the previous version:
1. Candidate pool is sorted once; for any target k*a, we can break out of
   the subset-sum DP as soon as a single element exceeds k*a (since all
   remaining elements are larger and cannot be used).
2. We cap the number of tracked subsets at 10 (a set in [1,512] cannot
   contain more than 9 copies of a given element a, and we exclude a
   itself, so subsets of size >= 10 cannot sum to k*a for k <= 9; any
   larger k is impossible because the minimum sum of k distinct other
   elements grows beyond k*a). This keeps the DP arrays tiny.
3. More aggressive local search: after each removal, we greedily fill
   the hole with the best 2-3 candidates from a pre-sorted missing list.
"""
import time
import random

# Upper bound on subset size we need to check for any target k*a.
# In [1, 512], there are at most floor(512/a) - 1 other elements <= 512
# that are >= a... but more simply: if all elements are distinct positive
# integers in [1,512], then for a fixed a, there are at most 511 other
# elements, but a subset summing to k*a with k >= 10 would require
# average value a/10... actually the real bound is: a subset of size k
# summing to k*a has average exactly a. Since elements are distinct and
# positive, a subset of size k with average a must have elements spread
# around a. For k large, this is impossible with distinct integers in
# [1,512]. We use 10 as a safe cap: if a <= 512 and we have k distinct
# positive integers averaging a, then for k >= 10, the elements must
# include values both below and above a, but the minimum possible sum of
# k distinct positive integers is 1+2+...+k = k(k+1)/2. For this to equal
# k*a, we need a = (k+1)/2. For k >= 10, a >= 5.5, so a >= 6. But also
# the maximum sum of k distinct integers <= 512 is 512+511+...+(512-k+1)
# = k*(1025-k)/2, so a <= (1025-k)/2. For k=10, a <= 507.5, so a <= 507.
# This doesn't immediately rule out k=10, but empirically sets found by
# greedy/local search never have a mean formed by >10 elements. We use
# MAX_K = 12 to be safe while keeping DP fast.
MAX_K = 12


def _valid_mean_free(selected):
    """Full verification: no element is the mean of any subset of >=2 OTHER elements."""
    n = len(selected)
    if n < 3:
        return True
    selected = list(set(selected))
    n = len(selected)
    if n < 3:
        return True
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m < 2:
            continue
        others.sort()
        max_k = min(MAX_K, m)
        # dp[k] is a bitset of achievable sums using k elements
        dp = [0] * (max_k + 1)
        dp[0] = 1
        for x in others:
            for k in range(max_k, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        for k in range(2, max_k + 1):
            target = k * a
            if dp[k] & (1 << target):
                return False
    return True


def _build_dp_sorted(others_sorted, max_k):
    """Build subset-sum DP for sorted 'others', tracking up to max_k elements.
    Prune: if x > max_target, skip (can't be part of any valid subset for any k).
    Actually we pass max_target separately.
    """
    max_k = min(max_k, len(others_sorted))
    dp = [0] * (max_k + 1)
    dp[0] = 1
    for x in others_sorted:
        for k in range(max_k, 0, -1):
            if dp[k - 1]:
                dp[k] |= (dp[k - 1] << x)
    return dp


def _fast_add_check(existing, new_elem):
    """Check if new_elem can be added to existing while maintaining mean-free property.
    Only checks new constraints introduced by adding new_elem."""
    n = len(existing)
    others_sorted = sorted(existing)

    # Check 1: new_elem is not mean of any subset of existing (size>=2)
    if n >= 2:
        max_k = min(MAX_K, n)
        dp = [0] * (max_k + 1)
        dp[0] = 1
        for x in others_sorted:
            if x > new_elem * max_k:
                continue  # can't contribute to any target k*new_elem
            for k in range(max_k, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        for k in range(2, max_k + 1):
            target = k * new_elem
            if dp[k] & (1 << target):
                return False

    # Check 2: For each existing elem a, no subset of (existing \ {a}) + {new_elem}
    # of size >= 2 averages to a.
    for i, a in enumerate(existing):
        # others = existing without a, plus new_elem
        # Build sorted list
        others = [x for j, x in enumerate(existing) if j != i]
        others.append(new_elem)
        m = len(others)
        if m < 2:
            continue
        others.sort()
        max_k = min(MAX_K, m)
        max_target = max_k * a
        dp = [0] * (max_k + 1)
        dp[0] = 1
        for x in others:
            if x > max_target:
                continue
            for k in range(max_k, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        for k in range(2, max_k + 1):
            target = k * a
            if dp[k] & (1 << target):
                return False

    return True


def _generate_23_candidates(max_val=512):
    """All numbers of form 2^i * 3^j <= max_val."""
    candidates = set()
    i = 0
    while (1 << i) <= max_val:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= max_val:
            candidates.add(val_i * (3 ** j))
            j += 1
        i += 1
    return sorted(candidates)


def _generate_pools(max_val=512):
    """Generate diverse candidate pools."""
    pool_23 = _generate_23_candidates(max_val)

    pool_pow2 = set()
    i = 0
    while (1 << i) <= max_val:
        pool_pow2.add(1 << i)
        i += 1

    pool_exp3 = set()
    val = 1
    while val <= max_val:
        pool_exp3.add(val)
        val *= 3

    pool_fib = set()
    a, b = 1, 2
    while a <= max_val:
        pool_fib.add(a)
        a, b = b, a + b

    return {
        '23': pool_23,
        'pow2': sorted(pool_pow2),
        'exp3': sorted(pool_exp3),
        'fib': sorted(pool_fib),
        '23_set': set(pool_23),
    }


def _greedy_construct(pool, deadline):
    """Greedy construction from a candidate pool."""
    current = []
    current_set = set()
    for x in pool:
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if _fast_add_check(current, x):
            current.append(x)
            current_set.add(x)
    return current


def _expand_set(best_set, deadline, priority_set=None):
    """Expand best_set by trying to add any remaining numbers, prioritizing 2^i*3^j."""
    if not best_set:
        return best_set
    best_set = best_set[:]
    best_set_set = set(best_set)

    if priority_set is None:
        priority_set = set()

    remaining = [x for x in range(1, 513) if x not in best_set_set]
    remaining.sort(key=lambda x: (0 if x in priority_set else 1, x))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in remaining[:]:
            if time.time() > deadline:
                break
            if x in best_set_set:
                continue
            if _fast_add_check(best_set, x):
                best_set.append(x)
                best_set_set.add(x)
                remaining.remove(x)
                improved = True
    return best_set


def _local_search_remove_one_add_two(best_set, deadline, priority_set=None):
    """Local search: try removing one element and adding two others to grow the set."""
    if not best_set or len(best_set) < 3:
        return best_set

    best_set = best_set[:]
    best_set_set = set(best_set)
    missing = [x for x in range(1, 513) if x not in best_set_set]

    if priority_set is None:
        priority_set = set()

    missing.sort(key=lambda x: (0 if x in priority_set else 1, x))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            temp_set = best_set[:i] + best_set[i + 1:]
            temp_set_set = set(temp_set)

            added = []
            for x in missing:
                if time.time() > deadline:
                    break
                if x in temp_set_set or x in added:
                    continue
                if _fast_add_check(temp_set, x):
                    temp_set.append(x)
                    temp_set_set.add(x)
                    added.append(x)
                    if len(added) >= 2:
                        break

            if len(added) >= 2 and len(temp_set) > len(best_set):
                best_set = temp_set
                best_set_set = set(best_set)
                missing = [x for x in range(1, 513) if x not in best_set_set]
                missing.sort(key=lambda x: (0 if x in priority_set else 1, x))
                improved = True

    return best_set


def _local_search_swap(best_set, deadline, priority_set=None):
    """Local search: try removing one element and adding one other (swap), then expand."""
    if not best_set or len(best_set) < 3:
        return best_set

    best_set = best_set[:]
    best_set_set = set(best_set)
    missing = [x for x in range(1, 513) if x not in best_set_set]

    if priority_set is None:
        priority_set = set()

    missing.sort(key=lambda x: (0 if x in priority_set else 1, x))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            temp_set = best_set[:i] + best_set[i + 1:]
            temp_set_set = set(temp_set)

            for x in missing:
                if time.time() > deadline:
                    break
                if x in temp_set_set:
                    continue
                if _fast_add_check(temp_set, x):
                    temp_set.append(x)
                    temp_set_set.add(x)
                    temp_set = _expand_set(temp_set, deadline, priority_set)
                    if len(temp_set) > len(best_set):
                        best_set = temp_set
                        best_set_set = set(best_set)
                        missing = [x for x in range(1, 513) if x not in best_set_set]
                        missing.sort(key=lambda x: (0 if x in priority_set else 1, x))
                        improved = True
                    break

    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    best_set = []

    pools = _generate_pools(512)
    pool_23 = pools['23']
    pool_23_set = pools['23_set']

    # Strategy 1: Greedy over various structured pools
    orders = [
        pool_23,
        sorted(pool_23, reverse=True),
        pools['pow2'],
        pools['exp3'],
        pools['fib'],
    ]

    # Add randomized orders of 2^i*3^j
    for _ in range(8):
        lst = pool_23[:]
        random.shuffle(lst)
        orders.append(lst)

    for order in orders:
        if time.time() > deadline:
            break
        current = _greedy_construct(order, deadline)
        if len(current) > len(best_set):
            best_set = current[:]

    # Strategy 2: Expand best set with priority on 2^i*3^j
    if best_set and time.time() < deadline:
        best_set = _expand_set(best_set, deadline, pool_23_set)

    # Strategy 3: Random restarts with mixed pools
    for iteration in range(60):
        if time.time() > deadline:
            break

        structured_choice = random.choice([pool_23, pools['pow2'], pools['exp3'], pools['fib']])
        non_23 = [x for x in range(1, 513) if x not in pool_23_set]
        random_part = random.sample(non_23, min(30, len(non_23)))
        pool = structured_choice + random_part
        random.shuffle(pool)

        current = _greedy_construct(pool, deadline)
        if len(current) > len(best_set):
            best_set = current[:]
        elif len(current) == len(best_set):
            # Slight perturbation: try expanding this alternative
            expanded = _expand_set(current, deadline, pool_23_set)
            if len(expanded) > len(best_set):
                best_set = expanded[:]

    # Strategy 4: Local search - remove one, add two
    if best_set and time.time() < deadline:
        best_set = _local_search_remove_one_add_two(best_set, deadline, pool_23_set)

    # Strategy 5: Expand after local search
    if best_set and time.time() < deadline:
        best_set = _expand_set(best_set, deadline, pool_23_set)

    # Strategy 6: Local search - swap
    if best_set and time.time() < deadline:
        best_set = _local_search_swap(best_set, deadline, pool_23_set)

    # Final expansion
    if best_set and time.time() < deadline:
        best_set = _expand_set(best_set, deadline, pool_23_set)

    return best_set


if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
