import time
from itertools import combinations

N = 5040
TARGET = 4979

# Precompute all divisors of 5040
_divisors = []
for d in range(1, N + 1):
    if N % d == 0:
        _divisors.append(d)

# Sort divisors descending for greedy
_divisors_desc = sorted(_divisors, reverse=True)

# Greedy baseline
def _greedy():
    rem = TARGET
    chosen = []
    for d in _divisors_desc:
        if d <= rem:
            chosen.append(d)
            rem -= d
            if rem == 0:
                break
    if rem != 0:
        return None
    return chosen

_greedy_result = _greedy()
_greedy_len = len(_greedy_result) if _greedy_result else float('inf')


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s kill

    # Greedy gives a valid upper bound on count
    if _greedy_result is not None:
        best = _greedy_result[:]
        best_len = len(best)
    else:
        best = None
        best_len = 257  # impossible initially

    # Try to find solutions with fewer divisors
    # For k divisors, check if any k-subset sums to TARGET
    # Start from k=1 upward, stop when k >= best_len

    # For small k (1,2,3,4,5), brute-force combinations
    # For larger k, use meet-in-the-middle or other heuristics

    # Precompute set of divisors for fast lookup
    div_set = set(_divisors)
    div_list = _divisors  # ascending

    # Try k = 1, 2, 3, ... up to best_len - 1
    for k in range(1, best_len):
        if time.time() > deadline - 2.0:
            break

        if k == 1:
            # Single divisor equals target?
            if TARGET in div_set:
                best = [TARGET]
                best_len = 1
                break
            continue

        if k == 2:
            # Two distinct divisors sum to target
            found = False
            for d in div_list:
                comp = TARGET - d
                if comp in div_set and comp != d and comp > 0:
                    best = [d, comp]
                    best_len = 2
                    found = True
                    break
            if found:
                break
            continue

        if k == 3:
            # Three distinct divisors sum to target
            found = False
            n_divs = len(div_list)
            # Use two-pointer style with set lookup for third
            for i in range(n_divs):
                if time.time() > deadline - 2.0:
                    break
                d1 = div_list[i]
                if d1 >= TARGET:
                    break
                rem2 = TARGET - d1
                for j in range(i + 1, n_divs):
                    d2 = div_list[j]
                    if d2 >= rem2:
                        break
                    d3 = rem2 - d2
                    if d3 in div_set and d3 > d2:
                        best = [d1, d2, d3]
                        best_len = 3
                        found = True
                        break
                if found:
                    break
            if found:
                break
            continue

        if k == 4:
            # Four distinct divisors sum to target
            # Meet in the middle: split into two pairs
            found = False
            n_divs = len(div_list)
            # Build map: sum of two distinct divisors -> (d1, d2)
            pair_sum_map = {}
            for i in range(n_divs):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 1, n_divs):
                    s = div_list[i] + div_list[j]
                    if s < TARGET:
                        pair_sum_map.setdefault(s, (div_list[i], div_list[j]))
            # Now look for two disjoint pairs whose sums add to TARGET
            for s1, (d1, d2) in pair_sum_map.items():
                if time.time() > deadline - 2.0:
                    break
                s2 = TARGET - s1
                if s2 in pair_sum_map:
                    d3, d4 = pair_sum_map[s2]
                    # Check all four are distinct
                    if len({d1, d2, d3, d4}) == 4:
                        best = [d1, d2, d3, d4]
                        best_len = 4
                        found = True
                        break
            if found:
                break
            continue

        if k >= 5:
            # For k >= 5, try a smarter approach
            # Use the fact that large divisors are efficient
            # Try: pick k-2 largest divisors, then solve for remaining 2
            # Or use iterative deepening with pruning

            # Strategy: for each subset of (k-2) divisors from a limited set,
            # check if remaining sum can be made with 2 divisors
            # This is still expensive, so use heuristics

            # Better: use the greedy result and try to improve
            # Try replacing some divisors in greedy with different ones
            if best is None:
                break

            # Try local search: for each position in best, try swapping
            # with another divisor to maintain sum but reduce count isn't possible
            # directly, so try to find a k-combination via random sampling
            import random
            random.seed(42)

            attempts = 0
            found = False
            while attempts < 200000:
                if time.time() > deadline - 2.0:
                    break
                # Random sample of k divisors
                sample = random.sample(div_list, k)
                if sum(sample) == TARGET:
                    best = sample
                    best_len = k
                    found = True
                    break
                attempts += 1

            if found:
                break

            # Also try: take k-1 largest divisors, see if last one is a divisor
            top_k = _divisors_desc[:k]
            sum_top_k_1 = sum(top_k[:k-1])
            last = TARGET - sum_top_k_1
            if last in div_set and last not in set(top_k[:k-1]):
                best = top_k[:k-1] + [last]
                best_len = k
                break

    return best


# Verify greedy works
assert _greedy_result is not None, "Greedy failed to find a solution"
assert len(_greedy_result) <= 256
assert all(N % d == 0 for d in _greedy_result)
assert len(set(_greedy_result)) == len(_greedy_result)
assert sum(_greedy_result) == TARGET
