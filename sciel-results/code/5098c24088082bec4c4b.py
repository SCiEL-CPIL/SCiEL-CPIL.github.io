import time
from fractions import Fraction

def construct():
    n = 256

    # Exact weights
    weights = [Fraction(0)] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = Fraction(1, i)

    # Suffix sums of exact weights for tight upper bound
    suffix = [Fraction(0)] * (n + 2)
    for i in range(n, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]

    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s kill

    best_set = []
    best_sum = Fraction(0)

    # Greedy seed: add in increasing order, checking 3-AP via set
    S = set()
    for x in range(1, n + 1):
        ok = True
        for b in S:
            a = 2 * b - x
            if a in S:
                ok = False
                break
        if ok:
            S.add(x)
    best_set = list(S)
    best_sum = sum((weights[x] for x in S), Fraction(0))

    # Precompute, for each x, the list of b in [1, x) such that a=2b-x is in [1,x)
    # and a < b. Then a 3-AP with x as the largest term exists iff some b in this list
    # has both a and b in S.
    # Since x is the largest, the only APs involving x are (a,b,x) with 2b=a+x.
    # For a given x, b ranges over values where a=2b-x is in [1,x) and a<b.
    # a < b  <=>  2b-x < b  <=>  b < x.  Always true for b<x.
    # a >= 1  <=>  2b-x >= 1  <=>  b >= (x+1)/2.
    # b < x  and  b in [1, x-1].
    # Also need b in S and a in S.
    # We'll just iterate b from ceil((x+1)/2) to x-1.
    # That's at most x/2 iterations, but we can precompute the (a,b) pairs per x.

    # Actually, to make the check fast, precompute for each x the list of (a,b) pairs.
    # But the number of pairs per x is ~x/2, and total is O(n^2) ~ 32768, fine.
    conflict_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        # b from ceil((x+1)/2) to x-1
        b_min = (x + 1 + 1) // 2  # ceil((x+1)/2)
        # Actually ceil((x+1)/2): for x=1, (2)/2=1 -> b from 1 to 0, none.
        # Let's just compute: a = 2b - x >= 1 => b >= (x+1)/2.
        # b integer, so b >= ceil((x+1)/2).
        # b < x.
        for b in range(b_min, x):
            a = 2 * b - x
            if a >= 1 and a < b:
                conflict_pairs[x].append((a, b))

    # Now search with exact upper bound.
    # State: S (set), current_sum (Fraction), next_idx (int)
    # We process indices in increasing order.

    # To speed up time checks, check every so often.
    check_interval = 200
    node_count = 0

    def search(S, current_sum, next_idx):
        nonlocal best_set, best_sum, node_count, deadline, start_time

        node_count += 1
        if node_count % check_interval == 0:
            if time.time() > deadline:
                return

        # Update best
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(S)

        if next_idx > n:
            return

        # Upper bound: current_sum + suffix[next_idx]
        if current_sum + suffix[next_idx] <= best_sum:
            return

        x = next_idx

        # Try including x
        # Check conflicts: for each (a,b) in conflict_pairs[x], if a in S and b in S -> conflict
        can_add = True
        for a, b in conflict_pairs[x]:
            if a in S and b in S:
                can_add = False
                break

        if can_add:
            S.add(x)
            search(S, current_sum + weights[x], next_idx + 1)
            S.remove(x)

        # Try excluding x
        search(S, current_sum, next_idx + 1)

    # Start search from empty set with the greedy lower bound
    search(set(), Fraction(0), 1)

    return best_set
