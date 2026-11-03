def compute_gain(x, B, uncovered, t):
    if len(B) < t - 1:
        return 0
    # B is a list of ints
    # We need to count s in uncovered such that x in s and s\{x} subset of B
    # s\{x} is a (t-1)-subset of B
    count = 0
    # Iterate over (t-1)-subsets of B
    # To avoid sorting B every time, keep B sorted?
    # B is built incrementally. We can keep it sorted.
    # But we pop from heap, order is not sorted.
    # Let's just sort B in compute_gain? Or keep a sorted list of B?
    # Since k is small (7), sorting is cheap.
    B_sorted = tuple(sorted(B))
    for rest in itertools.combinations(B_sorted, t - 1):
        s = tuple(sorted(rest + (x,)))
        if s in uncovered:
            count += 1
    return count
