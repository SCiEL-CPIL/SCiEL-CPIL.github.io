def construct():
    # Use a random-ish structured set that maximizes additive collisions.
    # The goal is to minimize the largest Sidon subset (including diagonals, i.e., 2x).
    # We try a few candidate sets that are known to have high additive energy
    # or dense pair-sum collisions, and pick the one with the smallest max Sidon size.
    #
    # Strategy:
    # 1. Try the arithmetic progression {0,1,...,11}
    # 2. Try sets concentrated in small intervals with overlaps
    # 3. Try sets with many repeated differences
    #
    # For a set S, the largest Sidon subset (with 2x included) means:
    # for all unordered pairs (a,b) with a<=b, a+b must be unique.
    # This includes a=a (so 2a must also be unique among all pair sums).
    #
    # We'll do a quick search over several candidate constructions.

    import random
    from itertools import combinations

    def max_sidon_size(S):
        """Compute the largest subset of S with all unordered pair sums (including diagonals) unique."""
        S = sorted(S)
        n = len(S)
        if n <= 1:
            return n
        # Try subsets in decreasing size order
        for size in range(n, 0, -1):
            for subset in combinations(range(n), size):
                indices = list(subset)
                # Check all unordered pairs (including self-pairs)
                sums = set()
                valid = True
                for i in range(len(indices)):
                    for j in range(i, len(indices)):
                        s = S[indices[i]] + S[indices[j]]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    return size
        return 0

    # Candidate 1: Arithmetic progression
    c1 = list(range(12))

    # Candidate 2: Two overlapping intervals
    c2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # same as c1, try different

    # Candidate 3: Clustered set - many elements close together
    c3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Let's think differently. The AP {0,...,11} has the property that
    # pair sums range from 0 to 22, and there are 12*13/2 = 78 pair sums
    # (including diagonals) but only 23 possible sums (0 to 22).
    # So there are massive collisions.

    # What if we try a set where elements are more spread out but still have
    # many collisions? For example, a set with many pairs summing to the same value.

    # Candidate: symmetric around a center, like {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    # Actually, let's try: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} vs
    # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12} - replacing 11 with 12

    # Let me try a different approach: use a set with high additive energy.
    # A set with many solutions to a+b=c+d.

    # Try: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} - this is the AP
    # Try: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15}
    # Try: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15}

    # Actually, the key insight: to minimize the largest Sidon subset,
    # we want every large subset to have collisions. The AP likely achieves
    # the minimum possible, which is 4.

    # Let me verify: in {0,...,11}, can we find a Sidon subset of size 5?
    # A Sidon set of size 5 in [0,11] with all pair sums (including 2x) unique.
    # The smallest possible Sidon set of size 5 with these constraints...
    # Actually, for a set of size k, we need k(k+1)/2 distinct pair sums.
    # For k=5, that's 15 distinct sums, ranging from 0 to 22, so 23 possible values.
    # It's possible in principle, but the question is whether it exists within {0,...,11}.

    # Let's just compute it directly.
    candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14],
    ]

    best_set = None
    best_size = 999

    for c in candidates:
        if len(c) != 12 or len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = c

    if best_set is None:
        return list(range(12))
    return best_set
