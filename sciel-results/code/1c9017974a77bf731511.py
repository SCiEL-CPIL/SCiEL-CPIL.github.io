def construct():
    import math
    from itertools import combinations

    def score_of(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        diff_set = set()
        sum_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
                sum_set.add(a + b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0 or n_diff == 0:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1
        return 1 + math.log(ratio) / denom

    best_candidate = [0]
    best_score = 0

    # Search parameters
    # n: size of U
    # M_max: max value for elements
    search_params = [
        (3, 20),
        (4, 30),
        (5, 40),
        (6, 35),
        (7, 25),
        (8, 20),
    ]

    for n, M_max in search_params:
        # Max element must be at least n-1
        for max_u in range(n - 1, M_max + 1):
            # Generate all subsets of size n-1 from 1..max_u
            for combo in combinations(range(1, max_u + 1), n - 1):
                U = [0] + list(combo)
                sc = score_of(U)
                if sc > best_score:
                    best_score = sc
                    best_candidate = U[:]

    return best_candidate
