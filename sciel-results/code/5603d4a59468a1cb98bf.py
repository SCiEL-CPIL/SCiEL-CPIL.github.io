    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = 0
            for d in combo:
                s += inverses[d]
            s %= MOD
            subsets_by_len[length].append((combo, s))
