    # Greedy initialization
    def greedy_init():
        diffs = frozenset()
        sel = []
        for x in range(1, 23):
            val = D_LCM // x
            if val not in diffs:
                # Check if adding is valid
                # Actually, if val not in diffs, it's valid.
                sel.append(x)
                new_diffs = set(diffs)
                new_diffs.add(val)
                for d in diffs:
                    d1 = d + val
                    d2 = abs(d - val)
                    if d1 > 0: new_diffs.add(d1)
                    if d2 > 0: new_diffs.add(d2)
                diffs = frozenset(new_diffs)
        return sel

    best_set = greedy_init()
