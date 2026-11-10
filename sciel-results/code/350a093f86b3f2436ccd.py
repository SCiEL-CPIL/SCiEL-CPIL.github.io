def construct():
    D = [0, 1, 3, 4, 6, 7]
    selected = set()
    for d2 in D:
        for d1 in D:
            for d0 in D:
                val = d2 * 64 + d1 * 8 + d0
                # val is in [0, 511]
                # Shift to [1, 512]
                selected.add(val + 1)
    return sorted(selected)
