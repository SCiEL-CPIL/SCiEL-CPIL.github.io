    mask = 0
    for x in range(1, 1009):
        r = (x * x) % 1009
        if r != 0:
            mask |= (1 << r)
