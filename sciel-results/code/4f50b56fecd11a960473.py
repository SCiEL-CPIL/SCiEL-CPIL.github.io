def _greedy_reverse(bits_start=0):
    bits = bits_start
    for x in range(N, 0, -1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits
