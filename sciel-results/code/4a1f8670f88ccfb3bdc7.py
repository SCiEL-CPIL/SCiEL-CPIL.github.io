    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        max_a = int(math.isqrt(LIMIT // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= LIMIT:
                powerful.add(val)
    