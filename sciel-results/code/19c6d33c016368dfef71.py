    target = np.zeros(1 << N, dtype=np.uint8)
    # target[x] = 1 if popcount(x) > 0
    # We can compute popcount for all x
    # Or simply: x > 0
    target[1:] = 1
    