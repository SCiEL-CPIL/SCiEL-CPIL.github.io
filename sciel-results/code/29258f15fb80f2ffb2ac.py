    indices = np.arange(2**N)
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1
