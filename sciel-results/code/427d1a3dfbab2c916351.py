M_expected = np.zeros((n * m, m * p))
for b in range(m):
    idx_a = i * m + b
    idx_b = b * p + k
    M_expected[idx_a, idx_b] = 1.0
