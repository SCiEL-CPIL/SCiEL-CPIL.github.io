# Precompute all 4-APs for reverse greedy
all_4aps = []
for a in range(1, N - 2):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e <= N:
            all_4aps.append((a, b, c, e))
