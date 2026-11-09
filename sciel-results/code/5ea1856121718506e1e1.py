edges = [(u, v) for u in range(k) for v in range(u + 1, k)]  # no random
edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
