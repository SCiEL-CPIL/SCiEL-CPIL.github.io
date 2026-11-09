<<<<<<< SEARCH
def _random_start(density):
    adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < density:
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    return adj
=======
def _random_start(density):
    adj = [0] * N
    # Randomize edge order to avoid bias
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(pairs)
    for u, v in pairs:
        if random.random() < density:
            if _can_add_edge_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj
>>>>>>> REPLACE

