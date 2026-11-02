def _build_edges(n, offsets, k):
    edges = []
    for i in range(n):
        for off in offsets:
            j = (i + off) % n
            if i < j:
                edges.append((i, j))
    return edges

def _find_offsets_exhaustive(n, k, g):
    half_k = k // 2
    if half_k != 2:
        return None
    for a in range(1, n // 2):
        for b in range(a + 1, n // 2):
            offsets = [a, b]
            edges = _build_edges(n, offsets, k)
            if _girth_from_0_is_at_least_g(n, edges, g):
                return offsets
    return None

def _find_offsets_random(n, k, g, max_attempts=100):
    half_k = k // 2
    valid_offsets = list(range(1, n // 2))
    if len(valid_offsets) < half_k:
        return None
    for _ in range(max_attempts):
        offsets = random.sample(valid_offsets, half_k)
        edges = _build_edges(n, offsets, k)
        if _girth_from_0_is_at_least_g(n, edges, g):
            return offsets
    return None
