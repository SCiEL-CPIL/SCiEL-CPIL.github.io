def construct():
    import time
    start = time.time()
    deadline = start + 80.0

    n = 16
    half = n // 2

    # K_{8,8} is the unique (up to isomorphism) extremal triangle-free graph
    # on 16 vertices by Mantel's theorem, giving 64 edges.
    # Every edge in K_{8,8} connects vertices with no common neighbor,
    # so every edge is critical. Diameter is exactly 2.
    edge_list = [[u, v] for u in range(half) for v in range(half, n)]
    
    # Ensure types are correct: Python int, list of lists
    result = [[int(u), int(v)] for u, v in edge_list]
    
    if time.time() > deadline:
        return result
    
    return result
