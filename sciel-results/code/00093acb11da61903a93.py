def construct():
    import time
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s limit
    
    n = 16
    half = n // 2
    
    # K_{8,8} is the unique (up to isomorphism) extremal triangle-free graph
    # on 16 vertices by Mantel's theorem, giving 64 edges.
    # Every edge in K_{8,8} connects vertices with no common neighbor,
    # so every edge is critical. Diameter is exactly 2.
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Ensure types are correct: Python int, list of lists
    result = [[int(u), int(v)] for u, v in edges]
    
    if time.time() > deadline:
        # Should never happen for this O(1) construction
        return result
    
    return result
