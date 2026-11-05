def construct():
    import time
    start_time = time.time()
    
    edges = []
    # K_{3,13}
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    # Cycle on vertices 3..15
    for i in range(3, 16):
        edges.append([i, (i % 13) + 3])
    # Note: (15 % 13) + 3 = 2 + 3 = 5? No, 15 % 13 = 2, so 2+3=5. That's wrong.
    # Let me fix: vertices are 3,4,...,15. Cycle: 3-4-5-...-15-3
    edges = []
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    # Cycle on 3..15
    for i in range(3, 15):
        edges.append([i, i+1])
    edges.append([15, 3])
    
    return edges
