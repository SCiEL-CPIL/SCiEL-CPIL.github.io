def construct():
    # Optimal 32-point configuration on triangular lattice
    # Start with hexagon of radius 2 (19 points), add 13 points from ring 3 greedily
    
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
    
    # Hexagon of radius 2: all points with hex_dist <= 2
    selected = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) <= 2:
                selected.add((a, b))
    
    # Ring 3 points
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append((a, b))
    
    # Greedily add 13 points from ring 3 with most neighbors in selected
    for _ in range(13):
        best = None
        max_neighbors = -1
        for p in ring3:
            if p in selected:
                continue
            neighbors = 0
            for da, db in dirs:
                if (p[0] + da, p[1] + db) in selected:
                    neighbors += 1
            if neighbors > max_neighbors:
                max_neighbors = neighbors
                best = p
        if best is not None:
            selected.add(best)
    
    return [[a, b] for a, b in selected]
