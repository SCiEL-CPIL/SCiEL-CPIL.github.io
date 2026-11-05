    # Initial Heuristic to set a good lower bound
    def heuristic_greedy():
        s = set()
        # Order by degree in conflict graph (ascending)
        # Conflict degree: number of u in valid_vertices such that v+u is square
        degs = []
        for v in valid_vertices:
            d = 0
            for u in valid_vertices:
                if u != v and v + u in squares:
                    d += 1
            degs.append((d, v))
        degs.sort()
        
        for d, v in degs:
            if all(v + u not in squares for u in s):
                s.add(v)
        return s

    init_set = heuristic_greedy()
    if len(init_set) > 0:
        best_size = len(init_set)
        best_clique = [valid_vertices.index(v) for v in init_set]
