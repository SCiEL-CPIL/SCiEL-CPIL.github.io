def construct():
    from itertools import product, combinations
    
    vertices = list(range(9))
    
    # Known optimal: Fano plane on 7 vertices (subset of 0..8)
    # We construct it and verify it's 3-chromatic
    edges = [
        [0, 1, 3],
        [0, 2, 4],
        [0, 5, 6],
        [1, 2, 5],
        [1, 4, 6],
        [2, 3, 6],
        [3, 4, 5]
    ]
    
    # Provide a valid 3-coloring
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify: every edge is non-monochromatic
    for e in edges:
        c = set(colors[v] for v in e)
        if len(c) == 1:
            raise ValueError(f"Edge {e} is monochromatic")
    
    # Verify: no 2-coloring exists (exhaustive check)
    # Only vertices 0-6 are used in edges; 7,8 are isolated
    used_vertices = set()
    for e in edges:
        used_vertices.update(e)
    
    found_2coloring = False
    for assignment in product([0, 1], repeat=len(used_vertices)):
        color_map = dict(zip(sorted(used_vertices), assignment))
        valid = True
        for e in edges:
            c = set(color_map[v] for v in e)
            if len(c) == 1:
                valid = False
                break
        if valid:
            found_2coloring = True
            break
    
    if found_2coloring:
        raise ValueError("Hypergraph admits a 2-coloring, not 3-chromatic")
    
    return {'edges': edges, 'colors': colors}
