def construct():
    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges. Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # We use the Fano plane on vertices 0..6 and leave 7,8 isolated.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    # Valid 3-coloring for Fano plane vertices, plus arbitrary for 7,8
    colors = [0, 1, 2, 1, 1, 2, 0, 0, 0]
    
    # Ensure types are Python int and lists
    # edges are already lists of ints
    # colors are already ints
    # But we enforce it to be safe against any potential type coercion issues
    edges = [list(e) for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
