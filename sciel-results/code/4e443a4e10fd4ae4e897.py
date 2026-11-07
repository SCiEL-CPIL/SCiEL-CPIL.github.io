def construct():
    # K_{8,8} complete bipartite graph on 16 vertices
    # Part A: vertices 0-7, Part B: vertices 8-15
    # Triangle-free (Mantel's theorem: max 64 edges for n=16)
    # Diameter exactly 2; every edge essential (no common neighbors for adjacent pairs)
    # This is the theoretical maximum: graph must be triangle-free, and Mantel bounds edges at 64
    return [[int(u), int(v)] for u in range(8) for v in range(8, 16)]
