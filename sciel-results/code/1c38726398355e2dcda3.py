"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # The graph is C_21(1,2,5), a circulant graph.
    # We decompose all 63 edges into 3 edge-disjoint Hamiltonian cycles
    # using steps +1, +2, and +5 (all coprime to 21).
    
    cycles = []
    for step in [1, 2, 5]:
        cycle = []
        v = 0
        for _ in range(n):
            cycle.append(int(v))
            v = (v + step) % n
        cycles.append(cycle)
    
    # Validate: all cycles use same vertex set, edges are valid and disjoint
    vertex_sets = [set(c) for c in cycles]
    assert all(vs == set(range(n)) for vs in vertex_sets), "Vertex sets differ"
    
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    used_edges = set()
    for cycle in cycles:
        for i in range(n):
            a = cycle[i]
            b = cycle[(i + 1) % n]
            e = (min(a, b), max(a, b))
            assert e in edge_set, f"Edge {e} not in graph"
            assert e not in used_edges, f"Edge {e} used twice"
            used_edges.add(e)
    
    return cycles
# EVOLVE-BLOCK-END
