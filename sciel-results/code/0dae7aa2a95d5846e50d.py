"""
Optimal: 3 edge-disjoint Hamiltonian cycles in the 6-regular 21-vertex circulant graph C_21(1,2,5).
The graph has 21 vertices and is 6-regular, so it has (21 * 6) / 2 = 63 edges.
A Hamiltonian cycle on 21 vertices uses 21 edges.
Maximum number of edge-disjoint Hamiltonian cycles is floor(63 / 21) = 3.
Since 63 = 3 * 21 exactly, 3 cycles form a Hamiltonian decomposition (use all edges).
This is the theoretical maximum; no solution can exceed 3 cycles.
"""

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    
    # The graph is the circulant C_21(1, 2, 5): each vertex i connects to
    # (i±1) mod 21, (i±2) mod 21, (i±5) mod 21.
    # Three steps {1, 2, 5} generate three edge-disjoint Hamiltonian cycles
    # that together partition all 63 edges (a Hamiltonian decomposition).
    steps = [1, 2, 5]
    cycles = []
    
    for k in steps:
        cyc = []
        cur = 0
        for _ in range(n):
            cyc.append(cur)
            cur = (cur + k) % n
        # Ensure strict Python int types
        cyc = [int(v) for v in cyc]
        cycles.append(cyc)
        
    # Verification: check all constraints
    if len(cycles) != 3:
        return []
        
    target_vertices = set(range(n))
    used_edges = set()
    
    for cyc in cycles:
        # Check: all vertices distinct and cover full vertex set
        if len(cyc) != n:
            return []
        if set(cyc) != target_vertices:
            return []
        
        # Check: consecutive pairs (including wrap) are host edges and not repeated
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            edge = (min(u, v), max(u, v))
            if edge in used_edges:
                return []
            used_edges.add(edge)
            
    # Verify all edges are host edges
    host_edge_set = set()
    for e in P['edges']:
        host_edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    for edge in used_edges:
        if edge not in host_edge_set:
            return []
            
    return cycles
# EVOLVE-BLOCK-END
