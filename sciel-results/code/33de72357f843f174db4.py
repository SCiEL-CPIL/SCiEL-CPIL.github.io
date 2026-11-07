"""
Optimal: 3 edge-disjoint Hamiltonian cycles in the 6-regular 21-vertex circulant graph.
The graph has 21 vertices and is 6-regular, so it has (21 * 6) / 2 = 63 edges.
A Hamiltonian cycle on 21 vertices uses 21 edges.
Maximum number of edge-disjoint Hamiltonian cycles is floor(63 / 21) = 3.
This program constructs these 3 cycles deterministically using the circulant structure.
"""

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    
    # The graph is a circulant graph C_21(1, 2, 5).
    # Steps 1, 2, 5 generate 3 edge-disjoint Hamiltonian cycles.
    steps = [1, 2, 5]
    cycles = []
    
    for k in steps:
        cyc = [0]
        cur = 0
        for _ in range(n - 1):
            cur = (cur + k) % n
            cyc.append(cur)
            
        # Ensure strict Python int types
        cyc = [int(v) for v in cyc]
        cycles.append(cyc)
        
    # Minimal verification to ensure validity
    if len(cycles) != 3:
        return []
        
    target_vertices = set(range(n))
    used_edges = set()
    
    for cyc in cycles:
        if set(cyc) != target_vertices:
            return []
            
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            edge = (min(u, v), max(u, v))
            if edge in used_edges:
                return []
            used_edges.add(edge)
            
    return cycles
# EVOLVE-BLOCK-END
