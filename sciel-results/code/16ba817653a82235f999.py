"""
Constructs a K4-free graph on 18 vertices that minimizes the size of the largest 
triangle-free induced subgraph.

The strategy uses the complete 3-partite Turán graph T_3(18). 
In this graph, the vertices are partitioned into 3 sets of size 6. 
Edges exist between all pairs of vertices in different sets, and no edges 
exist within a set. 

This graph is K4-free because any K4 would require vertices from 4 distinct 
parts (or 3 parts with specific internal structure which is absent), but 
more simply, a K4 in a complete 3-partite graph is impossible as it would 
require 4 vertices no two of which are in the same part, but there are only 
3 parts. (Actually, a K4 in T_3(n) is impossible because any 4 vertices must 
have at least two in the same part, and there is no edge between them).

The largest triangle-free induced subgraph in T_3(18) consists of the union 
of any two parts, as the subgraph induced by two parts is a complete bipartite 
graph, which is triangle-free. The size is 6 + 6 = 12. Any subset including 
vertices from all three parts will contain a triangle (one from each part).
"""

def construct():
    n = 18
    # Partition vertices into 3 parts of equal size
    part_size = n // 3
    parts = [
        list(range(0, part_size)),
        list(range(part_size, 2 * part_size)),
        list(range(2 * part_size, 3 * part_size))
    ]
    
    edges = []
    # Add edges between all vertices in different parts
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
                    
    return edges
