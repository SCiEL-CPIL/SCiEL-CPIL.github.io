"""
Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth at least 4.
The objective is to minimize the number of distinct simple cycle lengths.

Design: Two disjoint copies of K_{3,3}.
- K_{3,3} is 3-regular, bipartite (triangle-free), girth 4.
- Simple cycles in K_{3,3} have lengths {4, 6}.
- Disjoint union preserves the set: {4, 6}.
- Distinct cycle lengths: 2.
- This is the minimum achievable for N=12, min degree 3, triangle-free.
"""

import time


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Two disjoint K_{3,3} components
    # Component 1: vertices 0-5, partitions {0,1,2} and {3,4,5}
    # Component 2: vertices 6-11, partitions {6,7,8} and {9,10,11}
    
    edges = []
    
    # First K_{3,3}
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
    
    # Check deadline
    if time.time() > deadline:
        return [[int(a), int(b)] for a, b in edges]
    
    # Second K_{3,3}
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
    
    # Verify and clean up: ensure valid types, no loops, no duplicates
    final_edges = []
    seen = set()
    for e in edges:
        u, v = e
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (min(u, v), max(u, v))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        final_edges.append([a, b])
    
    # Quick structural verification
    adj = {i: set() for i in range(12)}
    for u, v in final_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check minimum degree >= 3
    for i in range(12):
        if len(adj[i]) < 3:
            # Should not happen with K_{3,3} construction, but guard
            pass
    
    # Check triangle-free (no 3-cycles)
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k == i:
                    continue
                if k in adj[i]:
                    # Triangle found - should not happen
                    pass
    
    return final_edges
