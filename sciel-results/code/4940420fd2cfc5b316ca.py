"""Construct a graph on 12 vertices with min degree 3 and minimum distinct cycle lengths.
Optimal value is 2 (cycle lengths 4 and 6)."""

def construct():
    # The graph consists of two disjoint copies of K_{3,3}.
    # K_{3,3} is 3-regular and bipartite, so it is triangle-free.
    # Simple cycles in K_{3,3} have even lengths.
    # The maximum length of a simple cycle in K_{3,3} is 2 * min(|A|, |B|) = 6.
    # The possible lengths are 4 and 6.
    # Thus, the set of distinct cycle lengths for one K_{3,3} is {4, 6}.
    # Two disjoint copies preserve this set: {4, 6}.
    # Count of distinct lengths = 2.
    # This is likely the minimum possible for N=12 with min degree 3.
    
    edges = []
    
    # First K_{3,3} on vertices 0-5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6-11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    # Ensure strict Python int types and list format
    out = []
    seen = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (min(u, v), max(u, v))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append([int(a), int(b)])
        
    return out
