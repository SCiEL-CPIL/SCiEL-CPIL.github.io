"""Rewritten program for the suite task: two disjoint K_{3,3} copies."""

def construct():
    """
    Construct an undirected simple graph on vertices 0..11.
    Two disjoint copies of K_{3,3}:
      - Copy 1: bipartition {0,1,2} and {3,4,5}
      - Copy 2: bipartition {6,7,8} and {9,10,11}
    
    Properties:
      - Every vertex has degree exactly 3 (satisfies minimum degree >= 3).
      - The graph is bipartite, hence triangle-free (girth >= 4).
      - In K_{3,3}, the only simple cycle lengths are 4 and 6.
      - Therefore, the distinct cycle lengths are {4, 6}, giving raw value 2.
    """
    edges = []
    
    # First K_{3,3}: left {0,1,2}, right {3,4,5}
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
    
    # Second K_{3,3}: left {6,7,8}, right {9,10,11}
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
    
    # Ensure edges are sorted for consistency, with u < v
    edges = [sorted(e) for e in edges]
    edges.sort()
    
    return edges

# EVOLVE-BLOCK-START
# (The construct function above is the complete solution.)
# EVOLVE-BLOCK-END
