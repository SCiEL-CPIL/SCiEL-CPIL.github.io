"""Improved program for the suite task."""

def construct():
    edges = []
    
    # First K_{3,3} component using vertices 0..5
    # Partition A: {0, 1, 2}, Partition B: {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} component using vertices 6..11
    # Partition A: {6, 7, 8}, Partition B: {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges
