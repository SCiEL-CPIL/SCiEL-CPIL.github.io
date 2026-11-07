import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth >= 4.
    The graph consists of two disjoint copies of K_3,3.
    
    K_3,3 properties:
    - Bipartite (triangle-free).
    - 3-regular (min degree 3).
    - Simple cycles have lengths 4 and 6 only.
    
    Distinct cycle lengths: {4, 6}. Count = 2.
    This is the theoretical minimum for this problem class.
    """
    # Deadline guard is trivially satisfied as the operation is O(1)
    # start_time = time.time()
    # deadline = start_time + 59.0
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 (Left) and 3, 4, 5 (Right)
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([int(u), int(v)])
            
    # Component 2: K_{3,3} on vertices 6, 7, 8 (Left) and 9, 10, 11 (Right)
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([int(u), int(v)])
            
    # Verification (Optional, kept minimal for speed)
    # The logic is static and verified. 
    # Any dynamic search for a better solution (value 1) is not pursued 
    # as value 2 is the established lower bound for n=12, min_deg=3, girth>=4.
    
    return edges
