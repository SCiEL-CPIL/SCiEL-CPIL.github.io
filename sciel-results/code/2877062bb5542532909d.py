import time

def construct():
    """
    Returns edges for a triangle-free graph on 12 vertices with min degree >= 3
    that minimizes the number of distinct simple cycle lengths.
    
    Strategy: Use two disjoint copies of K_{3,3}.
    K_{3,3} is bipartite (triangle-free), 3-regular, and has cycle lengths {4, 6}.
    The union of two such components has distinct cycle lengths {4, 6}, raw value 2.
    
    This construction is valid, fast, and achieves the theoretical minimum 
    for this graph size and constraints.
    """
    t_start = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Component 1: K_{3,3} on vertices 0,1,2 and 3,4,5
    comp1_edges = [
        [0, 3], [0, 4], [0, 5],
        [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5]
    ]
    
    # Component 2: K_{3,3} on vertices 6,7,8 and 9,10,11
    comp2_edges = [
        [6, 9], [6, 10], [6, 11],
        [7, 9], [7, 10], [7, 11],
        [8, 9], [8, 10], [8, 11]
    ]
    
    edges = comp1_edges + comp2_edges
    
    # Ensure strict type checking: convert to Python int
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    # Safety check for deadline
    if time.time() - t_start > deadline:
        return final_edges
        
    return final_edges
