import time

def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3,
    triangle-free, minimizing distinct simple cycle lengths.
    
    Two disjoint K_{3,3} components give cycle lengths {4, 6}, raw value = 2.
    This is optimal given the constraints.
    """
    start_time = time.time()
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.append([u, v])
    
    # Ensure all elements are Python int
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
