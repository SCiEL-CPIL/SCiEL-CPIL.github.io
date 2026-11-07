import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Connect the two components with a single edge to make the graph connected
    # This does not create any new cycles because there is no path back
    edges.append([0, 6])

    if time.time() > deadline:
         # Fallback
         return [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [8,9], [9,10], [10,11], [11,8]]

    return edges
