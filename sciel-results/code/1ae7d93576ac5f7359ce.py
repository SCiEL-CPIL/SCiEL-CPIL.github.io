import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 12
    
    # Known optimal: two disjoint K_{3,3}
    # Each K_{3,3} has cycles of length 4 and 6 only
    # Total distinct cycle lengths = {4, 6} = 2
    
    edges = []
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    for i in range(3):
        for j in range(3):
            edges.append([i, j + 3])
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    for i in range(3):
        for j in range(3):
            edges.append([i + 6, j + 9])
    
    return edges
