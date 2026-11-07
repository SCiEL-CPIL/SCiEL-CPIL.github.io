import time

def construct():
    deadline = time.time() + 85.0  # 85s safety margin within 90s limit
    n = 15
    # Partition vertices into two sets A and B
    # A = {0, ..., 6} (size 7)
    # B = {7, ..., 14} (size 8)
    # Construct complete bipartite graph K_{7,8}
    # Degrees in A are all 8.
    # Degrees in B are all 7.
    # Since 8 != 7, any path of length 3 (which goes A-B-A-B or B-A-B-A)
    # will have endpoints in different partitions with degrees 8 and 7, which are distinct.
    
    edges = []
    for i in range(7):
        for j in range(7, 15):
            if time.time() > deadline:
                break
            edges.append([int(i), int(j)])
        if time.time() > deadline:
            break
    
    # Ensure all edges are proper Python lists of Python ints
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
