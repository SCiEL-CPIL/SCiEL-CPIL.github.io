import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # safety margin before 90s kill
    
    # K_{8,8} complete bipartite graph on 16 vertices
    # Part A: vertices 0-7, Part B: vertices 8-15
    # Every edge connects A to B; no common neighbors for adjacent vertices
    # Diameter is exactly 2; every edge is essential (removal disconnects endpoints)
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
    
    # Verify types are correct
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
