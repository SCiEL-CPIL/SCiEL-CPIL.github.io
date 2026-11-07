import time

def construct():
    n = 15
    # Strategy: Construct a complete bipartite graph K_7,8.
    # Vertices 0..6 form set A (size 7)
    # Vertices 7..14 form set B (size 8)
    # Degrees in A are 8, Degrees in B are 7.
    # In a bipartite graph, any path of odd length (like 3) connects vertices from different partitions.
    # s in A -> t in B (or vice versa).
    # deg(s) in {8}, deg(t) in {7}.
    # 8 != 7, so the condition d(s) != d(t) is always satisfied.
    # Number of edges = 7 * 8 = 56.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    edges = []
    for u in range(7):
        if time.time() > deadline:
            break
        for v in range(7, 15):
            edges.append([u, v])
            if time.time() > deadline:
                break
    
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in edges]
    
    return final_edges
