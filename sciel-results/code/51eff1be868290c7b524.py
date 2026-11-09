import time

def construct():
    n = 15
    # Partitions A (0-6) and B (7-14)
    # Complete bipartite graph K_7,8
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([int(u), int(v)])
    
    # Add one edge within A to get 57 edges
    # Edge (0, 1)
    edges.append([int(0), int(1)])
    
    return edges
