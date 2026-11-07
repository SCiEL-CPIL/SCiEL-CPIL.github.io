import time

def construct():
    n = 16
    # Optimal solution is K_8,8 with 64 edges.
    # It is the maximum edge count for a triangle-free diameter-2 graph.
    # Any added edge creates a triangle, making some edges non-critical 
    # (distance to 2 without increasing global diameter > 2).
    
    # Part A and Part B
    part_a = set(range(0, 8))
    part_b = set(range(8, 16))
    
    edges = []
    # Generate edges between the two partitions
    for u in part_a:
        for v in part_b:
            edges.append([int(u), int(v)])
            
    return edges
