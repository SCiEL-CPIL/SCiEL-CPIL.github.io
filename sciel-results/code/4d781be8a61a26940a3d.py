import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    edges = []
    
    # Part 0: 0-5
    # Part 1: 6-11
    # Part 2: 12-17
    
    # Add edges between parts
    for i in range(n):
        for j in range(i + 1, n):
            part_i = i // 6
            part_j = j // 6
            if part_i != part_j:
                edges.append([i, j])
                
    # Add matchings inside parts to reduce independence number
    # Part 0
    edges.append([0, 1])
    edges.append([2, 3])
    edges.append([4, 5])
    
    # Part 1
    edges.append([6, 7])
    edges.append([8, 9])
    edges.append([10, 11])
    
    # Part 2
    edges.append([12, 13])
    edges.append([14, 15])
    edges.append([16, 17])
    
    return edges
