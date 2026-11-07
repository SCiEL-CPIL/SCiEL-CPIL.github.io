import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Hubs: 11 (Odd), 12 (Even), 13 (Odd)
    # 12 connects to all Odds: 1, 3, ..., 23
    # 11 connects to Evens 0, 2, ..., 10
    # 13 connects to Evens 14, 16, ..., 22
    
    edges_to_add = []
    
    # 12 to Odds
    for v in range(1, 24, 2):
        edges_to_add.append([12, v])
        
    # 11 to Evens 0..10
    for v in range(0, 11, 2):
        edges_to_add.append([11, v])
        
    # 13 to Evens 14..22
    for v in range(14, 23, 2):
        edges_to_add.append([13, v])
        
    # Convert to int lists
    result = [[int(u), int(v)] for u, v in edges_to_add]
    
    return result
