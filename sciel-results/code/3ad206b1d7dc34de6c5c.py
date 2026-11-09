import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    h = n // 2
    edges = [[u, v] for u in range(h) for v in range(h, n)]
    
    if time.time() > deadline:
        return edges
        
    return edges
