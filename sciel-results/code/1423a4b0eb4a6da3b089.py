import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # 8 disjoint K4s on vertices 0..31
    # K4_b = {4*b, 4*b+1, 4*b+2, 4*b+3} for b in 0..7
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                edges.append([4*b + i, 4*b + j])
    
    # Vertex 32: connect to all 4 vertices of K4_0, and to one vertex in each K4_1..K4_7
    # K4_0 vertices: 0,1,2,3
    for v in range(4):
        edges.append([32, v])
    # K4_b for b=1..7: connect to vertex 4*b
    for b in range(1, 8):
        edges.append([32, 4*b])
    
    n = 33
    
    return {'n': n, 'edges': edges}
