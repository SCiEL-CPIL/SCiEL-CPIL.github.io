import time
from itertools import product

def construct():
    # Define Grötzsch graph structure (Mycielski of C5)
    # Vertices:
    # 0-4: Cycle v0..v4
    # 5-9: u0..u4 (u_i adjacent to v_i and v_{i-1 mod 5})
    # 10: w (adjacent to all u_i)
    
    base_adj = {
        0: [1, 4, 6, 9], 
        1: [0, 2, 5, 7], 
        2: [1, 3, 6, 8], 
        3: [2, 4, 7, 9], 
        4: [3, 0, 5, 8], 
        5: [0, 1, 10],   
        6: [0, 2, 10],   
        7: [1, 3, 10],   
        8: [2, 4, 10],   
        9: [3, 4, 10],   
        10: [5, 6, 7, 8, 9]
    }
    
    # Pre-calculate edge list for base graph
    base_edges = []
    for i in range(11):
        for j in base_adj[i]:
            if i < j:
                base_edges.append((i, j))
                
    def calc_edges(k):
        total = 0
        for i, j in base_edges:
            total += k[i] * k[j]
        return total

    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = 0
    best_k = None
    
    # Generate all compositions of 5 into 11 parts
    # sum(x) = 5, x_i >= 0
    # k_i = x_i + 1
    
    def generate_compositions(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(n + 1):
                for rest in generate_compositions(n - i, k - 1):
                    yield (i,) + rest

    for x in generate_compositions(5, 11):
        k = [xi + 1 for xi in x]
        
        if time.time() > deadline:
            break
            
        e = calc_edges(k)
        if e > best_edges:
            best_edges = e
            best_k = k
            
    if best_k is None:
        best_k = [1]*11
        for _ in range(5):
            best_gain = -1
            best_idx = -1
            for i in range(11):
                gain = sum(best_k[j] for j in base_adj[i])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            best_k[best_idx] += 1

    # Construct the final graph
    role_vertices = [[] for _ in range(11)]
    current_id = 0
    for i in range(11):
        for _ in range(best_k[i]):
            role_vertices[i].append(current_id)
            current_id += 1
            
    edges = []
    for i, j in base_edges:
        for u in role_vertices[i]:
            for v in role_vertices[j]:
                edges.append([u, v])
                
    return edges
