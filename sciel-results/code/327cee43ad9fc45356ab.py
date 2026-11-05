import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 180
    vertices = list(range(1, N + 1))
    
    triples = []
    triples_by_vertex = {v: [] for v in vertices}
    degrees = {v: 0 for v in vertices}
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if math.gcd(x, y) != 1:
                continue
            max_val = max(x, y) * (x + y)
            if max_val > N:
                break
            max_k = N // max_val
            for k in range(1, max_k + 1):
                a = k * x * y
                b = k * x * (x + y)
                c = k * y * (x + y)
                triple = (a, b, c)
                triples.append(triple)
                triples_by_vertex[a].append((b, c))
                triples_by_vertex[b].append((a, c))
                triples_by_vertex[c].append((a, b))
                degrees[a] += 1
                degrees[b] += 1
                degrees[c] += 1
                
    def greedy(order):
        S = set()
        for v in order:
            valid = True
            for u, w in triples_by_vertex[v]:
                if u in S and w in S:
                    valid = False
                    break
            if valid:
                S.add(v)
        return S

    best_set = set()
    best_size = 0
    
    # Strategy 1: Descending value
    order1 = sorted(vertices, reverse=True)
    S = greedy(order1)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # Strategy 2: Ascending degree
    order2 = sorted(vertices, key=lambda v: degrees[v])
    S = greedy(order2)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # Strategy 3: Random
    while time.time() - start_time < deadline - 2.0:
        order = vertices[:]
        random.shuffle(order)
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return list(best_set)
