import itertools
import random
import time

def construct():
    n = 16
    vertices = list(range(n))
    all_edges = [tuple(sorted(e)) for e in itertools.combinations(vertices, 3)]
    
    best_count = 0
    best_edges = []
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        E = set()
        
        for e in all_edges:
            a, b, c = e
            conflict = False
            rem = [v for v in vertices if v != a and v != b and v != c]
            
            for x2 in rem:
                for y2 in rem:
                    if y2 == x2:
                        continue
                    for z2 in rem:
                        if z2 == x2 or z2 == y2:
                            continue
                        
                        if (a, y2, c) in E and (a, b, z2) in E and (x2, b, c) in E \
                           and (a, y2, z2) in E and (x2, b, z2) in E and (x2, y2, c) in E \
                           and (x2, y2, z2) in E:
                            conflict = True
                            break
                if conflict:
                    break
            if not conflict:
                E.add(e)
                
        if len(E) > best_count:
            best_count = len(E)
            best_edges = list(E)
            
    return [[a, b, c] for a, b, c in best_edges]
