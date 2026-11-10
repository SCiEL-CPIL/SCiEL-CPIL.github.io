import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    limit = 55.0 # 55s safety margin
    
    n = 128
    vertices = list(range(1, n + 1))
    
    # Precompute gcds for speed? 
    # gcd is fast in python math module.
    
    # Find all bad triples
    bad_triples = []
    # We need a, b, c distinct.
    # a < b < c
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, n + 1):
                g_ac = gcd(a, c)
                if g_ac != g_ab:
                    continue
                g_bc = gcd(b, c)
                if g_bc == g_ab:
                    bad_triples.append((a, b, c))
    
    # Map vertex to list of triple indices
    vertex_to_triples = {v: [] for v in vertices}
    for i, t in enumerate(bad_triples):
        for v in t:
            vertex_to_triples[v].append(i)
    
    # Greedy removal
    active_triples = [True] * len(bad_triples)
    active_vertices = set(vertices)
    
    # Degrees
    degrees = {v: len(vertex_to_triples[v]) for v in vertices}
    
    # We need to update degrees as we remove vertices.
    # When we remove v, all triples containing v become inactive.
    # For each other vertex u in those triples, its degree decreases by 1.
    
    # To be efficient, we can just recalculate or maintain a count.
    # Since N is small, we can just find max degree by scanning active vertices each time.
    
    while True:
        # Check if any active triples remain
        # We can keep a count of active triples
        # Or just check if we can find a max degree > 0
        
        # Find vertex with max degree among active vertices
        max_deg = -1
        max_v = None
        for v in active_vertices:
            if degrees[v] > max_deg:
                max_deg = degrees[v]
                max_v = v
        
        if max_deg <= 0:
            break
            
        # Remove max_v
        active_vertices.remove(max_v)
        
        # Update triples
        for i in vertex_to_triples[max_v]:
            if active_triples[i]:
                active_triples[i] = False
                t = bad_triples[i]
                # Update degrees of other vertices in this triple
                for u in t:
                    if u != max_v:
                        degrees[u] -= 1
        
        # Time check
        if time.time() - start_time > limit:
            break
            
    return list(active_vertices)
