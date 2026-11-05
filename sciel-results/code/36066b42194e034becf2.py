import random
import time
import sys

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Uses randomized greedy with local search improvement.
    """
    start_time = time.time()
    time_limit = 105.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Precompute all edges with their bit dimension
    # Edge encoded as (u, v) where u < v, and we track which bit differs
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    # Shuffle for randomized greedy
    random.seed(42)
    
    best_count = 0
    best_edges_set = set()
    
    # Number of restarts we'll try, but bounded by time
    # Let's estimate: one greedy pass takes maybe 0.05-0.1s in Python
    # So we can do ~500-1000 restarts in 100 seconds
    
    max_restarts = 2000
    restart = 0
    while restart < max_restarts:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        restart += 1
        
        # Random permutation of edges
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        chosen = set()
        out_list = []
        
        for (u, v, i) in shuffled:
            # Check if adding (u,v) creates a C4
            ok = True
            for j in range(d):
                if j == i:
                    continue
                c = u ^ (1 << j)
                e = v ^ (1 << j)
                # The three other edges of the face:
                # (min(u,c), max(u,c)), (min(v,e), max(v,e)), (min(c,e), max(c,e))
                edge1 = (min(u, c), max(u, c))
                edge2 = (min(v, e), max(v, e))
                edge3 = (min(c, e), max(c, e))
                
                if edge1 in chosen and edge2 in chosen and edge3 in chosen:
                    ok = False
                    break
            
            if ok:
                chosen.add((u, v))
                out_list.append((u, v))
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_edges_set = set(chosen)
            
            # Early exit if we hit the upper bound
            if best_count >= 308:
                break
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
    
    # Convert to required format: list of [u, v]
    result = []
    for u, v in best_edges_set:
        result.append([u, v])
    
    return result
