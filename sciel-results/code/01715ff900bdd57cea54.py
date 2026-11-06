import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 40
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def creates_c6(u, v, neighbors):
        # Check for simple path u-a-x2-b-v
        # a in N(u), x2 in N(a), b in N(v), x2 in N(b)
        # a != v, b != u, a != b, x2 != u, x2 != v
        
        # Optimization: Pre-compute N(v) and N(u) are sets.
        # Iterate a in N(u)
        for a in neighbors[u]:
            if a == v:
                continue
            for x2 in neighbors[a]:
                if x2 == u or x2 == v:
                    continue
                # Check if there is b in N(v) such that x2 in N(b) and b != u and b != a
                # We can break early if found
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    if x2 in neighbors[b]:
                        return True
        return False

    def greedy_build(seed_val):
        random.seed(seed_val)
        edges = all_edges[:]
        random.shuffle(edges)
        
        neighbors = [set() for _ in range(n)]
        edge_list = []
        
        for u, v in edges:
            if not creates_c6(u, v, neighbors):
                neighbors[u].add(v)
                neighbors[v].add(u)
                edge_list.append([u, v])
                
        return edge_list

    # Time one run
    if time.time() > deadline:
        return []
        
    t0 = time.time()
    test_edges = greedy_build(0)
    t1 = time.time()
    time_per_run = t1 - t0
    if time_per_run < 0.01:
        time_per_run = 0.01

    remaining = deadline - time.time()
    max_runs = int(remaining / (time_per_run * 1.1))
    max_runs = max(1, min(max_runs, 5000))
    
    best_count = len(test_edges)
    best_result = test_edges
    
    for i in range(1, max_runs):
        if time.time() > deadline:
            break
        current_edges = greedy_build(i)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_result = current_edges
            
    return best_result
