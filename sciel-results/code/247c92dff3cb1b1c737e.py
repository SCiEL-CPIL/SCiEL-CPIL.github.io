import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # Partition sizes
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size) # 122
    
    # Split B into two sets for a bipartite structure within B
    # This ensures G_B is triangle-free
    B1 = B[:15]
    B2 = B[15:]
    
    # We need to select 122 edges between B1 and B2
    # such that max degree <= 8
    # Total capacity: 15*8 + 16*8 = 248. Needed sum of degrees: 244.
    # This is feasible.
    
    B_edges = []
    degree = {v: 0 for v in B}
    
    # Generate all potential edges between B1 and B2
    potential_edges = list(itertools.product(B1, B2))
    
    # To ensure we find a valid solution, we use a greedy approach
    # prioritizing vertices with lower current degrees
    # We will iterate until we have enough edges
    
    # Sort potential edges? 
    # A better way: Use a priority queue or just repeat passes.
    # Given the small size, we can just shuffle and pick valid ones,
    # but to be robust against getting stuck, we use a "least constrained" heuristic.
    
    # Actually, a simple greedy with random restarts is fast.
    # But let's try a deterministic "balanced" approach.
    
    # Initialize a list of available edges
    available = set(potential_edges)
    
    # We will add edges one by one.
    # To keep degrees low, we always pick an edge (u,v) that minimizes deg[u] + deg[v]
    # among available edges.
    
    # For speed, we can just iterate. 122 edges is small.
    
    # Let's use a simple greedy:
    # 1. Shuffle potential edges to break symmetry
    random.shuffle(potential_edges)
    
    # 2. Iterate and add if degree < 8
    # This might fail if we run out of "low degree" pairs.
    # But since capacity is high, it should be fine.
    
    # To be extra safe, if simple greedy fails, we can use a backtracking search
    # or just accept that 122 edges on 31 vertices with D=8 is very dense but feasible.
    
    # Let's try the simple greedy first.
    for u, v in potential_edges:
        if len(B_edges) == needed_in_B:
            break
        if degree[u] < 8 and degree[v] < 8:
            B_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
            
    # If we didn't reach 122, it means we got stuck.
    # This is unlikely, but let's handle it.
    if len(B_edges) < needed_in_B:
        # Try a different random order
        found = False
        for _ in range(10):
            random.shuffle(potential_edges)
            B_edges = []
            degree = {v: 0 for v in B}
            for u, v in potential_edges:
                if len(B_edges) == needed_in_B:
                    break
                if degree[u] < 8 and degree[v] < 8:
                    B_edges.append((u, v))
                    degree[u] += 1
                    degree[v] += 1
            if len(B_edges) == needed_in_B:
                found = True
                break
        if not found:
            # Fallback: Allow degree 9
            B_edges = []
            degree = {v: 0 for v in B}
            for u, v in potential_edges:
                if len(B_edges) == needed_in_B:
                    break
                if degree[u] < 9 and degree[v] < 9:
                    B_edges.append((u, v))
                    degree[u] += 1
                    degree[v] += 1

    # Add B edges to main set
    for u, v in B_edges:
        edges.add((u, v))
        
    # Format output
    result = [[u, v] for u, v in edges]
    
    return result
