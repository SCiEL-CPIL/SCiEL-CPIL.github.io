import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Precompute neighbors for C4 check
    # For an edge (u, v), a C4 is formed if there exists w such that (u, w) and (v, w) are in the set? No.
    # A C4 in Q_d involves vertices u, v, u^bit_i, v^bit_i? No.
    # Vertices: a, b, c, d. Edges: (a,b), (b,c), (c,d), (d,a).
    # In hypercube, a C4 is a face. The vertices are x, x^i, x^j, x^(i^j).
    # Edges: (x, x^i), (x^i, x^(i^j)), (x^(i^j), x^j), (x^j, x).
    # So, to check if adding edge (u, v) creates a C4:
    # u and v differ in exactly one bit, say bit k.
    # The C4s containing (u, v) are formed by choosing another bit j != k.
    # The vertices are u, v, u^j, v^j.
    # The other 3 edges are: (u, u^j), (v, v^j), (u^j, v^j).
    # So (u, v) can be added iff for all j != k, it is NOT the case that all three other edges are already in the set.
    
    # We will use a set for O(1) lookup.
    # We'll run randomized greedy multiple times.
    
    best_set = set()
    best_size = 0
    
    start_time = time.time()
    time_limit = 100 # Leave some buffer
    
    for _ in range(200): # Try 200 randomizations
        if time.time() - start_time > time_limit:
            break
            
        random.shuffle(all_edges)
        current_set = set()
        
        for edge in all_edges:
            u, v = edge
            # Determine the bit index k where u and v differ
            diff = u ^ v
            k = diff.bit_length() - 1
            
            can_add = True
            for j in range(d):
                if j == k:
                    continue
                # Check the other 3 edges of the face
                u_j = u ^ (1 << j)
                v_j = v ^ (1 << j)
                
                e1 = (min(u, u_j), max(u, u_j))
                e2 = (min(v, v_j), max(v, v_j))
                e3 = (min(u_j, v_j), max(u_j, v_j))
                
                if e1 in current_set and e2 in current_set and e3 in current_set:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(edge)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # Try to add more edges greedily to the best set found so far
            # This is a "fill in" pass
            # We iterate through all edges not in best_set and try to add them
            # We do this in a random order to avoid bias
            remaining = [e for e in all_edges if e not in best_set]
            random.shuffle(remaining)
            
            for edge in remaining:
                u, v = edge
                diff = u ^ v
                k = diff.bit_length() - 1
                can_add = True
                for j in range(d):
                    if j == k:
                        continue
                    u_j = u ^ (1 << j)
                    v_j = v ^ (1 << j)
                    e1 = (min(u, u_j), max(u, u_j))
                    e2 = (min(v, v_j), max(v, v_j))
                    e3 = (min(u_j, v_j), max(u_j, v_j))
                    if e1 in best_set and e2 in best_set and e3 in best_set:
                        can_add = False
                        break
                if can_add:
                    best_set.add(edge)
                    best_size += 1

    return [[u, v] for u, v in best_set]
