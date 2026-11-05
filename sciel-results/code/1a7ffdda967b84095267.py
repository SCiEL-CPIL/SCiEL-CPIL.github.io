import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    best_count = 0
    best_edges = []
    
    # Precompute neighbors for faster cycle check?
    # For each edge (u,v), store the list of other 3 edges for each of the 6 faces it belongs to.
    # Actually, an edge in Q7 belongs to 6 faces (one for each other dimension).
    # So for each edge e, we can precompute the sets of 3 edges that would complete a cycle.
    
    edge_to_cycle_conditions = {}
    for (u, v) in all_edges:
        # Find which bit differs
        diff = u ^ v
        i = diff.bit_length() - 1
        conditions = []
        for j in range(d):
            if j == i:
                continue
            u_prime = u ^ (1 << j)
            v_prime = v ^ (1 << j)
            
            e_opp = (min(u_prime, v_prime), max(u_prime, v_prime))
            e_side1 = (min(u, u_prime), max(u, u_prime))
            e_side2 = (min(v, v_prime), max(v, v_prime))
            
            conditions.append((e_opp, e_side1, e_side2))
        edge_to_cycle_conditions[(u, v)] = conditions

    # Run randomized greedy multiple times
    # 1000 iterations should be fast enough.
    # 448 edges * 6 checks * set lookup is very fast.
    num_trials = 1000
    
    for trial in range(num_trials):
        random.shuffle(all_edges)
        chosen = set()
        current_edges = []
        
        for e in all_edges:
            u, v = e
            ok = True
            for (e_opp, e_side1, e_side2) in edge_to_cycle_conditions[e]:
                if e_opp in chosen and e_side1 in chosen and e_side2 in chosen:
                    ok = False
                    break
            if ok:
                chosen.add(e)
                current_edges.append(e)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
            # Early stopping if we reach a very high bound? 
            # Best known is 304, upper 308. If we hit 308, we can stop.
            if best_count >= 308:
                break

    # Convert to list of lists as required
    return [[u, v] for u, v in best_edges]
