import random
import time

def construct():
    n = 20
    U = list(range(10))
    V = list(range(10, 20))
    
    def is_k33_free_after_add(edge_set, u, v):
        neighbors_u = [x for x in V if (u, x) in edge_set]
        neighbors_v = [x for x in U if (x, v) in edge_set]
        
        if len(neighbors_u) < 2 or len(neighbors_v) < 2:
            return True
            
        for i in range(len(neighbors_v)):
            for j in range(i+1, len(neighbors_v)):
                u1 = neighbors_v[i]
                u2 = neighbors_v[j]
                common_v = 0
                for v_c in neighbors_u:
                    if (u1, v_c) in edge_set and (u2, v_c) in edge_set:
                        common_v += 1
                        if common_v >= 2:
                            return False
        return True

    best_edges = []
    start_time = time.time()
    
    # Heuristic: Start with a random sparse graph, then add edges
    for restart in range(200):
        if time.time() - start_time > 115:
            break
            
        edge_set = set()
        all_edges = [(u, v) for u in U for v in V]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_k33_free_after_add(edge_set, u, v):
                edge_set.add((u, v))
        
        # Local Search: Try to swap edges to increase count? 
        # Or just try to add more edges if we didn't go through all?
        # The loop goes through all edges, so it's a maximal graph.
        
        if len(edge_set) > len(best_edges):
            best_edges = list(edge_set)
            
    return [[u, v] for u, v in best_edges]
