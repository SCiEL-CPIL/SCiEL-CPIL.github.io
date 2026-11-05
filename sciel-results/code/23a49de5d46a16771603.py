import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = 110.0
    
    best_edges = []
    best_count = 0
    
    def is_k33_free(edges_set, u, v):
        if u == v:
            return False
        if frozenset((u, v)) in edges_set:
            return False
        
        neighbors_u = {w for w in range(n) if frozenset((u, w)) in edges_set}
        neighbors_v = {w for w in range(n) if frozenset((v, w)) in edges_set}
        
        # Case 1: u and v in different parts
        cand_x = [w for w in neighbors_v if w != u]
        cand_y = [w for w in neighbors_u if w != v]
        
        for i in range(len(cand_x)):
            x1 = cand_x[i]
            for j in range(i+1, len(cand_x)):
                x2 = cand_x[j]
                common_y = []
                for y in cand_y:
                    if frozenset((x1, y)) in edges_set and frozenset((x2, y)) in edges_set:
                        common_y.append(y)
                if len(common_y) >= 2:
                    return False
        
        # Case 2: u and v in same part
        common_neigh_uv = neighbors_u & neighbors_v
        m = len(common_neigh_uv)
        if m >= 3:
            for y1, y2, y3 in itertools.combinations(common_neigh_uv, 3):
                neigh_y1 = {w for w in range(n) if frozenset((y1, w)) in edges_set}
                neigh_y2 = {w for w in range(n) if frozenset((y2, w)) in edges_set}
                neigh_y3 = {w for w in range(n) if frozenset((y3, w)) in edges_set}
                
                common_x = neigh_y1 & neigh_y2 & neigh_y3
                common_x.discard(u)
                common_x.discard(v)
                common_x.discard(y1)
                common_x.discard(y2)
                common_x.discard(y3)
                
                if len(common_x) > 0:
                    return False
        
        return True

    # Randomized Greedy
    for trial in range(200):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        edges_set = set()
        edges_list = []
        
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if is_k33_free(edges_set, u, v):
                edges_set.add(frozenset((u, v)))
                edges_list.append([u, v])
        
        if len(edges_list) > best_count:
            best_count = len(edges_list)
            best_edges = edges_list[:]
            
    # Local Search on best
    if best_edges and time.time() < deadline:
        edges_set = set(frozenset(e) for e in best_edges)
        # Try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if frozenset((u, v)) not in edges_set:
                        if is_k33_free(edges_set, u, v):
                            edges_set.add(frozenset((u, v)))
                            improved = True
                            break
                if improved:
                    break
        
        best_edges = [list(e) for e in edges_set]
        best_count = len(best_edges)

    return best_edges
