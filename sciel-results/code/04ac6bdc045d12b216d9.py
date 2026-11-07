import time
import random

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        cliques = []
        k_edges_covered = set()
        
        # 1. I-cliques
        for v in range(k, n):
            nbs = []
            for u in range(k):
                if (u, v) in edge_set:
                    nbs.append(u)
            
            if not nbs:
                continue
                
            clique = [v] + nbs
            cliques.append(clique)
            
            for i in range(len(nbs)):
                for j in range(i + 1, len(nbs)):
                    u1, u2 = nbs[i], nbs[j]
                    k_edges_covered.add((min(u1, u2), max(u1, u2)))

        # 2. K-cliques for remaining edges
        remaining_edges_set = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in k_edges_covered:
                    remaining_edges_set.add((u, v))
        
        while remaining_edges_set:
            u, v = next(iter(remaining_edges_set))
            current_clique = [u, v]
            
            changed = True
            while changed:
                changed = False
                for w in range(k):
                    if w in current_clique:
                        continue
                    is_connected = True
                    for x in current_clique:
                        edge = (min(w, x), max(w, x))
                        if edge not in remaining_edges_set:
                            is_connected = False
                            break
                    if is_connected:
                        current_clique.append(w)
                        changed = True
            
            cliques.append(current_clique)
            
            for i in range(len(current_clique)):
                for j in range(i + 1, len(current_clique)):
                    edge = (min(current_clique[i], current_clique[j]), max(current_clique[i], current_clique[j]))
                    if edge in remaining_edges_set:
                        remaining_edges_set.remove(edge)

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    start_time = time.time()
    timeout = 115
    
    for spec in specs:
        if time.time() - start_time > timeout:
            # Fallback: return seed-style result for remaining specs to ensure validity
            # But we must return a list of 3 lists.
            # If we haven't finished, we can't easily generate valid partial results for specific specs without running them.
            # Given the speed, this shouldn't happen.
            # We'll just break and pad with empty lists? No, that's invalid.
            # We assume it finishes.
            pass
            
        results.append(solve_instance(*spec))
        
    return results
