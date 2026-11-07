import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edge_set = set(k_edges)
        
        best_cliques = None
        best_count = float('inf')
        
        num_restarts = 100
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            used_edges = set()
            cliques = []
            
            # Random order of I vertices
            vs = list(range(k, n))
            random.shuffle(vs)
            
            for v in vs:
                N_v = neighbors[v]
                if len(N_v) < 2:
                    continue
                
                # Find a maximal matching in N_v using unused edges
                # Since G_v is a clique, any pair is valid.
                # We just need to pick disjoint pairs from N_v such that the edge is not in used_edges.
                
                candidates = list(N_v)
                random.shuffle(candidates)
                
                matched = set()
                # Greedy matching
                i = 0
                while i < len(candidates):
                    u = candidates[i]
                    if u in matched:
                        i += 1
                        continue
                    
                    # Find a partner w for u
                    partner_found = False
                    for j in range(i + 1, len(candidates)):
                        w = candidates[j]
                        if w in matched:
                            continue
                        
                        e = (min(u, w), max(u, w))
                        if e not in used_edges:
                            # Use this edge
                            used_edges.add(e)
                            cliques.append([u, w, v])
                            matched.add(u)
                            matched.add(w)
                            partner_found = True
                            break
                    
                    i += 1
                
                # Cover remaining neighbors with 2-cliques
                for u in N_v:
                    if u not in matched:
                        cliques.append([u, v])
            
            # Cover remaining K-edges
            remaining_k_edges = [e for e in k_edges if e not in used_edges]
            
            if remaining_k_edges:
                # Greedy clique cover
                rem_edges_set = set(remaining_k_edges)
                while rem_edges_set:
                    if time.time() > deadline:
                        break
                    e = next(iter(rem_edges_set))
                    u, w = e
                    clique = {u, w}
                    
                    changed = True
                    while changed:
                        changed = False
                        for x in range(k):
                            if x not in clique:
                                if all((min(x, y), max(x, y)) in rem_edges_set for y in clique):
                                    clique.add(x)
                                    changed = True
                                    break
                    
                    if len(clique) >= 2:
                        cliques.append(sorted(clique))
                        for a, b in combinations(clique, 2):
                            rem_edges_set.discard((min(a, b), max(a, b)))
                    else:
                        cliques.append([u, w])
                        rem_edges_set.discard(e)
            
            clean_cliques = [[int(x) for x in c] for c in cliques]
            count = len(clean_cliques)
            
            if count < best_count:
                best_count = count
                best_cliques = clean_cliques
        
        if best_cliques is None:
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u+1, k):
                    cliques.append([u, w])
            best_cliques = [[int(x) for x in c] for c in cliques]
            
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
