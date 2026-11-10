import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_bottleneck_partition_v2(n, edges, seed_heur, mode='min'):
        adj_rem = [set() for _ in range(n)]
        for u, v in edges:
            adj_rem[u].add(v)
            adj_rem[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            target_edge = None
            
            if mode == 'min':
                # Find edge with min common neighbors in remaining graph
                min_common = float('inf')
                for e in remaining_edges:
                    u, v = e
                    common = len(adj_rem[u] & adj_rem[v])
                    if common < min_common:
                        min_common = common
                        target_edge = e
                        if common == 0:
                            break
            else: # mode == 'max'
                # Find edge with max common neighbors in remaining graph
                max_common = -1
                for e in remaining_edges:
                    u, v = e
                    common = len(adj_rem[u] & adj_rem[v])
                    if common > max_common:
                        max_common = common
                        target_edge = e
                        # No early break here, we want the max
        
            if target_edge is None:
                target_edge = next(iter(remaining_edges))
            
            u, v = target_edge
            # Find maximal clique containing u, v
            C = [u, v]
            P = adj_rem[u] & adj_rem[v]
            
            while P:
                # Pick vertex in P with max degree in P
                candidates = list(P)
                # To speed up, we can just pick one with max degree
                # Using a simple loop to find max degree
                max_d = -1
                best_cands = []
                for x in candidates:
                    d = len(adj_rem[x] & P)
                    if d > max_d:
                        max_d = d
                        best_cands = [x]
                    elif d == max_d:
                        best_cands.append(x)
                
                w = rng_heur.choice(best_cands)
                
                C.append(w)
                P = P & adj_rem[w]
            
            # Remove edges of C from remaining_edges and adj_rem
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        adj_rem[a].discard(b)
                        adj_rem[b].discard(a)
            
            cliques.append(C)
            
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        while time.time() < time_limit:
            # Alternate between min and max modes to diversify search
            # Or just use different seeds with min mode?
            # The prompt suggests "min_common" heuristic mode.
            # Let's stick to min mode but vary seed.
            # Or maybe alternate?
            # Let's try alternating to explore different parts of the solution space.
            mode = 'min' if attempt % 2 == 0 else 'max'
            
            heur_seed = seed + attempt * 1000
            cliques = greedy_bottleneck_partition_v2(n, edges, heur_seed, mode)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            
            attempt += 1
                
        if best_cliques is None:
            # Fallback: single edge cliques
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time
    t1_limit = start_time + 30.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
