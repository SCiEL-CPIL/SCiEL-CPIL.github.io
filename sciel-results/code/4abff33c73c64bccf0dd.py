import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        if not edge_set:
            return []

        # Build adjacency for C vertices
        # We need to know N_C(v) for each v in I
        # And we know C is a clique.
        
        # Identify edges for C-C
        cc_edges = set()
        ci_edges = set()
        for u, v in edge_set:
            if u < k and v < k:
                cc_edges.add((u, v))
            else:
                # One is in C, one in I
                # Ensure u < k for consistency, though set handles it
                ci_edges.add((u, v))
                
        # Candidates
        candidates = []
        
        # 1. The core clique C
        if k >= 2:
            candidates.append(list(range(k)))
            
        # 2. Cliques for each v in I
        # We need N_C(v)
        # We can derive this from ci_edges
        # For each v in I, find all u in C such that (u,v) in ci_edges
        neighbors_in_c = {v: set() for v in range(k, n)}
        for u, v in ci_edges:
            if u < k:
                neighbors_in_c[v].add(u)
            else:
                neighbors_in_c[u].add(v) # v is in C here? No, v is in I.
                # Wait, the loop `for u, v in ci_edges`
                # If u < k, v >= k. So u in C, v in I.
                # If u >= k, v < k. So u in I, v in C.
                # My condition `if u < k` handles the first case.
                # Else, u is in I, v is in C.
                # So neighbors_in_c[u].add(v) is correct.
        
        for v in range(k, n):
            ns = neighbors_in_c[v]
            if ns:
                # Clique is ns + [v]
                clique = list(ns) + [v]
                candidates.append(clique)
                
        # Greedy Set Cover
        # Precompute edge sets for candidates
        cand_edge_sets = []
        for clique in candidates:
            es = set()
            # Optimization: 
            # For C, es is cc_edges.
            # For K_v, es is edges within ns (subset of cc_edges) + edges between v and ns (subset of ci_edges).
            # We can compute this efficiently.
            
            if clique == list(range(k)):
                es = cc_edges
            else:
                # It's a K_v
                v = clique[-1]
                ns = set(clique[:-1])
                # Edges within ns
                # These are a subset of cc_edges
                # Edges between v and ns
                # These are a subset of ci_edges
                es = set()
                # Add edges within ns
                # Since ns is subset of C, and C is clique, all pairs in ns are edges.
                # We can just generate them and intersect with cc_edges? 
                # Or just generate them, they are all in cc_edges.
                ns_list = list(ns)
                for i in range(len(ns_list)):
                    for j in range(i+1, len(ns_list)):
                        es.add((ns_list[i], ns_list[j]))
                # Add edges between v and ns
                for u in ns:
                    es.add((u, v))
                    
            cand_edge_sets.append((clique, es))
            
        remaining = set(edge_set)
        selected = []
        
        while remaining:
            best_idx = -1
            best_count = 0
            best_es = set()
            
            for i, (clique, es) in enumerate(cand_edge_sets):
                # Count uncovered edges
                # es & remaining
                count = len(es & remaining)
                if count > best_count:
                    best_count = count
                    best_idx = i
                    best_es = es
                    
            if best_idx == -1:
                # Fallback: should not happen
                break
                
            selected.append(cand_edge_sets[best_idx][0])
            remaining -= best_es
            
        return selected

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            results.append([])
            continue
        res = solve_instance(*spec)
        results.append(res)
        
    final_results = []
    for r in results:
        final_results.append([list(clique) for clique in r])
        
    return final_results
