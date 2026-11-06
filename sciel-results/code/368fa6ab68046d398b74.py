import itertools
import random
import time

def construct():
    # Vertices 0..15
    all_edges = set(itertools.combinations(range(16), 2))
    
    def find_q3(graph_edges):
        # Returns a set of edges of a Q3 subgraph, or None
        for S in itertools.combinations(range(16), 8):
            S_set = set(S)
            edges_in_S = [e for e in graph_edges if e[0] in S_set and e[1] in S_set]
            if len(edges_in_S) < 12:
                continue
            
            deg = {v: 0 for v in S}
            for u, v in edges_in_S:
                deg[u] += 1
                deg[v] += 1
            if sum(1 for d in deg.values() if d >= 3) < 4:
                continue
            
            # Backtracking search
            for s0 in S:
                n0 = [v for v in S if v != s0 and (min(s0,v), max(s0,v)) in graph_edges]
                if len(n0) < 3:
                    continue
                
                for n1, n2, n4 in itertools.permutations(n0, 3):
                    cand_3 = [v for v in S if v not in [s0, n1, n2, n4] and (min(n1,v), max(n1,v)) in graph_edges and (min(n2,v), max(n2,v)) in graph_edges]
                    cand_5 = [v for v in S if v not in [s0, n1, n2, n4] and (min(n1,v), max(n1,v)) in graph_edges and (min(n4,v), max(n4,v)) in graph_edges]
                    cand_6 = [v for v in S if v not in [s0, n1, n2, n4] and (min(n2,v), max(n2,v)) in graph_edges and (min(n4,v), max(n4,v)) in graph_edges]
                    
                    if not cand_3 or not cand_5 or not cand_6:
                        continue
                        
                    for c3 in cand_3:
                        for c5 in cand_5:
                            for c6 in cand_6:
                                if c3 == c5 or c3 == c6 or c5 == c6:
                                    continue
                                used = {s0, n1, n2, n4, c3, c5, c6}
                                remaining = [v for v in S if v not in used]
                                for c7 in remaining:
                                    if (min(c3,c7), max(c3,c7)) in graph_edges and \
                                       (min(c5,c7), max(c5,c7)) in graph_edges and \
                                       (min(c6,c7), max(c6,c7)) in graph_edges:
                                        q3_edges = {(s0,n1), (s0,n2), (s0,n4), 
                                                     (n1,c3), (n1,c5), 
                                                     (n2,c3), (n2,c6), 
                                                     (n4,c5), (n4,c6), 
                                                     (c3,c7), (c5,c7), (c6,c7)}
                                        return q3_edges
            return None
        return None

    deadline = time.time() + 110
    best_G = set()
    best_count = 0
    
    # Start with K_3,13 as a safe initial guess
    A = [0, 1, 2]
    B = list(range(3, 16))
    init_G = set()
    for a in A:
        for b in B:
            init_G.add((min(a, b), max(a, b)))
            
    best_G = set(init_G)
    best_count = len(init_G)
    
    G = set(all_edges)
    
    while time.time() < deadline:
        # Phase 1: Remove edges to make Q3-free
        while time.time() < deadline:
            q3 = find_q3(G)
            if q3 is None:
                break
            # Remove a random edge from q3
            e = random.choice(list(q3))
            G.remove(e)
        
        # Phase 2: Try to add edges
        if len(G) > best_count:
            best_count = len(G)
            best_G = set(G)
            
        # Try to add missing edges
        missing = list(all_edges - G)
        random.shuffle(missing)
        for e in missing:
            if time.time() >= deadline:
                break
            G.add(e)
            q3_new = find_q3(G)
            if q3_new is not None:
                G.remove(e)
            else:
                if len(G) > best_count:
                    best_count = len(G)
                    best_G = set(G)
        
        # Restart with a different random seed or strategy?
        # For now, just repeat.
        # To avoid getting stuck, we can shuffle the removal order.
        # But the current loop already does that implicitly.
        
    return [list(e) for e in best_G]
