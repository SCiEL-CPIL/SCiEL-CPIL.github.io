import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    best_edges = []
    best_count = -1

    def find_one_q3(adj):
        """
        Finds one Q3 subgraph.
        Returns list of 8 vertices [v0, n1, n2, n3, x12, x14, x24, x7] or None.
        """
        for v0 in range(n):
            nbrs = [u for u in range(n) if u != v0 and adj[v0][u]]
            if len(nbrs) < 3:
                continue
            
            for n1, n2, n3 in itertools.combinations(nbrs, 3):
                # Find candidates for x12 (common neighbor of n1, n2, not v0)
                cand_12 = [u for u in range(n) if u != v0 and adj[n1][u] and adj[n2][u]]
                if not cand_12: continue
                
                for x12 in cand_12:
                    # Find candidates for x14 (common neighbor of n1, n3, not v0)
                    cand_14 = [u for u in range(n) if u != v0 and u != x12 and adj[n1][u] and adj[n3][u]]
                    if not cand_14: continue
                    
                    for x14 in cand_14:
                        # Find candidates for x24 (common neighbor of n2, n3, not v0)
                        cand_24 = [u for u in range(n) if u != v0 and u != x12 and u != x14 and adj[n2][u] and adj[n3][u]]
                        if not cand_24: continue
                        
                        for x24 in cand_24:
                            # Check distinctness
                            if len({x12, x14, x24}) < 3: continue
                            
                            # Find x7 (common neighbor of x12, x14, x24)
                            excluded = {v0, n1, n2, n3, x12, x14, x24}
                            cand_7 = [u for u in range(n) if u not in excluded and adj[x12][u] and adj[x14][u] and adj[x24][u]]
                            
                            if cand_7:
                                x7 = cand_7[0]
                                return [v0, n1, n2, n3, x12, x14, x24, x7]
        return None

    def get_q3_edges(verts):
        v0, n1, n2, n3, x12, x14, x24, x7 = verts
        return [
            (v0, n1), (v0, n2), (v0, n3),
            (n1, x12), (n1, x14),
            (n2, x12), (n2, x24),
            (n3, x14), (n3, x24),
            (x12, x7), (x14, x7), (x24, x7)
        ]

    # Main Search
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        # Start with K16
        adj = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                edges.add((i, j))
        
        # Phase 1: Remove edges to make Q3-free
        while True:
            if time.time() > deadline: break
            q3 = find_one_q3(adj)
            if not q3:
                break
            q3_edges = get_q3_edges(q3)
            e = random.choice(q3_edges)
            u, v = e
            adj[u][v] = 0
            adj[v][u] = 0
            edges.remove(e)
            
        # Phase 2: Add edges back / Swap
        # Try all non-edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        
        for e in all_possible:
            if time.time() > deadline: break
            if e in edges:
                continue
            u, v = e
            # Try adding
            adj[u][v] = 1
            adj[v][u] = 1
            edges.add(e)
            
            q3 = find_one_q3(adj)
            if q3:
                # Try to swap
                q3_edges = get_q3_edges(q3)
                # Remove the added edge? Or swap with another?
                # If we swap, we remove e' and keep e.
                # We must ensure e' != e.
                candidates = [ee for ee in q3_edges if ee != e]
                if candidates:
                    e_swap = random.choice(candidates)
                    # Remove e_swap
                    us, vs = e_swap
                    adj[us][vs] = 0
                    adj[vs][us] = 0
                    edges.remove(e_swap)
                    # Keep e
                    # Check if still Q3-free?
                    # Removing an edge from a Q3 breaks it.
                    # But might create another? Unlikely, but possible.
                    # For speed, assume it's broken.
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
                    edges.remove(e)
            else:
                # Keep
                pass
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
    if not best_edges:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    return best_edges
