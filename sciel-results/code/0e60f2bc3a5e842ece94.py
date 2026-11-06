# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def has_q3(adj):
        """
        Returns True if the graph represented by adj (list of sets) contains a Q3.
        """
        for v in range(n):
            neighbors = list(adj[v])
            d = len(neighbors)
            if d < 3:
                continue
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        u1, u2, u3 = neighbors[i], neighbors[j], neighbors[k]
                        
                        c12 = adj[u1] & adj[u2]
                        c12.discard(v)
                        
                        c13 = adj[u1] & adj[u3]
                        c13.discard(v)
                        
                        c23 = adj[u2] & adj[u3]
                        c23.discard(v)
                        
                        for w12 in c12:
                            for w13 in c13:
                                for w23 in c23:
                                    verts = {v, u1, u2, u3, w12, w13, w23}
                                    if len(verts) < 7:
                                        continue
                                    
                                    c123 = adj[w12] & adj[w13] & adj[w23]
                                    c123 -= verts
                                    
                                    if c123:
                                        return True
        return False

    def greedy_solve(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        
        # Start with a dense random graph
        p = 0.8
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Phase 1: Remove edges to eliminate all Q3s
        # We need to identify which edges to remove.
        # has_q3 only tells us if one exists.
        # We need a function to find one Q3 and return one of its edges.
        
        def find_one_q3_edge(adj):
            """Returns an edge (u,v) that is part of a Q3, or None if no Q3."""
            for v in range(n):
                neighbors = list(adj[v])
                d = len(neighbors)
                if d < 3:
                    continue
                for i in range(d):
                    for j in range(i + 1, d):
                        for k in range(j + 1, d):
                            u1, u2, u3 = neighbors[i], neighbors[j], neighbors[k]
                            
                            c12 = adj[u1] & adj[u2]
                            c12.discard(v)
                            
                            c13 = adj[u1] & adj[u3]
                            c13.discard(v)
                            
                            c23 = adj[u2] & adj[u3]
                            c23.discard(v)
                            
                            for w12 in c12:
                                for w13 in c13:
                                    for w23 in c23:
                                        verts = {v, u1, u2, u3, w12, w13, w23}
                                        if len(verts) < 7:
                                            continue
                                        
                                        c123 = adj[w12] & adj[w13] & adj[w23]
                                        c123 -= verts
                                        
                                        if c123:
                                            # Found a Q3. Return one of its edges.
                                            # e.g., (v, u1)
                                            return (v, u1)
            return None

        while True:
            edge_to_remove = find_one_q3_edge(adj)
            if edge_to_remove is None:
                break
            u, v = edge_to_remove
            adj[u].remove(v)
            adj[v].remove(u)
            
        # Phase 2: Add edges
        # We want to add as many edges as possible.
        # We try all non-edges in random order.
        # If adding an edge creates a Q3, we skip it.
        # We repeat this process until no more edges can be added.
        
        # To be more effective, we can try to add edges in a specific order or use a heuristic.
        # Simple approach:
        while True:
            added_any = False
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
            random.shuffle(non_edges)
            for u, v in non_edges:
                adj[u].add(v)
                adj[v].add(u)
                if has_q3(adj):
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    added_any = True
            if not added_any:
                break
        
        edge_count = sum(len(s) for s in adj) // 2
        return edge_count, adj

    best_adj = None
    best_count = 0
    
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 115:
        count, adj = greedy_solve(seed)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
        seed += 1
        
    if best_adj is None:
        # Fallback
        return []
        
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges

# EVOLVE-BLOCK-END
