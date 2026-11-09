import random
import time

def construct():
    n = 16
    vertices = list(range(n))
    edges = []
    edge_set = set()
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edge_count = 0
    best_edges = []
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    while time.time() - start_time < time_limit:
        random.shuffle(all_edges)
        current_edges = []
        current_edge_set = set()
        
        # Adjacency list for fast lookup
        adj = [set() for _ in range(n)]
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a Q3
            creates_q3 = False
            
            # Get neighbors
            Nu = adj[u]
            Nv = adj[v]
            
            # We need a, b in Nu and c, d in Nv
            # such that (a,c) and (b,d) are in current_edge_set
            # and there exists e in Na Nb and w in Ne Nc Nd
            
            # Optimization: Pre-filter pairs (a,b) in Nu and (c,d) in Nv
            # Actually, just iterate
            
            Nu_list = list(Nu)
            Nv_list = list(Nv)
            
            # If degrees are small, this is fast
            for a in Nu_list:
                for b in Nu_list:
                    if a == b: continue
                    # Check if (a,b) is an edge? No, a,b are neighbors of u, not necessarily connected.
                    # In Q3, a and b are not connected.
                    
                    for c in Nv_list:
                        if c in (u, v, a, b): continue
                        # Check (a, c)
                        if c not in adj[a]: continue
                        
                        for d in Nv_list:
                            if d == c: continue
                            if d in (u, v, a, b): continue
                            # Check (b, d)
                            if d not in adj[b]: continue
                            
                            # Check if a, b, c, d are distinct and not u, v
                            # Already checked c, d not in u,v,a,b.
                            # Need a != b (done), c != d (done).
                            # Need a,b,c,d distinct from each other?
                            # a,b from Nu, c,d from Nv.
                            # If graph is bipartite, Nu and Nv disjoint? Not necessarily.
                            # But u, v are distinct.
                            # a,b are neighbors of u. c,d neighbors of v.
                            # If a=c, then a is neighbor of u and v.
                            # In Q3, a is neighbor of u, c is neighbor of v.
                            # a and c are distinct vertices in Q3.
                            # So we require a,b,c,d distinct.
                            
                            if len({a,b,c,d}) < 4: continue
                            
                            # Find e in Na Nb
                            # e must be distinct from u,v,a,b,c,d
                            Na = adj[a]
                            Nb = adj[b]
                            common_e = Na & Nb
                            # Remove forbidden vertices
                            forbidden = {u, v, a, b, c, d}
                            common_e -= forbidden
                            
                            for e in common_e:
                                # Find w in Ne Nc Nd
                                Ne = adj[e]
                                Nc = adj[c]
                                Nd = adj[d]
                                common_w = Ne & Nc & Nd
                                # w must be distinct from u,v,a,b,c,d,e
                                forbidden_w = {u, v, a, b, c, d, e}
                                common_w -= forbidden_w
                                
                                if common_w:
                                    creates_q3 = True
                                    break
                            if creates_q3:
                                break
                        if creates_q3:
                            break
                    if creates_q3:
                        break
                if creates_q3:
                    break
            
            if not creates_q3:
                current_edges.append((u, v))
                current_edge_set.add((u, v))
                current_edge_set.add((v, u))
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = current_edges[:]
            
    return best_edges
