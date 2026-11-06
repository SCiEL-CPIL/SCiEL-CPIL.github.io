import numpy as np
import random

def construct():
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = 0
    best_graph = []
    
    # Number of trials
    trials = 50
    
    for _ in range(trials):
        random.shuffle(all_pairs)
        
        # Adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        edges = set()
        
        # Greedy Phase
        for u, v in all_pairs:
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            if len(nu) == 0 or len(nv) == 0:
                safe = True
            else:
                if np.any(adj[np.ix_(nu, nv)]):
                    safe = False
                else:
                    safe = True
            
            if safe:
                edges.add((u, v))
                adj[u, v] = 1
                adj[v, u] = 1
        
        # Local Search Phase
        # Try to improve by removing one edge and re-adding a set of edges
        improved = True
        max_iters = 100
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if not edges:
                break
                
            # Pick a random edge to remove
            e_remove = random.choice(list(edges))
            u_rem, v_rem = e_remove
            
            # Save state
            edges_before = set(edges)
            adj_before = adj.copy()
            
            # Remove edge
            edges.remove(e_remove)
            adj[u_rem, v_rem] = 0
            adj[v_rem, u_rem] = 0
            
            # Try to add edges from the complement
            # We want to try to add the removed edge back, and others
            # Shuffle the missing edges
            missing = [p for p in all_pairs if p not in edges]
            # Prioritize the removed edge? Or just random.
            # To increase chance of finding a better config, we can just shuffle all missing
            random.shuffle(missing)
            
            # Add edges greedily
            for u, v in missing:
                nu = np.where(adj[u] == 1)[0]
                nv = np.where(adj[v] == 1)[0]
                
                if len(nu) == 0 or len(nv) == 0:
                    safe = True
                else:
                    if np.any(adj[np.ix_(nu, nv)]):
                        safe = False
                    else:
                        safe = True
                
                if safe:
                    edges.add((u, v))
                    adj[u, v] = 1
                    adj[v, u] = 1
            
            if len(edges) > len(edges_before):
                improved = True
            else:
                # Revert
                edges = edges_before
                adj = adj_before
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = list(edges)
            
    result = []
    for u, v in best_graph:
        result.append([u, v, 0])
        
    return result
