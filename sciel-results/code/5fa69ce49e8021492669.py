import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # Fast book size calculation using adjacency sets
    def calc_max_book(adj, edges):
        max_bs = 0
        for u, v in edges:
            bs = len(adj[u] & adj[v])
            if bs > max_bs:
                max_bs = bs
        return max_bs
    
    # Local search with edge swaps
    def local_search(edges, adj, max_steps=5000):
        current_score = calc_max_book(adj, edges)
        edge_set = set(map(tuple, edges))
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if current_score <= 7:
                break
                
            # Pick random edge to remove
            idx = random.randint(0, len(edges)-1)
            u, v = edges[idx]
            
            # Pick random non-edge to add
            added = False
            for _ in range(20):
                new_u = random.randint(0, n-1)
                new_v = random.randint(0, n-1)
                if new_u == new_v:
                    continue
                if (new_u, new_v) in edge_set or (new_v, new_u) in edge_set:
                    continue
                
                # Try the swap
                # Remove (u,v)
                adj[u].discard(v)
                adj[v].discard(u)
                edges.pop(idx)
                edge_set.discard((u, v))
                edge_set.discard((v, u))
                
                # Add (new_u, new_v)
                adj[new_u].add(new_v)
                adj[new_v].add(new_u)
                edges.append([new_u, new_v])
                edge_set.add((new_u, new_v))
                edge_set.add((new_v, new_u))
                
                new_score = calc_max_book(adj, edges)
                
                if new_score <= current_score:
                    current_score = new_score
                    added = True
                    break
                else:
                    # Revert
                    adj[new_u].discard(new_v)
                    adj[new_v].discard(new_u)
                    edges.pop()
                    edge_set.discard((new_u, new_v))
                    edge_set.discard((new_v, new_u))
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.insert(idx, [u, v])
                    edge_set.add((u, v))
                    edge_set.add((v, u))
            
            if not added:
                # Try again or break
                continue
                
        return edges, adj, current_score
    
    best_score = float('inf')
    best_edges = None
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Start with a random graph
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible_edges)
        edges = [list(e) for e in possible_edges[:target_edges]]
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        edges, adj, score = local_search(edges, adj, max_steps=2000)
        
        if score < best_score:
            best_score = score
            best_edges = [e[:] for e in edges]
            if best_score <= 7:
                break
    
    if best_edges is not None and best_score < 11:
        return best_edges
    
    # Fallback: current construction with score 11
    edges = []
    for u in range(20):
        for v in range(20, 40):
            if u == 0 and 20 <= v <= 28:
                continue
            edges.append([u, v])
    for i in range(1, 11):
        edges.append([0, i])
    
    return edges
