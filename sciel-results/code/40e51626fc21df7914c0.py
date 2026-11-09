import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    range_n = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v using bitmasks."""
        common = adj[u] & adj[v]
        return bin(common).count('1')
    
    def is_k33_free(adj):
        """Check if graph is K3,3-free: no two vertices have 3+ common neighbors."""
        for u in range_n:
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    return False
        return True
    
    def greedy_build(seed_order=None):
        """Build a K3,3-free graph using greedy edge addition."""
        adj = [0] * n
        edges = []
        
        if seed_order is None:
            order = all_edges[:]
            random.shuffle(order)
        else:
            order = seed_order[:]
            random.shuffle(order)
        
        for u, v in order:
            # Tentatively add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if this creates a K3,3: any pair of vertices now has >=3 common neighbors
            # Only need to check pairs involving u or v
            valid = True
            for w in range_n:
                if w == u or w == v:
                    continue
                if count_common_neighbors(adj, u, w) >= 3:
                    valid = False
                    break
                if count_common_neighbors(adj, v, w) >= 3:
                    valid = False
                    break
            
            if valid:
                edges.append([u, v])
            else:
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return edges, adj
    
    def local_search(edges, adj):
        """Improve a K3,3-free graph via local search (add/remove edges)."""
        current_edges = edges[:]
        current_adj = adj[:]
        current_count = len(current_edges)
        edge_set = set((min(e), max(e)) for e in current_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding edges
            for u, v in all_edges:
                if (u, v) in edge_set:
                    continue
                # Check if adding (u,v) keeps graph K3,3-free
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                valid = True
                for w in range_n:
                    if w == u or w == v:
                        continue
                    if count_common_neighbors(current_adj, u, w) >= 3:
                        valid = False
                        break
                    if count_common_neighbors(current_adj, v, w) >= 3:
                        valid = False
                        break
                
                if valid:
                    current_edges.append([u, v])
                    edge_set.add((u, v))
                    current_count += 1
                    improved = True
                else:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
            
            # Try removing one edge and adding two others (2-opt style)
            if not improved:
                for i in range(len(current_edges)):
                    if time.time() > deadline - 5:
                        break
                    u, v = current_edges[i]
                    # Remove edge (u,v)
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges.pop(i)
                    edge_set.discard((u, v))
                    current_count -= 1
                    
                    # Try to add two new edges
                    added = 0
                    for u2, v2 in all_edges:
                        if added >= 2:
                            break
                        if (u2, v2) in edge_set:
                            continue
                        if u2 == u and v2 == v:
                            continue
                        
                        current_adj[u2] |= (1 << v2)
                        current_adj[v2] |= (1 << u2)
                        
                        valid = True
                        for w in range_n:
                            if w == u2 or w == v2:
                                continue
                            if count_common_neighbors(current_adj, u2, w) >= 3:
                                valid = False
                                break
                            if count_common_neighbors(current_adj, v2, w) >= 3:
                                valid = False
                                break
                        
                        if valid:
                            current_edges.append([u2, v2])
                            edge_set.add((u2, v2))
                            current_count += 1
                            added += 1
                        else:
                            current_adj[u2] &= ~(1 << v2)
                            current_adj[v2] &= ~(1 << u2)
                    
                    # Restore removed edge if we couldn't improve
                    if added < 1:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges.append([u, v])
                        edge_set.add((u, v))
                        current_count += 1
                    elif current_count > best_count:
                        improved = True
        
        return current_edges, current_adj, current_count
    
    # Phase 1: Multiple greedy builds with local search
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        edges, adj = greedy_build()
        edges, adj, count = local_search(edges, adj)
        
        if count > best_count:
            best_count = count
            best_edges = edges
    
    return best_edges
