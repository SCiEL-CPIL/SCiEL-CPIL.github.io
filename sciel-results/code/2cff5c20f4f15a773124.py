import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 20
    range_n = range(n)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """Check if adding edge (u, v) keeps graph K3,3-free."""
        # For each w adjacent to v, check if u and w have >= 2 common neighbors
        mask_v = adj[v]
        for w in range_n:
            if w == u or w == v:
                continue
            if (mask_v >> w) & 1:
                common_uw = (adj[u] & adj[w]).bit_count()
                if common_uw >= 2:
                    return False
        
        # For each w adjacent to u, check if v and w have >= 2 common neighbors
        mask_u = adj[u]
        for w in range_n:
            if w == u or w == v:
                continue
            if (mask_u >> w) & 1:
                common_vw = (adj[v] & adj[w]).bit_count()
                if common_vw >= 2:
                    return False
        
        return True

    def greedy_build():
        """Build a K3,3-free graph using greedy edge addition with random order."""
        adj = [0] * n
        edges = []
        order = all_edges[:]
        random.shuffle(order)
        
        for u, v in order:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        
        return edges, adj

    def local_search_add(edges, adj):
        """Improve a K3,3-free graph by adding all possible edges."""
        current_edges = edges[:]
        current_adj = adj[:]
        edge_set = set(current_edges)
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline - 5:
                    break
                if (u, v) in edge_set:
                    continue
                
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    edge_set.add((u, v))
                    improved = True
        
        return current_edges, current_adj

    def perturb(edges, adj, k=2):
        """Remove k random edges to escape local optimum."""
        current_edges = edges[:]
        current_adj = adj[:]
        
        # Remove k random edges
        for _ in range(k):
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edges.pop(idx)
            
        return current_edges, current_adj

    best_edges = []
    best_count = 0
    
    # Phase 1: Iterated Local Search
    while time.time() < deadline - 10:
        # Start with a greedy build
        edges, adj = greedy_build()
        
        # Local search to maximize edges
        edges, adj = local_search_add(edges, adj)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        # Perturb and search again
        # Try a few perturbations
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            p_edges, p_adj = perturb(edges, adj, k=random.randint(1, 3))
            p_edges, p_adj = local_search_add(p_edges, p_adj)
            
            if len(p_edges) > best_count:
                best_count = len(p_edges)
                best_edges = p_edges[:]
            
            # If we found a better solution, continue from there
            if len(p_edges) > len(edges):
                edges, adj = p_edges, p_adj
            else:
                # If not, break inner loop and restart outer
                break
                
    return [[u, v] for u, v in best_edges]
