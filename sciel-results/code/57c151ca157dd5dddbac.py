import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    
    best_edges = 0
    best_adj = None
    
    def check_valid(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        # Case 1: v becomes common neighbor of triple {u, a, b}
        # v must be adjacent to a and b
        for a in range(n):
            if a == u or a == v:
                continue
            if not ((adj[v] >> a) & 1):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not ((adj[v] >> b) & 1):
                    continue
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() >= 2:
                    return False
        # Case 2: u becomes common neighbor of triple {v, a, b}
        for a in range(n):
            if a == u or a == v:
                continue
            if not ((adj[u] >> a) & 1):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not ((adj[u] >> b) & 1):
                    continue
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() >= 2:
                    return False
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2
    
    def greedy_build(adj, edge_order):
        """Add edges in given order if valid."""
        for u, v in edge_order:
            if check_valid(adj, u, v):
                add_edge(adj, u, v)
        return adj
    
    def local_search(adj):
        """Try to improve by removing edges and re-adding."""
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        improved = True
        max_iterations = 200
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Build list of current edges
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current_adj[i] >> j) & 1:
                        current_edges.append((i, j))
            
            # Try removing each edge and re-adding
            for e in current_edges:
                if time.time() > deadline - 5.0:
                    break
                u, v = e
                remove_edge(current_adj, u, v)
                
                # Try to re-add edges in random order
                shuffled = all_edges[:]
                random.shuffle(shuffled)
                
                new_count = count_edges(current_adj)
                for eu, ev in shuffled:
                    if (current_adj[eu] >> ev) & 1:
                        continue
                    if check_valid(current_adj, eu, ev):
                        add_edge(current_adj, eu, ev)
                        new_count += 1
                
                if new_count > current_count:
                    current_count = new_count
                    improved = True
                    break
                else:
                    # Restore edge
                    add_edge(current_adj, u, v)
        
        return current_adj, current_count
    
    # Multiple greedy restarts with local search
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        
        # Random edge ordering for greedy
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        adj = [0] * n
        greedy_build(adj, edge_order)
        
        count = count_edges(adj)
        
        # Local search
        if time.time() < deadline - 15.0:
            adj, count = local_search(adj)
        
        if count > best_edges:
            best_edges = count
            best_adj = adj[:]
        
        # If we've found a very good solution, do more local search on it
        if best_edges >= 100 and time.time() < deadline - 20.0:
            # Additional local search passes on best solution
            adj2 = best_adj[:]
            adj2, c2 = local_search(adj2)
            if c2 > best_edges:
                best_edges = c2
                best_adj = adj2
    
    # Reconstruct edge list
    if best_adj is None:
        return []
    
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    
    return res
