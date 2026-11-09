import random
import time

def construct():
    n = 20
    best_edges = 0
    best_graph = None
    start_time = time.time()
    timeout = 140.0  # safety margin
    
    # Precompute pairs
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_valid_add(adj, u, v):
        # 1. u and v don't have 3 common neighbors
        if bin(adj[u] & adj[v]).count('1') > 2:
            return False
        
        # 2. For all w in N(v), u and w don't have 2 common neighbors
        #    because v will become a common neighbor
        vw = adj[v]
        while vw:
            lsb = vw & -vw
            w = lsb.bit_length() - 1
            vw ^= lsb
            if bin(adj[u] & adj[w]).count('1') > 1:
                return False
        
        # 3. For all w in N(u), v and w don't have 2 common neighbors
        #    because u will become a common neighbor
        uw = adj[u]
        while uw:
            lsb = uw & -uw
            w = lsb.bit_length() - 1
            uw ^= lsb
            if bin(adj[v] & adj[w]).count('1') > 1:
                return False
        
        return True

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    # Initial greedy with random order
    for _ in range(10000):
        if time.time() - start_time > timeout:
            break
            
        random.shuffle(pairs)
        adj = [0] * n
        for u, v in pairs:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        e_count = count_edges(adj)
        if e_count > best_edges:
            best_edges = e_count
            best_graph = adj.copy()
            
    # Local search on best graph
    # Try to add edges that were skipped?
    # Or try to swap edges?
    # Given the constraint, adding edges is the primary goal.
    # If we get stuck, we can try to remove an edge to allow adding two others.
    # But simple greedy with many restarts usually works well for n=20.
    
    if best_graph is None:
        # Fallback to seed
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    edges = []
    for u in range(n):
        v_mask = best_graph[u]
        while v_mask:
            lsb = v_mask & -v_mask
            v = lsb.bit_length() - 1
            v_mask ^= lsb
            if u < v:
                edges.append([u, v])
    return edges
