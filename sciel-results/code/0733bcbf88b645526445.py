import time
import random

def construct():
    n = 20
    start_time = time.time()
    # Use a conservative deadline to ensure we return before the 120s kill
    deadline = start_time + 100.0 

    def is_valid_after_add(adj, u, v):
        """
        Checks if adding edge (u,v) to adj creates a K3,3.
        Only checks triples involving u or v, as only their neighbor sets change.
        """
        # Triples containing u: {u, a, b}
        # a, b are distinct, not u
        for a in range(n):
            if a == u:
                continue
            for b in range(a + 1, n):
                if b == u:
                    continue
                # Check common neighbors of u, a, b
                # adj[u] already includes v because we set it before calling this
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        # Triples containing v but not u: {v, a, b}
        # a, b are distinct, not u, not v
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                # Check common neighbors of v, a, b
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """
        Greedily adds edges in random order.
        Returns the number of edges in the final adj.
        """
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid_after_add(adj, u, v):
                pass # Keep edge
            else:
                # Rollback
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    best_adj = [0] * n
    best_count = 0
    current_adj = [0] * n
    
    # Initial greedy fill from empty graph
    current_count = greedy_fill(current_adj)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]

    # Local search / Restart strategy
    # Since greedy_fill is fast, we can do many restarts.
    # We also try a "kick" strategy: remove a few random edges, then re-greedy.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        current_adj = best_adj[:]
        
        # Randomly remove 1 to 5 edges to escape local optima
        num_removals = random.randint(1, 5)
        edges_to_remove = []
        # Get current edges
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    edges_to_remove.append((u, v))
        
        if not edges_to_remove:
            continue
            
        random.shuffle(edges_to_remove)
        for u, v in edges_to_remove[:num_removals]:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        
        current_count = greedy_fill(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
