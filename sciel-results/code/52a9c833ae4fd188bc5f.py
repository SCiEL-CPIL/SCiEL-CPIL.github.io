import random
import time

def construct():
    n = 20
    # adj[u] is a bitmask of neighbors
    adj = [0] * n
    
    # Helper to get list of neighbors
    def get_neighbors(u):
        return [i for i in range(n) if adj[u] & (1 << i)]
    
    # Helper to check if edge (u, v) exists
    def has_edge(u, v):
        return (adj[u] & (1 << v)) != 0
        
    # Helper to add edge
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Helper to remove edge
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Check if we can add edge (u, v)
    # 1. u, v not adjacent
    # 2. common(u, v) <= 2
    # 3. For all y in N(v), common(u, y) <= 1
    # 4. For all y in N(u), common(v, y) <= 1
    def can_add(u, v):
        if u == v:
            return False
        if has_edge(u, v):
            return False
            
        # Check common neighbors of u and v
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
            
        # Check neighbors of v
        # We need to iterate through set bits of adj[v]
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            y = lsb.bit_length() - 1
            if (adj[u] & adj[y]).bit_count() > 1:
                return False
            mask ^= lsb
            
        # Check neighbors of u
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            y = lsb.bit_length() - 1
            if (adj[v] & adj[y]).bit_count() > 1:
                return False
            mask ^= lsb
            
        return True

    def get_edge_count():
        return sum(a.bit_count() for a in adj) // 2

    def get_edges_list():
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                mask ^= lsb
        return edges

    start_time = time.time()
    time_limit = 115.0
    
    # Initial random graph
    # Probability of edge ~ 0.3
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.3:
                add_edge(u, v)
                
    # Try to fix initial graph by removing invalid edges?
    # The random graph might have K2,3.
    # We should start with a valid graph or clean it.
    # Let's clean it.
    # Iterate and remove edges that violate constraint?
    # Simpler: Start with empty graph and add greedily?
    # Or start with Shrikhande?
    
    # Let's try to construct Shrikhande + 4 vertices as initial state.
    # Shrikhande on 16 vertices:
    # Vertices 0..15.
    # Map (x, y) to x * 4 + y.
    # Adjacent if (dx, dy) in { (1,0), (-1,0), (0,1), (0,-1) } 
    # AND NOT (dx, dy) in { (1,1), (1,-1), (-1,1), (-1,-1) }?
    # Wait, Shrikhande is degree 6.
    # The 4x4 grid with edges (1,0), (0,1) is degree 4.
    # Adding diagonals (1,1) makes it degree 8.
    # Shrikhande is the graph obtained from the 4x4 grid by adding edges between (x,y) and (x+1, y+1) and (x+1, y-1) but REMOVING (x, y+1) and (x, y-1)? No.
    
    # Let's just use a random valid graph generator.
    # Start empty, add edges greedily in random order.
    adj = [0] * n
    edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(edges_to_try)
    for u, v in edges_to_try:
        if can_add(u, v):
            add_edge(u, v)
            
    best_count = get_edge_count()
    best_adj = adj[:]
    
    # Local Search
    # State: adj
    # Move: Add edge, Remove edge, Swap edge (remove one, add two)
    
    current_adj = best_adj[:]
    current_count = best_count
    
    while time.time() - start_time < time_limit:
        # Try to add a random edge
        # Pick random u, v
        improved = False
        for _ in range(50): # Try 50 random additions
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if can_add(u, v):
                # Add
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_count += 1
                improved = True
                break
        
        if not improved:
            # Try to remove an edge and add a different one (or two)
            # Pick a random existing edge to remove
            edges_list = get_edges_list()
            if edges_list:
                u, v = random.choice(edges_list)
                # Remove
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_count -= 1
                
                # Try to add up to 2 edges
                added = 0
                for _ in range(100):
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if can_add(u2, v2):
                        current_adj[u2] |= (1 << v2)
                        current_adj[v2] |= (1 << u2)
                        current_count += 1
                        added += 1
                        if added == 2:
                            break
                            
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                elif current_count == best_count:
                    # Keep current state, it's equally good, maybe it's a different local optimum
                    pass
                else:
                    # Rollback? Or keep exploring from this worse state?
                    # Usually, we want to stay in the basin.
                    # If we got worse, we might revert to best_adj or just continue.
                    # Let's revert to best_adj to avoid drift.
                    current_adj = best_adj[:]
                    current_count = best_count
            else:
                # No edges, restart
                adj = [0] * n
                current_adj = adj[:]
                current_count = 0

    # Convert best_adj to list of edges
    final_edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                final_edges.append([u, v])
            mask ^= lsb
            
    return final_edges
