import random
import time

def count_c4(adj):
    """
    Count the number of C4s in the graph represented by adj.
    adj is a list of integers where adj[i] is a bitmask of neighbors of i.
    """
    total = 0
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            # Intersection of neighbor sets
            common = adj[i] & adj[j]
            k = common.bit_count()
            total += k * (k - 1) // 2
    return total

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = float('inf')
    best_adj = None
    best_edges = None

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only add edges where i < j to avoid duplicates
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def random_graph():
        adj = [0] * n
        edges = set()
        # Generate m random edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:m]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add((u, v))
        return adj, edges

    def local_search(adj, edges):
        current_score = count_c4(adj)
        no_improvement_count = 0
        max_no_improvement = 10000
        
        while no_improvement_count < max_no_improvement:
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            u, v = random.choice(list(edges))
            # Pick a random non-edge to add
            # We need to ensure we don't pick an existing edge
            # Efficient way: pick random i, j, check if edge exists
            added = False
            attempts = 0
            while not added and attempts < 100:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if (x, y) in edges or (y, x) in edges:
                    continue
                # Check if (x,y) is the same as (u,v) to avoid no-op
                if (x == u and y == v) or (x == v and y == u):
                    continue
                added = True
                attempts += 1
            
            if not added:
                continue

            # Perform swap
            # Remove (u, v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            edges.discard((u, v))
            edges.discard((v, u)) # Just in case, though we store (min, max) or set of tuples
            # Actually, let's keep edges as a set of frozensets or sorted tuples for simplicity
            # My edges set above contains (u,v) where u,v are from the loop. 
            # Let's normalize edges to (min, max) for the set.
            
            # Re-normalize logic: 
            # I will store edges as a set of (min(u,v), max(u,v))
            
            # Add (x, y)
            nx, ny = min(x, y), max(x, y)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            edges.add((nx, ny))
            
            new_score = count_c4(adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improvement_count = 0
            else:
                # Revert
                adj[x] ^= (1 << y)
                adj[y] ^= (1 << x)
                edges.discard((nx, ny))
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                nu, nv = min(u, v), max(u, v)
                edges.add((nu, nv))
                
                no_improvement_count += 1
                
        return current_score, adj, edges

    # Main loop with random restarts
    while time.time() < deadline:
        # Generate random initial graph
        adj, edges = random_graph()
        # Normalize edges to (min, max)
        norm_edges = set()
        for u, v in edges:
            norm_edges.add((min(u, v), max(u, v)))
        edges = norm_edges
        
        score, adj, edges = local_search(adj, edges)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_edges = list(edges)
            
            # Optional: if we find 0 or 1, we can stop early? 
            # Min C4 for 91 edges on 31 vertices.
            # ex(31, C4) = 90. So 91 edges must have at least 1 C4.
            # If we find 1, it's likely optimal or near optimal.
            if best_score <= 1:
                break

    if best_adj is None:
        # Fallback
        adj, edges = random_graph()
        norm_edges = set((min(u,v), max(u,v)) for u,v in edges)
        best_edges = list(norm_edges)
        
    return best_edges

