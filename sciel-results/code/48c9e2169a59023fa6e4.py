def _tensor_product_search(k, g, deadline, current_best_n):
    """
    Create a 4-regular graph as tensor (Kron) product of two cycles,
    then use edge-switching to eliminate short cycles.
    """
    best_n = None
    best_edges = None
    
    # Candidate sizes: a*b should be >= 243 and < current_best_n
    candidates = [
        (17, 15), (15, 17), (16, 16), (18, 14), (14, 18),
        (19, 13), (13, 19), (20, 13), (13, 20), (21, 12), (12, 21),
        (22, 11), (11, 22), (23, 11), (11, 23),
    ]
    
    # Filter: must be < current_best_n and >= 243
    candidates = [(a, b) for a, b in candidates if a * b < current_best_n and a * b >= 243]
    
    if not candidates:
        return None
    
    random.seed(42)
    
    for a, b in candidates:
        if time.time() > deadline - 15:
            break
        
        n = a * b
        
        # Build tensor product: vertex (i,j) -> i*b + j
        # Neighbors of (i,j): (i+1,j), (i-1,j), (i,j+1), (i,j-1)
        adj = [[None] * 4 for _ in range(n)]
        
        for i in range(a):
            for j in range(b):
                v = i * b + j
                adj[v][0] = ((i + 1) % a) * b + j
                adj[v][1] = ((i - 1) % a) * b + j
                adj[v][2] = i * b + ((j + 1) % b)
                adj[v][3] = i * b + ((j - 1) % b)
        
        # Check initial girth
        if _verify_girth_fast(n, adj, g):
            edges = _adj_to_edges(n, adj)
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            continue
        
        # Local Search with Simulated Annealing
        # Cost function: number of short cycles (approximated by count of cycles < g)
        # We use _count_short_cycles as a proxy
        
        current_cost = _count_short_cycles(n, adj, g)
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        
        temperature = 1.0
        cooling = 0.99
        max_iterations = 5000
        
        for it in range(max_iterations):
            if time.time() > deadline - 10:
                break
            if temperature < 0.01:
                break
                
            # Pick a random edge to switch
            u = random.randint(0, n - 1)
            w = random.randint(0, n - 1)
            if u == w or w in adj[u]:
                continue
                
            pos_u = random.randint(0, 3)
            v = adj[u][pos_u]
            
            pos_w = random.randint(0, 3)
            x = adj[w][pos_w]
            
            if x == u or x == v:
                continue
            if x in adj[v]:
                continue
            
            # Perform switch
            adj[u][pos_u] = w
            # Find v's neighbor u
            pos_v = -1
            for p in range(4):
                if adj[v][p] == u:
                    pos_v = p
                    break
            if pos_v == -1:
                adj[u][pos_u] = v
                continue
            adj[v][pos_v] = x
            
            # Find w's neighbor x
            pos_w2 = -1
            for p in range(4):
                if adj[w][p] == x:
                    pos_w2 = p
                    break
            if pos_w2 == -1:
                adj[u][pos_u] = v
                adj[v][pos_v] = u
                continue
            adj[w][pos_w2] = u
            
            # Find x's neighbor w
            pos_x = -1
            for p in range(4):
                if adj[x][p] == w:
                    pos_x = p
                    break
            if pos_x == -1:
                adj[u][pos_u] = v
                adj[v][pos_v] = u
                adj[w][pos_w2] = x
                continue
            adj[x][pos_x] = v
            
            # Verify simplicity
            valid = True
            for vert in [u, v, w, x]:
                seen = set(adj[vert])
                if len(seen) != 4:
                    valid = False
                    break
            if not valid:
                # Undo
                adj[u][pos_u] = v
                adj[v][pos_v] = u
                adj[w][pos_w2] = x
                adj[x][pos_x] = w
                continue
            
            new_cost = _count_short_cycles(n, adj, g)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = [row[:] for row in adj]
                    if best_cost == 0:
                        break
            else:
                # Accept with probability exp(-delta/T)
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / temperature):
                    current_cost = new_cost
                else:
                    # Undo
                    adj[u][pos_u] = v
                    adj[v][pos_v] = u
                    adj[w][pos_w2] = x
                    adj[x][pos_x] = w
            
            temperature *= cooling
        
        # Check best solution found
        if best_cost == 0:
            edges = _adj_to_edges(n, best_adj)
            if _verify_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break
    
    return (best_n, best_edges) if best_n is not None else None


def _count_short_cycles(n, adj, g):
    """Count the number of cycles of length < g."""
    count = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            if du >= (g - 1) // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if du + 1 <= (g - 1) // 2:
                        q.append(v)
                else:
                    if du + dist[v] + 1 < g:
                        count += 1
    return count
