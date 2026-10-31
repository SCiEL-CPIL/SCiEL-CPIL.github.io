<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110  # 10s safety margin
    
    # Moore bound for k=3, g=16 is 95. Start from 96.
    # Known best is 936.
    # We search circulant graphs C_n(1, -1, s)
    
    # Helper to check girth of C_n(1, -1, s)
    def check_girth_circulant(n, s, g):
        # Steps are 1, -1, s, -s
        # We want to know if there is a cycle of length < g
        # BFS from 0
        dist = [-1] * n
        dist[0] = 0
        q = deque()
        q.append(0)
        
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= g - 1:
                continue
            
            # Neighbors: u+1, u-1, u+s, u-s
            neighbors = [(u + 1) % n, (u - 1) % n, (u + s) % n, (u - s) % n]
            for v in neighbors:
                if v == 0:
                    # Found a cycle of length d + 1
                    if d + 1 < g:
                        return False
                    else:
                        # Since BFS explores in increasing distance, 
                        # this is the shortest cycle through 0.
                        # If it's >= g, no shorter cycle exists.
                        return True
                elif dist[v] == -1:
                    dist[v] = d + 1
                    q.append(v)
        
        # If we exit loop, we didn't find a cycle < g?
        # But graph is connected, so 0 must be reachable from neighbors of 0?
        # Actually, if we didn't return True, it means we didn't hit 0 from a neighbor
        # with d < g-1.
        # This implies girth >= g.
        return True

    # Search for n and s
    # n must be even for 3-regular? No, 3-regular requires sum of degrees 3n to be even, so n must be even.
    
    # Start from Moore bound
    min_n = 2 * (k * (k-1) ** (g//2 - 1)) 
    # For k=3, g=16: 2 * (3 * 2^7) = 2 * 384 = 768?
    # Moore bound formula: 1 + k sum_{i=0}^{g/2-1} (k-1)^i
    # g=16, g/2 = 8.
    # 1 + 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 1 + 3 * 255 = 766.
    # So min_n = 766. Next even is 768.
    
    # However, we can try smaller n if we are lucky? No, Moore bound is a lower bound.
    # So start n = 768.
    
    # But wait, the Moore bound is for the *existence* of a graph.
    # It's a lower bound on the number of vertices.
    # So any graph must have n >= 766.
    
    # Let's start from 768 and go up.
    # Step 2 to keep n even.
    
    # To improve chances, we can also try some random s values first for each n.
    
    for n in range(768, 2000, 2):
        if time.time() > deadline:
            break
            
        # Try various s
        # s must be such that s != 1, s != n-1
        # Also s and n-s are symmetric, so we can just try s in 2..n//2
        
        # Heuristic: try random s first, then systematic?
        # Systematic is safer for finding the *first* valid one if we want minimal n.
        # But we want *any* valid graph with small n.
        # Let's try a mix.
        
        # Try s values that are likely to have large girth.
        # Often s ~ n/3 or n/4 works well.
        
        # Generate a list of s to try
        # We'll just iterate s from 2 to n//2
        # But to speed up, we can skip s that cause obvious short cycles.
        # e.g. s=2, s=3, s=4 often cause short cycles.
        
        found = False
        # Try a subset of s first: random ones
        # But deterministic is better for reproducibility.
        
        # Let's just iterate s. 
        # To avoid long runs, we can limit attempts per n?
        # No, we need to find *one*.
        
        for s in range(2, n // 2 + 1):
            if time.time() > deadline:
                break
            
            # Skip s that create 4-cycles easily?
            # If s == 2: 4-cycle.
            # If s == 3: 4-cycle.
            # If s == 4: 5-cycle?
            # Let's just run the check.
            
            if check_girth_circulant(n, s, g):
                # Construct edges
                edges = set()
                for i in range(n):
                    # Edges (i, i+1), (i, i-1), (i, i+s)
                    # Add unique edges
                    edges.add((min(i, (i+1)%n), max(i, (i+1)%n)))
                    edges.add((min(i, (i-1)%n), max(i, (i-1)%n)))
                    edges.add((min(i, (i+s)%n), max(i, (i+s)%n)))
                
                # Verify degree 3 and simplicity
                # The construction guarantees 3-regular if s is valid.
                # s != 1, s != -1 (mod n). s in 2..n//2 ensures s != 1, s != n-1.
                # Also s != -s (mod n) => 2s != 0 => s != n/2.
                # If s == n/2, then (i, i+s) is same as (i, i-s).
                # So we must skip s = n/2.
                
                if s == n // 2:
                    continue
                    
                return n, list(edges)
        
    # If no graph found in range, raise error
    raise ValueError("Could not construct a valid graph")
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110
    
    # Moore bound calculation
    half_g = g // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(half_g))
    
    # n must be even for 3-regular
    start_n = moore_bound
    if start_n % 2 != 0:
        start_n += 1
        
    for n in range(start_n, 5000, 2):
        if time.time() > deadline - 5:
            break
            
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            try:
                G = random_regular_graph(k, n, seed=random.randint(0, 10**9))
                n_comp, _ = connected_components(G)
                if n_comp != 1:
                    continue
                
                edges_set = set()
                G_coo = G.tocoo()
                for i, j in zip(G_coo.row, G_coo.col):
                    if i == j:
                        continue
                    if i < j:
                        edges_set.add((i, j))
                    else:
                        edges_set.add((j, i))
                        
                if _check_girth(n, edges_set, g):
                    return n, list(edges_set)
                    
            except Exception:
                continue
                
    raise ValueError("Could not construct a valid graph")

def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = g // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if v == parent[u]:
                        continue
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
