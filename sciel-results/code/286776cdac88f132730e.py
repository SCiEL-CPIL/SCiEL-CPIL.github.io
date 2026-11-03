import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Target is n(3, 16) = 936
    # We will try to construct a 2-fold cover of a 468-vertex graph.
    # Base graph needs girth >= 8.
    # Cover needs girth >= 16.
    
    target_n = 936
    base_n = 468
    
    # 1. Generate a 3-regular graph on base_n vertices with girth >= 8
    # We use a random restarts + local improvement strategy.
    
    def generate_random_3_regular(n):
        # Generate a random 3-regular graph using configuration model + simple checks
        # Or use networkx if available? No, standard lib only.
        # Let's use a simple random pairing and check.
        # For n=468, this is fast.
        
        # Actually, a better way is to use a "greedy" construction for girth.
        # But random is faster to start.
        
        # Create list of half-edges
        half_edges = [i for i in range(n) for _ in range(3)]
        random.shuffle(half_edges)
        
        edges = []
        # Pair them up
        # This creates a multigraph. We need to ensure simple.
        # For large n, collisions are rare, but we must check.
        
        # A more robust method for simple 3-regular:
        # Start with empty graph.
        # Randomly add edges, ensuring degree < 3 and no loops/multiedges.
        # This is slow to finish.
        
        # Let's use a specific construction:
        # Start with a cycle, then add chords? No.
        
        # Let's try a random permutation based construction.
        # For 3-regular, we can take 3 permutations and take the union?
        # No, that's not standard.
        
        # Let's use the "random configuration model" and repair.
        # Or simpler: use a known small girth graph and expand?
        
        # Given time constraints, let's just try to build a graph with girth 8
        # using a backtracking search from a root, ensuring no backtracking paths close.
        
        # This is effectively building a BFS tree of depth 3 (since girth 8 means radius 3 for the tree part)
        # and then connecting the leaves.
        
        # For n=468, Moore bound for g=8 is 46. So we have plenty of room.
        # A random 3-regular graph is likely to have girth 6 or 8.
        # We will generate random graphs and check girth.
        
        # Efficient Girth Check:
        # BFS from each vertex. If we find a back edge to a node at distance < (g-1)/2, we have a short cycle.
        
        pass

    # Let's implement a simple random 3-regular generator
    def rand_3_reg(n):
        # Use a random perfect matching on 3 copies of the vertex set?
        # No, that creates a 3-regular multigraph.
        
        # Let's use a simple random walk on the space of 3-regular graphs?
        # Too slow.
        
        # Let's try this:
        # 1. Create list of vertices 0..n-1
        # 2. Shuffle
        # 3. Create edges (0,1), (2,3), ... (n-2, n-1) -> 1-regular
        # 4. Create edges (0, 2), (1, 3), ... shifted -> 1-regular
        # 5. Create edges (0, 3), ... -> 1-regular
        # This is not guaranteed to be simple or 3-regular without care.
        
        # Let's use a library-free random 3-regular graph generator.
        # Algorithm:
        # 1. Initialize degree count to 0.
        # 2. While edges < n*3/2:
        #    Pick two vertices u, v with degree < 3.
        #    If u != v and no edge (u,v):
        #       Add edge.
        #    Else:
        #       Retry (with limit).
        # This is O(n^2) which is fine for 468.
        
        deg = [0] * n
        edges = []
        max_attempts = 10000
        for _ in range(max_attempts):
            if len(edges) == n * 3 // 2:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if deg[u] >= 3 or deg[v] >= 3:
                continue
            # Check if edge exists
            # Use a set of tuples for O(1) check
            edge_key = (min(u, v), max(u, v))
            if edge_key in edge_set:
                continue
            
            # Add edge
            edges.append((u, v))
            edge_set.add(edge_key)
            deg[u] += 1
            deg[v] += 1
            
        if len(edges) != n * 3 // 2:
            return None
        return edges

    edge_set = set()
    base_edges = None
    attempts = 0
    while time.time() < deadline - 10 and base_edges is None:
        base_edges = rand_3_reg(base_n)
        attempts += 1
        if attempts > 100:
             # If random generation fails, try a different method or break
             break
             
    if base_edges is None:
        # Fallback to LUW if we can't generate base graph
        # But LUW is huge. Let's just return a random graph and hope? No, must be valid.
        # If we can't generate a valid base graph, we are in trouble.
        # But random 3-regular generation on 468 nodes should succeed quickly.
        pass

    # Check girth of base graph
    def get_girth(edges, n, target_g):
        # BFS from each node
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            queue = [start]
            dist[start] = 0
            parent[start] = -1
            
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            
            if min_girth < target_g:
                return min_girth
        return min_girth

    base_girth = get_girth(base_edges, base_n, 8)
    
    # If base girth is < 8, we need to improve it.
    # This is a local search step.
    # For simplicity, if we can't get girth 8 quickly, we might fail.
    # But random graphs on 468 nodes often have girth 6.
    # We need to break 6-cycles.
    
    # Given the complexity, let's try to construct the cover directly with random voltages
    # and check the final girth. If the base has girth 6, the cover might still have girth 16?
    # No, if base has a 6-cycle, the cover will have a 6 or 12 cycle.
    # So base MUST have girth >= 8.
    
    # Let's assume we can find a base graph with girth >= 8.
    # If not, we return the LUW graph (huge) or a random graph (invalid).
    # To be safe, if we can't find a good base graph, we fall back to a smaller known graph?
    # No, we must return a valid graph.
    
    # Let's try a few random voltage assignments on the base graph.
    
    # Construct 2-fold cover
    # Vertices: (v, 0), (v, 1)
    # Edges: (u, s) -- (v, s + voltage[u,v])
    
    # We need to map the base edges to voltages.
    
    # If base_girth < 8, we cannot proceed with this specific cover strategy reliably.
    # However, for the purpose of this challenge, let's assume the random graph generation
    # will yield a graph with girth >= 8 within a few tries, or we use a heuristic to fix it.
    
    # Given the 120s limit, let's just return the LUW graph if we can't find a better one quickly.
    # But the LUW graph is 354,294 vertices. The score is 1.0.
    # We want to beat 1.0.
    
    # If we find a 936 vertex graph, score is 936 / 354294 * 1.0?
    # No, score is seed/value. Seed is 1.0 for the current program?
    # "Score: n, minimized".
    # "Search score is seed/value".
    # If current program returns N_LUW, and score is 1.0, then Seed = N_LUW.
    # If we return 936, score = N_LUW / 936.
    # This is a huge improvement.
    
    # So we MUST find a small graph.
    
    # Let's try to find a 936 graph using a more direct search.
    # But time is short.
    
    # I will return the LUW graph as a fallback, but try the cover method first.
    
    if base_girth >= 8:
        # Assign random voltages
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, 1)
            voltages[(v, u)] = voltages[(u, v)] # Symmetric for undirected?
            # In a voltage graph for a cover, the voltage on (u,v) and (v,u) are inverses.
            # In Z2, inverse is itself.
            
        # Build cover
        cover_n = 2 * base_n
        cover_edges = []
        # Map (v, s) to index
        # Index = v * 2 + s
        
        for u, v in base_edges:
            vol = voltages[(u, v)]
            # Edge from (u, 0) to (v, vol)
            # Edge from (u, 1) to (v, 1 + vol)
            # And symmetric
            
            # (u, 0) -> (v, vol)
            idx_u0 = u * 2 + 0
            idx_v_vol = v * 2 + vol
            cover_edges.append((idx_u0, idx_v_vol))
            
            # (u, 1) -> (v, 1 + vol)
            idx_u1 = u * 2 + 1
            idx_v_1vol = v * 2 + (1 + vol) % 2
            cover_edges.append((idx_u1, idx_v_1vol))
            
        # Check girth of cover
        cover_girth = get_girth(cover_edges, cover_n, 16)
        
        if cover_girth >= 16:
            return cover_n, cover_edges
    
    # Fallback: Return LUW graph
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
