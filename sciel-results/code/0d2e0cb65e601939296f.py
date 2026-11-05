import time
import itertools

def _polarity_graph(q):
    """
    Generates the polarity graph of the projective plane of order q.
    Vertices are points of PG(2, q).
    Two points are adjacent if one lies on the polar of the other.
    This graph is C4-free.
    """
    if q < 2:
        return 0, []
    
    # Points in PG(2, q)
    # 1. Affine points: (x, y, 1)
    # 2. Points at infinity: (x, 1, 0) and (1, 0, 0)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n = len(points)
    
    edges = []
    adj = [[0]*n for _ in range(n)]
    
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i+1, n):
            x2, y2, z2 = points[j]
            # Check if x1*x2 + y1*y2 + z1*z2 == 0 mod q
            if (x1*x2 + y1*y2 + z1*z2) % q == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
                
    return n, edges

def _has_independent_set_size(adj, n, k, forbidden_mask=0, time_limit=5.0):
    """
    Returns True if an independent set of size k exists in the graph defined by adj,
    considering only vertices not in forbidden_mask.
    """
    start_time = time.time()
    
    # Create list of available vertices
    avail = []
    for i in range(n):
        if not (forbidden_mask & (1 << i)):
            avail.append(i)
            
    if len(avail) < k:
        return False
        
    # Bitmask for adjacency
    adj_bits = [0]*n
    for i in range(n):
        bits = 0
        for j in range(n):
            if adj[i][j]:
                bits |= (1 << j)
        adj_bits[i] = bits
        
    # Heuristic: order vertices by degree in the available subgraph
    # Lower degree is better for independent set (less constraints)
    def get_degree(v):
        deg = 0
        for u in avail:
            if u != v and adj[v][u]:
                deg += 1
        return deg
        
    # Sort avail by degree
    avail.sort(key=get_degree)
    
    # Remap for faster access? Or just use the list.
    # We will use a recursive search.
    
    found = [False]
    
    def backtrack(start, current_set, size):
        nonlocal found
        if time.time() - start_time > time_limit:
            return
        if found[0]:
            return
        if size == k:
            found[0] = True
            return
            
        # Pruning: not enough vertices left
        if len(avail) - start < k - size:
            return
            
        for i in range(start, len(avail)):
            if found[0]:
                return
            v = avail[i]
            
            # Check if v is compatible with current_set
            # v must not be adjacent to any u in current_set
            compatible = True
            for u in current_set:
                if adj[v][u]:
                    compatible = False
                    break
            if not compatible:
                continue
            
            # Recurse
            # Next candidates must be > i in avail list
            # And not adjacent to v
            # We can just pass i+1 as start, and the loop will check compatibility
            # But we can prune the list of candidates?
            # For simplicity, just recurse.
            backtrack(i+1, current_set + [v], size + 1)
            
    backtrack(0, [], 0)
    return found[0]

def _check_c4_constraint(adj, n, S):
    """
    Checks if adding a vertex connected to all vertices in S creates a C4.
    Condition: For all u != w in S, N(u) intersection N(w) must be empty.
    """
    S_set = set(S)
    for i, u in enumerate(S):
        for w in S[i+1:]:
            # Check if N(u) and N(w) have any common neighbor
            # N(u) is set of neighbors of u in current graph
            # We just need to check if there exists x such that adj[u][x] and adj[w][x]
            # Iterate x
            for x in range(n):
                if x == u or x == w:
                    continue
                if adj[u][x] and adj[w][x]:
                    return False
    return True

def construct():
    deadline = time.time() + 110  # Leave 10s margin
    
    # Start with PG(2,5)
    n, edges = _polarity_graph(5)
    
    # Build adjacency matrix
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        
    best_n = n
    best_edges = edges[:]
    
    # Function to check if we can add a vertex with neighbor set S
    def can_add_vertex(S):
        # 1. Check C4 constraint
        if not _check_c4_constraint(adj, n, S):
            return False
            
        # 2. Check Independence Number
        # We need alpha(G - N[v]) < 10
        # N[v] in the new graph includes v and S.
        # The vertices remaining are V(G) \ S.
        # We need to check if there is an independent set of size 10 in G[V \ S].
        # If yes, then alpha(G') >= 1 + 10 = 11 (if v is included) or just 10 (if not).
        # Wait, if there is an independent set I of size 10 in G[V\S],
        # then I is an independent set in G'.
        # Also {v} U I is an independent set in G' of size 11.
        # So we MUST NOT have an independent set of size 10 in G[V\S].
        
        forbidden_mask = 0
        for v in S:
            forbidden_mask |= (1 << v)
            
        # Check if independent set of size 10 exists in G[V\S]
        # Time limit for this check should be short
        if _has_independent_set_size(adj, n, 10, forbidden_mask, time_limit=1.0):
            return False
            
        return True

    # Greedy addition loop
    # We try to add vertices one by one.
    # To find S, we try:
    # 1. S = empty (isolated vertex)
    # 2. S = {u} for various u
    # 3. S = {u, w} for various pairs
    # etc.
    
    # To make it more effective, we can try to find S that "covers" the graph well
    # i.e., reduces alpha.
    
    # Simple strategy:
    # Try to add isolated vertices first until we can't.
    # Then try to add vertices with small S.
    
    added_count = 0
    max_additions = 43 - n # Max possible
    
    while added_count < max_additions and time.time() < deadline - 5:
        added = False
        current_n = n
        
        # Try S = empty
        if can_add_vertex([]):
            # Add isolated vertex
            new_v = current_n
            # adj is fixed size? No, we need to expand.
            # This is inefficient with list of lists.
            # Let's use a set of edges or a dynamic structure.
            # Given n <= 43, expanding list of lists is O(n^2) which is fine.
            
            # Expand adj
            new_adj = [[0]*(current_n+1) for _ in range(current_n+1)]
            for i in range(current_n):
                for j in range(current_n):
                    new_adj[i][j] = adj[i][j]
            
            adj = new_adj
            n = current_n + 1
            added_count += 1
            added = True
            if n > best_n:
                best_n = n
                # Rebuild edges from adj
                best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
            continue
            
        # Try S of size 1
        found_s = False
        for u in range(current_n):
            if time.time() > deadline - 5:
                break
            S = [u]
            if can_add_vertex(S):
                # Add vertex connected to u
                new_v = current_n
                new_adj = [[0]*(current_n+1) for _ in range(current_n+1)]
                for i in range(current_n):
                    for j in range(current_n):
                        new_adj[i][j] = adj[i][j]
                new_adj[new_v][u] = 1
                new_adj[u][new_v] = 1
                
                adj = new_adj
                n = current_n + 1
                added_count += 1
                added = True
                found_s = True
                if n > best_n:
                    best_n = n
                    best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
                break
                
        if added:
            continue
            
        # Try S of size 2
        found_s = False
        for u in range(current_n):
            if time.time() > deadline - 5:
                break
            for w in range(u+1, current_n):
                S = [u, w]
                if can_add_vertex(S):
                    new_v = current_n
                    new_adj = [[0]*(current_n+1) for _ in range(current_n+1)]
                    for i in range(current_n):
                        for j in range(current_n):
                            new_adj[i][j] = adj[i][j]
                    new_adj[new_v][u] = 1
                    new_adj[u][new_v] = 1
                    new_adj[new_v][w] = 1
                    new_adj[w][new_v] = 1
                    
                    adj = new_adj
                    n = current_n + 1
                    added_count += 1
                    added = True
                    found_s = True
                    if n > best_n:
                        best_n = n
                        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
                    break
            if found_s:
                break
                
        if added:
            continue
            
        # If no S of size 0, 1, 2 found, try size 3?
        # Or stop?
        # Given the constraints, size 3 might be needed.
        # But let's check time.
        if time.time() > deadline - 20:
            break
            
        found_s = False
        # Random sampling of S of size 3
        import random
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            S = random.sample(range(current_n), 3)
            if can_add_vertex(S):
                new_v = current_n
                new_adj = [[0]*(current_n+1) for _ in range(current_n+1)]
                for i in range(current_n):
                    for j in range(current_n):
                        new_adj[i][j] = adj[i][j]
                for u in S:
                    new_adj[new_v][u] = 1
                    new_adj[u][new_v] = 1
                    
                adj = new_adj
                n = current_n + 1
                added_count += 1
                added = True
                found_s = True
                if n > best_n:
                    best_n = n
                    best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
                break
                
        if not added:
            # Stuck. Try to restart with different seed or break.
            # Since we are deterministic, we just break.
            # But maybe we can try to "shake" the graph?
            # For now, break.
            break

    return {'n': best_n, 'edges': best_edges}
