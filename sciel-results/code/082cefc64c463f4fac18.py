import time

def construct(k, g):
    # Petersen graph: 10 vertices, 15 edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    # Build adjacency for Petersen
    adj = [[] for _ in range(10)]
    for i, (u, v) in enumerate(petersen_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    def get_thue_morse(n):
        tm = [0] * n
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    tm = get_thue_morse(100)
    
    start_time = time.time()
    
    # Enumerate all closed walks of length L in Petersen for L = 5 to g-1
    # and record the sequence of edge indices
    
    def find_closed_walks(max_len):
        """Find all closed walks of length up to max_len starting from vertex 0."""
        # Actually, we need all closed walks from all vertices
        # But by symmetry, we can start from vertex 0 and also from other vertices
        # Let's do it properly: for each start vertex, find closed walks
        
        walks = {}  # length -> list of edge sequences
        
        for start_v in range(10):
            # DFS from start_v
            stack = [(start_v, start_v, [])]
            while stack:
                v, start, path = stack.pop()
                if len(path) > 0 and v == start and len(path) >= 5:
                    if len(path) not in walks:
                        walks[len(path)] = []
                    walks[len(path)].append(path[:])
                if len(path) < max_len:
                    for (u, eidx) in adj[v]:
                        # Avoid immediate backtracking
                        if path and path[-1] == eidx:
                            continue
                        stack.append((u, start, path + [eidx]))
        
        return walks
    
    # This might be expensive. Let's just check the lift directly with a smarter method.
    
    for m in range(40, 150):
        if time.time() - start_time > 115:
            break
        
        patterns = [
            [tm[i] * (i + 1) % m for i in range(15)],
            [(tm[i] * (i + 1) + tm[(i + 7) % 15]) % m for i in range(15)],
            [(tm[i] * (i + 1) + tm[(i + 3) % 15] * (i + 2)) % m for i in range(15)],
        ]
        
        for vol in patterns:
            # Check if any closed walk of length < g has voltage sum 0 mod m
            # We'll do BFS on the lift graph but with early termination
            
            n_vertices = 10 * m
            adj_lift = [[] for _ in range(n_vertices)]
            
            for i, (u, v) in enumerate(petersen_edges):
                w = vol[i] % m
                for s in range(m):
                    u_id = u * m + s
                    v_id = v * m + (s + w) % m
                    adj_lift[u_id].append(v_id)
                    adj_lift[v_id].append(u_id)
            
            # Check girth: BFS from vertex 0,1,2,3,4 (one per coset representative)
            # Actually, we need to be more careful. Let's check from all 10 coset reps.
            
            girth_ok = True
            for start_coset in range(10):
                start = start_coset * m  # vertex (start_coset, 0)
                visited = {start: (0, -1)}
                queue = [(start, 0, -1)]
                
                while queue:
                    node, dist, parent = queue.pop(0)
                    if dist >= g - 1:
                        continue
                    for neighbor in adj_lift[node]:
                        if neighbor == parent:
                            continue
                        if neighbor in visited:
                            cycle_len = dist + 1 + visited[neighbor][0]
                            if cycle_len < g:
                                girth_ok = False
                                break
                        else:
                            visited[neighbor] = (dist + 1, node)
                            queue.append((neighbor, dist + 1, node))
                    if not girth_ok:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                edges = []
                for i, (u, v) in enumerate(petersen_edges):
                    w = vol[i] % m
                    for s in range(m):
                        u_id = u * m + s
                        v_id = v * m + (s + w) % m
                        if u_id < v_id:
                            edges.append((u_id, v_id))
                        else:
                            edges.append((v_id, u_id))
                
                return n_vertices, edges
    
    # Fallback
    m = 100
    vol = [tm[i] * (i + 1) % m for i in range(15)]
    n_vertices = 10 * m
    edges = []
    for i, (u, v) in enumerate(petersen_edges):
        w = vol[i] % m
        for s in range(m):
            u_id = u * m + s
            v_id = v * m + (s + w) % m
            if u_id < v_id:
                edges.append((u_id, v_id))
            else:
                edges.append((v_id, u_id))
    return n_vertices, edges
