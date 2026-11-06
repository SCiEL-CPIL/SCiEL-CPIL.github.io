import time
import itertools

def construct():
    start_time = time.time()
    deadline = 150.0 # 150s limit, keep margin
    
    n = 16
    
    # Helper to check if a subset of 8 vertices induces a Q3
    def is_q3_induced(adj, subset):
        # adj is a list of sets or bitmasks
        # subset is a tuple of 8 vertices
        # Check 12 edges
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if subset[j] in adj[subset[i]]:
                    edge_count += 1
        if edge_count != 12:
            return False
            
        # Check 3-regular
        for i in range(8):
            deg = 0
            for j in range(8):
                if i != j and subset[j] in adj[subset[i]]:
                    deg += 1
            if deg != 3:
                return False
                
        # Check distance properties from vertex 0
        v0 = subset[0]
        dist1 = []
        dist2 = []
        dist3 = []
        
        # BFS from v0
        visited = {v0: 0}
        queue = [v0]
        while queue:
            u = queue.pop(0)
            d = visited[u]
            if d == 3:
                continue
            for w in adj[u]:
                if w in subset:
                    if w not in visited:
                        visited[w] = d + 1
                        queue.append(w)
        
        # Now classify
        for v in subset:
            if v == v0:
                continue
            d = visited[v]
            if d == 1:
                dist1.append(v)
            elif d == 2:
                dist2.append(v)
            elif d == 3:
                dist3.append(v)
            
        if len(dist1) != 3 or len(dist2) != 3 or len(dist3) != 1:
            return False
            
        # Check independence of dist1 and dist2
        for i in range(len(dist1)):
            for j in range(i+1, len(dist1)):
                if dist1[j] in adj[dist1[i]]:
                    return False
        for i in range(len(dist2)):
            for j in range(i+1, len(dist2)):
                if dist2[j] in adj[dist2[i]]:
                    return False
                    
        return True

    def find_q3(adj):
        vertices = list(range(n))
        for subset in itertools.combinations(vertices, 8):
            if is_q3_induced(adj, subset):
                return subset
        return None

    def remove_edges_to_make_q3_free(adj):
        # adj is a list of sets
        # Return new adj
        adj = [set(s) for s in adj]
        while True:
            subset = find_q3(adj)
            if subset is None:
                break
            # Remove an edge from the subset
            # Pick the first edge found
            removed = False
            for i in range(8):
                for j in range(i+1, 8):
                    u = subset[i]
                    v = subset[j]
                    if v in adj[u]:
                        adj[u].remove(v)
                        adj[v].remove(u)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break # Should not happen
            if time.time() > deadline - 10:
                break
        return adj

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Generate Complement of Shrikhande Graph
    # Shrikhande is L(K_4,4)
    # Vertices of L(K_4,4) are edges of K_4,4.
    # K_4,4 has partitions A={0,1,2,3}, B={4,5,6,7}
    # Edges are (a,b) for a in A, b in B.
    # Map these 16 edges to vertices 0..15
    # Vertex index = a * 4 + b
    
    # Two vertices (a1,b1) and (a2,b2) in L(K_4,4) are adjacent if the edges (a1,b1) and (a2,b2) in K_4,4 share a vertex.
    # i.e., a1==a2 or b1==b2.
    
    adj_shrikhande = [set() for _ in range(n)]
    for a1 in range(4):
        for b1 in range(4):
            v1 = a1 * 4 + b1
            for a2 in range(4):
                for b2 in range(4):
                    v2 = a2 * 4 + b2
                    if v1 == v2:
                        continue
                    if a1 == a2 or b1 == b2:
                        adj_shrikhande[v1].add(v2)
                        adj_shrikhande[v2].add(v1)
                        
    # Complement
    adj_comp_shrikhande = [set(range(n)) - {i} - adj_shrikhande[i] for i in range(n)]
    
    # Generate Complement of Clebsch Graph
    # Clebsch is halved 5-cube.
    # Vertices are binary strings of length 5 with even weight.
    # There are 16 such strings.
    # Two vertices are adjacent if they differ in 2 bits.
    
    import itertools
    clebsch_vertices = []
    for mask in range(32):
        if bin(mask).count('1') % 2 == 0:
            clebsch_vertices.append(mask)
            
    # Map mask to index 0..15
    mask_to_idx = {m: i for i, m in enumerate(clebsch_vertices)}
    
    adj_clebsch = [set() for _ in range(n)]
    for i, m1 in enumerate(clebsch_vertices):
        for j, m2 in enumerate(clebsch_vertices):
            if i == j:
                continue
            diff = m1 ^ m2
            if bin(diff).count('1') == 2:
                adj_clebsch[i].add(j)
                adj_clebsch[j].add(i)
                
    adj_comp_clebsch = [set(range(n)) - {i} - adj_clebsch[i] for i in range(n)]
    
    candidates = [adj_comp_shrikhande, adj_comp_clebsch]
    
    best_edges = []
    best_count = -1
    
    for adj in candidates:
        # Check if already Q3-free
        if find_q3(adj) is None:
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
        else:
            # Remove edges
            adj_fixed = remove_edges_to_make_q3_free(adj)
            edges = get_edges(adj_fixed)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
                
    if best_edges:
        return best_edges
    else:
        # Fallback to seed
        return [[a, b] for a in range(3) for b in range(3, 16)]

