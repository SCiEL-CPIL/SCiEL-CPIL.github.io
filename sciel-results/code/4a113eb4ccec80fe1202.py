import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    best_score = 18
    best_edges = []
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def remove_k4s(adj):
        # Use bitmask adjacency for faster K4 detection
        adj_bm = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    adj_bm[i] |= (1 << j)
                    adj_bm[j] |= (1 << i)
        
        full_mask = (1 << n) - 1
        while True:
            # Find all K4s using bitmasks
            k4s = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_bm[i] & (1 << j)):
                        continue
                    # Common neighbors of i and j
                    common = adj_bm[i] & adj_bm[j] & full_mask
                    # Remove i and j from common
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    # For each pair of common neighbors that are connected, we have a K4
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        k = lsb.bit_length() - 1
                        cm ^= lsb
                        rest = common & (cm | (1 << k))  # neighbors of k within common after k
                        rest &= adj_bm[k]
                        while rest:
                            lsb2 = rest & (-rest)
                            l = lsb2.bit_length() - 1
                            rest ^= lsb2
                            if adj_bm[k] & (1 << l):
                                k4s.append((i, j, k, l))
            
            if not k4s:
                break
            
            # Count edge frequencies in K4s
            edge_counts = {}
            for quad in k4s:
                u, v, w, x = quad
                for a in range(4):
                    for b in range(a + 1, 4):
                        e = (quad[a], quad[b])
                        edge_counts[e] = edge_counts.get(e, 0) + 1
            
            max_count = max(edge_counts.values())
            candidates = [e for e, c in edge_counts.items() if c == max_count]
            e = random.choice(candidates)
            u, v = e
            adj[u][v] = 0
            adj[v][u] = 0
            adj_bm[u] &= ~(1 << v)
            adj_bm[v] &= ~(1 << u)
        
        return adj

    def max_triangle_free_size_exact(adj):
        # Bitmask-based search with pruning
        # adj_bm[i] = bitmask of neighbors of i
        adj_bm = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    adj_bm[i] |= (1 << j)
                    adj_bm[j] |= (1 << i)
        
        full_mask = (1 << n) - 1
        
        # Precompute for each vertex, the set of vertices that form a triangle with it
        # tri_mask[v] = bitmask of vertices w such that {v, w, x} forms a triangle for some x in current set
        # Actually, let's use a different approach:
        # A set S is triangle-free iff for every edge (u,v) in S, no w in S is adjacent to both u and v.
        
        # Use recursive backtracking with pruning
        best = 0
        
        def is_triangle_free(mask):
            # Check if the set represented by mask is triangle-free
            m = mask
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb
                # For each neighbor v of u in the set
                nv = adj_bm[u] & mask
                while nv:
                    lsb2 = nv & (-nv)
                    v = lsb2.bit_length() - 1
                    nv ^= lsb2
                    if v < u:
                        continue
                    # Check if any common neighbor of u and v is in the set
                    common = adj_bm[u] & adj_bm[v] & mask
                    if common:
                        return False
            return True
        
        # Try all subsets of size k from n down to best+1
        for k in range(n, best, -1):
            found = False
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_triangle_free(mask):
                    best = k
                    found = True
                    break
            if found:
                break
        
        return best

    def greedy_vertex_cover(triangles):
        if not triangles:
            return 0
        freq = [0] * n
        for t in triangles:
            for v in t:
                freq[v] += 1
        
        cover = set()
        remaining_tris = list(triangles)
        
        while remaining_tris:
            t = random.choice(remaining_tris)
            v = max(t, key=lambda x: freq[x])
            cover.add(v)
            remaining_tris = [tt for tt in remaining_tris if v not in tt]
            
        return len(cover)

    def get_triangles(adj):
        triangles = []
        for tri in itertools.combinations(range(n), 3):
            u, v, w = tri
            if adj[u][v] and adj[v][w] and adj[u][w]:
                triangles.append(tri)
        return triangles

    # Main loop
    for p in [0.4, 0.5, 0.6, 0.55, 0.45]:
        while time.time() < deadline:
            adj = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            adj_clean = remove_k4s(adj)
            
            triangles = get_triangles(adj_clean)
            
            c_greedy = greedy_vertex_cover(triangles)
            
            # Lower bound on max triangle-free size: n - c_greedy
            # If this lower bound >= best_score, skip
            if n - c_greedy >= best_score:
                continue
            
            score = max_triangle_free_size_exact(adj_clean)
            
            if score < best_score:
                best_score = score
                best_edges = get_edges_from_adj(adj_clean)
                if best_score <= 7:
                    break

    return best_edges
