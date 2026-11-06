import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    n = 18
    
    # Precompute masks by popcount
    masks_by_popcount = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        pop = bin(mask).count('1')
        masks_by_popcount[pop].append(mask)
    
    def build_adj_bitmasks(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_k4_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    others = common & ~(1 << k)
                    if adj[k] & others:
                        return False
        return True
    
    def max_triangle_free_induced(adj):
        """Find the size of the largest triangle-free induced subgraph."""
        # Collect all triangles as bitmasks
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                        triangles.append((1 << i) | (1 << j) | (1 << k))
        
        if not triangles:
            return n
        
        # Check from largest possible size downward
        for k in range(n, -1, -1):
            for mask in masks_by_popcount[k]:
                valid = True
                for t in triangles:
                    if (mask & t) == t:
                        valid = False
                        break
                if valid:
                    return k
        return 0
    
    def get_paley_edges(q):
        """Get edges for Paley graph of order q."""
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        
        edges = []
        for u in range(q):
            for v in range(u + 1, q):
                diff = (v - u) % q
                if diff in residues:
                    edges.append([u, v])
        return edges
    
    def random_k4_free_graph(seed=None):
        """Generate a random K4-free graph using greedy random edge addition."""
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * n
        edges = []
        
        possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            creates_k4 = False
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                others = common & ~(1 << k)
                if adj[k] & others:
                    creates_k4 = True
                    break
            
            if not creates_k4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        return edges, adj
    
    def complete_tripartite(n1, n2, n3):
        """Create complete tripartite graph K_{n1,n2,n3}."""
        edges = []
        # Partition: 0..n1-1, n1..n1+n2-1, n1+n2..n-1
        for i in range(n1):
            for j in range(n1, n1 + n2):
                edges.append([i, j])
            for j in range(n1 + n2, n):
                edges.append([i, j])
        for i in range(n1, n1 + n2):
            for j in range(n1 + n2, n):
                edges.append([i, j])
        return edges
    
    def local_search_improve(edges, adj, score):
        """Try to improve by adding edges that reduce the max triangle-free induced subgraph."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            addable_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        
                        creates_k4 = False
                        c = common
                        while c:
                            lsb = c & -c
                            k = lsb.bit_length() - 1
                            c ^= lsb
                            others = common & ~(1 << k)
                            if adj[k] & others:
                                creates_k4 = True
                                break
                        
                        if not creates_k4:
                            addable_edges.append((u, v))
                            
            if not addable_edges:
                break
            
            best_new_score = score
            best_edge = None
            random.shuffle(addable_edges)
            
            for u, v in addable_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                new_score = max_triangle_free_induced(adj)
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_edge = (u, v)
                    
            if best_edge is not None:
                u, v = best_edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                score = best_new_score
                improved = True
        
        return edges, adj, score
    
    best_edges = None
    best_score = float('inf')
    
    # Strategy 1: Paley(17) + isolated vertex
    edges_17 = get_paley_edges(17)
    adj_17 = build_adj_bitmasks(edges_17)
    if is_k4_free(adj_17):
        score_17 = max_triangle_free_induced(adj_17)
        best_edges = edges_17[:]
        best_score = score_17
    
    # Strategy 2: Complete tripartite K_{6,6,6}
    edges_ct = complete_tripartite(6, 6, 6)
    adj_ct = build_adj_bitmasks(edges_ct)
    if is_k4_free(adj_ct):
        score_ct = max_triangle_free_induced(adj_ct)
        if score_ct < best_score:
            best_score = score_ct
            best_edges = edges_ct[:]
    
    # Strategy 3: Random K4-free graphs with local search
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        if attempt == 0 and best_edges is not None:
            edges = best_edges[:]
            adj = build_adj_bitmasks(edges)
            score = best_score
        else:
            edges, adj = random_k4_free_graph(seed=attempt + 100)
            score = max_triangle_free_induced(adj)
        
        edges, adj, score = local_search_improve(edges, adj, score)
        
        if score < best_score:
            best_score = score
            best_edges = edges[:]
    
    # Strategy 4: Start from various structured graphs and refine
    # Try different partition sizes for tripartite
    for n1, n2, n3 in [(5, 6, 7), (6, 5, 7), (7, 5, 6), (5, 7, 6), (6, 7, 5), (7, 6, 5),
                        (4, 7, 7), (7, 4, 7), (7, 7, 4)]:
        if time.time() > deadline:
            break
        edges_ct = complete_tripartite(n1, n2, n3)
        adj_ct = build_adj_bitmasks(edges_ct)
        if is_k4_free(adj_ct):
            score_ct = max_triangle_free_induced(adj_ct)
            edges_ct, adj_ct, score_ct = local_search_improve(edges_ct, adj_ct, score_ct)
            if score_ct < best_score:
                best_score = score_ct
                best_edges = edges_ct[:]
    
    # Strategy 5: Spectral-inspired initialization - use random regular-ish graphs
    for attempt in range(40):
        if time.time() > deadline:
            break
        
        # Create a graph with roughly uniform degree
        adj = [0] * n
        edges = []
        random.seed(attempt + 200)
        
        # Try to create a graph where each vertex has degree around 8-9
        target_degree = 8
        degrees = [0] * n
        
        possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if degrees[u] >= target_degree + 2 or degrees[v] >= target_degree + 2:
                continue
            
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            creates_k4 = False
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                others = common & ~(1 << k)
                if adj[k] & others:
                    creates_k4 = True
                    break
            
            if not creates_k4:
                # Prefer adding edges to lower-degree vertices
                if degrees[u] <= degrees[v] and degrees[u] < target_degree:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
                    degrees[u] += 1
                    degrees[v] += 1
                elif degrees[v] < target_degree:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
                    degrees[u] += 1
                    degrees[v] += 1
        
        score = max_triangle_free_induced(adj)
        edges, adj, score = local_search_improve(edges, adj, score)
        
        if score < best_score:
            best_score = score
            best_edges = edges[:]
    
    if best_edges is None:
        return get_paley_edges(17)
    
    return best_edges
