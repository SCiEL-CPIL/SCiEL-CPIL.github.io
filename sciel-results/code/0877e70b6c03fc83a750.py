import random
import time

def construct():
    n = 20
    # Initialize with a random graph or bipartite
    # Let's start with a bipartite K10,10 and then optimize
    # Or just random
    adj = [[False] * n for _ in range(n)]
    
    # Start with a random graph with p=0.4
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.4:
                adj[i][j] = True
                adj[j][i] = True

    def get_neighbors(u):
        return [v for v in range(n) if adj[u][v]]

    def is_k33_free():
        # Check all triples
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    # Common neighbors of u, v, w
                    common = [x for x in range(n) if adj[u][x] and adj[v][x] and adj[w][x]]
                    if len(common) >= 3:
                        return False
        return True

    def find_k33():
        # Returns a K3,3 subgraph if exists, else None
        # We can just find a triple with 3 common neighbors
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    common = [x for x in range(n) if adj[u][x] and adj[v][x] and adj[w][x]]
                    if len(common) >= 3:
                        # Found K2,3, but we need K3,3?
                        # Wait, if u,v,w have 3 common neighbors x1, x2, x3,
                        # then {u,v,w} and {x1,x2,x3} form a K3,3?
                        # Yes, all edges between the two sets exist.
                        # So this IS a K3,3.
                        return (u, v, w, common[0], common[1], common[2])
        return None

    def check_add_edge(u, v):
        # Check if adding (u, v) creates a K3,3
        # u and v must be in opposite partitions of the K3,3
        # So we look for a2, a3 in N(v)\{u} and b2, b3 in N(u)\{v}
        # such that a2, a3 connected to b2, b3
        
        cu = [x for x in range(n) if adj[u][x] and x != v]
        cv = [x for x in range(n) if adj[v][x] and x != u]
        
        # We need a2, a3 in cv such that |N(a2) intersect N(a3) intersect cu| >= 2
        # To speed up, we can precompute N(a) as bitsets?
        # n=20, so bitsets are integers.
        
        # Let's use bitsets for speed
        # This function will be called many times, so it needs to be fast.
        # But for n=20, even lists might be okay if we limit calls.
        
        # Optimization: Use bit masks
        # Let's refactor to use bit masks in the main loop
        pass

    # Refactor to use bit masks for speed
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        adj_mask[i] = mask

    def get_neighbors_mask(u):
        return adj_mask[u]

    def count_common_neighbors_mask(u, v, w):
        return bin(get_neighbors_mask(u) & get_neighbors_mask(v) & get_neighbors_mask(w)).count("1")

    def is_k33_free_mask():
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    if count_common_neighbors_mask(u, v, w) >= 3:
                        return False
        return True

    def find_k33_mask():
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    common = get_neighbors_mask(u) & get_neighbors_mask(v) & get_neighbors_mask(w)
                    if bin(common).count("1") >= 3:
                        # Extract 3 bits
                        bits = [i for i in range(n) if (common >> i) & 1]
                        return (u, v, w, bits[0], bits[1], bits[2])
        return None

    def check_add_edge_mask(u, v):
        cu_mask = adj_mask[u] & ~(1 << v)
        cv_mask = adj_mask[v] & ~(1 << u)
        
        # Iterate over pairs in cv
        # cv has at most 18 bits
        cv_list = [i for i in range(n) if (cv_mask >> i) & 1]
        
        for i in range(len(cv_list)):
            a2 = cv_list[i]
            for j in range(i+1, len(cv_list)):
                a3 = cv_list[j]
                # Common neighbors of a2 and a3
                common_a2_a3 = adj_mask[a2] & adj_mask[a3]
                # Intersect with cu
                common_in_cu = common_a2_a3 & cu_mask
                if bin(common_in_cu).count("1") >= 2:
                    return False # Adding edge creates K3,3
        return True

    # Local Search
    best_edges = 0
    best_adj = [0] * n
    
    start_time = time.time()
    max_time = 110 # Safety margin
    
    # Initial cleanup
    while True:
        k33 = find_k33_mask()
        if k33 is None:
            break
        u, v, w, x1, x2, x3 = k33
        # Remove a random edge from the K3,3
        # Edges are between {u,v,w} and {x1,x2,x3}
        edges = [(u, x1), (u, x2), (u, x3),
                 (v, x1), (v, x2), (v, x3),
                 (w, x1), (w, x2), (w, x3)]
        edge_to_remove = random.choice(edges)
        a, b = edge_to_remove
        adj_mask[a] &= ~(1 << b)
        adj_mask[b] &= ~(1 << a)
        
        if time.time() - start_time > max_time:
            break

    # Record best
    current_edges = sum(bin(m).count("1") for m in adj_mask) // 2
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = adj_mask[:]

    # Try to add edges
    while time.time() - start_time < max_time:
        # Try to add a random non-edge
        improved = False
        for _ in range(100): # Try 100 random additions
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or adj_mask[u] & (1 << v):
                continue
            
            if check_add_edge_mask(u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                improved = True
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj_mask[:]
                break # Restart loop to try more
        
        if not improved:
            # No more edges can be added?
            # Maybe try removing an edge to allow more later?
            # Or just break if we've tried a few times
            # Let's try a random restart if stuck
            # Or just accept and break
            # For now, break
            break

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
