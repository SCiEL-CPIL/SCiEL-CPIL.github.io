import time

def construct():
    deadline = time.time() + 110
    n = 16
    
    # Helper: Check if graph is 3-colorable using bitmasks
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            for u in range(n):
                if colors[u] != -1 and (adj[v] >> u) & 1:
                    used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        return (adj[u] & adj[v]) != 0

    # 1. Generate Clebsch Graph
    # Vertices are 5-bit even parity strings
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    clebsch_adj = [0] * n
    clebsch_edges = []
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor_val).count('1') == 4:
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)
                clebsch_edges.append([u_idx, v_idx])
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges) # 40
    best_adj = clebsch_adj

    # Verify baseline
    if not is_3_colorable(best_adj):
        pass # It is 4-chromatic
    else:
        # Should not happen, but if it does, we have a problem
        best_edges = []
        best_count = 0
        best_adj = [0]*n

    # 2. Try to improve via local search (edge swaps)
    # Since Clebsch is maximal TF, we can't add edges.
    # We can try to swap edges to see if we find a denser 4-chromatic TF graph.
    # However, given the theoretical limits, 40 is likely optimal.
    # We will do a few random restarts of maximal TF generation just in case.
    
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def random_maximal_tf():
        adj = [0] * n
        edges = []
        possible_edges = all_pairs[:]
        import random
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if not creates_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return adj, edges

    import random
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        adj, edges = random_maximal_tf()
        
        if len(edges) > best_count:
            if not is_3_colorable(adj):
                best_edges = edges
                best_count = len(edges)
                best_adj = adj
                # If we found something better, we can stop or continue
                # Given 40 is likely max, this loop is mostly for safety.
                break 

    return best_edges
