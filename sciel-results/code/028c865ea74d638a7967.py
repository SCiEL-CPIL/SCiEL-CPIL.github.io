import time
import random

def construct():
    deadline = time.time() + 110
    n = 16
    
    # Helper: Check if graph is 3-colorable using bitmasks with forward checking
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

    # 2. Try to improve via edge-swap local search
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def random_maximal_tf():
        adj = [0] * n
        edges = []
        possible_edges = all_pairs[:]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if not creates_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return adj, edges

    def local_search(adj, edges, time_limit=5):
        """Try to improve by swapping edges"""
        current_adj = adj[:]
        current_edges = [e[:] for e in edges]
        current_count = len(current_edges)
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            # Try adding an edge
            for u, v in all_pairs:
                if (current_adj[u] >> v) & 1:
                    continue  # Edge already exists
                if creates_triangle(current_adj, u, v):
                    continue  # Would create triangle
                
                # Check if adding this edge keeps graph 3-colorable
                test_adj = current_adj[:]
                test_adj[u] |= (1 << v)
                test_adj[v] |= (1 << u)
                
                if not is_3_colorable(test_adj):
                    # Adding edge makes it 4-chromatic (good!)
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append([u, v])
                    current_count += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing an edge and adding another
            for i, (u1, v1) in enumerate(current_edges[:]):
                # Remove edge (u1, v1)
                test_adj = current_adj[:]
                test_adj[u1] &= ~(1 << v1)
                test_adj[v1] &= ~(1 << u1)
                
                # Try to add a different edge
                for u2, v2 in all_pairs:
                    if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                        continue
                    if (test_adj[u2] >> v2) & 1:
                        continue  # Edge already exists
                    if creates_triangle(test_adj, u2, v2):
                        continue  # Would create triangle
                    
                    # Check if the new graph is 3-colorable
                    test_adj2 = test_adj[:]
                    test_adj2[u2] |= (1 << v2)
                    test_adj2[v2] |= (1 << u1)
                    test_adj2[v2] |= (1 << u2)
                    
                    if not is_3_colorable(test_adj2):
                        # Apply the swap
                        current_adj[u1] &= ~(1 << v1)
                        current_adj[v1] &= ~(1 << u1)
                        current_adj[u2] |= (1 << v2)
                        current_adj[v2] |= (1 << u2)
                        current_edges[i] = [u2, v2]
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_adj, current_edges, current_count

    random.seed(42)
    
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        adj, edges = random_maximal_tf()
        
        if len(edges) >= best_count:
            if not is_3_colorable(adj):
                # Try local search to improve
                improved_adj, improved_edges, improved_count = local_search(adj, edges, time_limit=3)
                if improved_count > best_count:
                    best_edges = improved_edges
                    best_count = improved_count
                    best_adj = improved_adj
                    if improved_count > 40:
                        break  # Found something better than Clebsch
    
    return best_edges
