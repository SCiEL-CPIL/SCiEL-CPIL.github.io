import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    best_edges = None
    best_count = 0
    
    # Known good fallback: Clebsch graph (40 edges, triangle-free, chromatic number 4)
    # Clebsch graph: vertices are 4-bit strings, two vertices adjacent iff Hamming distance = 1
    clebsch_edges = []
    for i in range(16):
        for j in range(i+1, 16):
            d = bin(i ^ j).count('1')
            if d == 1:
                clebsch_edges.append([i, j])
    best_edges = clebsch_edges
    best_count = len(clebsch_edges)
    
    def is_triangle_free(adj):
        """Check if graph with adjacency bitmasks is triangle-free."""
        for u in range(n):
            au = adj[u]
            for v_bit in au:
                v = v_bit.bit_length() - 1
                if v <= u:
                    continue
                # Check common neighbors of u and v
                if adj[u] & adj[v]:
                    return False
        return True
    
    def is_triangle_free_edge(adj, u, v):
        """Check if adding edge (u,v) preserves triangle-freeness."""
        # No common neighbor of u and v
        return (adj[u] & adj[v]) == 0
    
    def is_3_colorable(adj):
        """Exact 3-colorability check via backtracking with bitmasks."""
        colors = [-1] * n  # -1 = uncolored
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        # Map to new ordering
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        # Reorder adjacency
        new_adj = [0] * n
        for i in range(n):
            v = order[i]
            for j in range(n):
                if adj[v] & (1 << j):
                    new_adj[i] |= (1 << pos[j])
        
        def backtrack(idx, color_masks):
            if idx == n:
                return True
            v = idx
            # Try each color
            for c in range(3):
                # Check if color c is available (no neighbor has color c)
                if new_adj[v] & color_masks[c]:
                    continue
                color_masks[c] |= (1 << v)
                if backtrack(idx + 1, color_masks):
                    return True
                color_masks[c] ^= (1 << v)
            return False
        
        # Try all assignments with symmetry breaking: vertex 0 gets color 0
        # Actually, let's just do full backtracking
        return backtrack(0, [0, 0, 0])
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    def edges_to_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def local_search(adj):
        """Try to add edges while maintaining triangle-free and non-3-colorable."""
        adj = list(adj)
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        continue
                    # Check triangle-freeness
                    if not is_triangle_free_edge(adj, u, v):
                        continue
                    # Try adding edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    # Check non-3-colorability
                    if not is_3_colorable(adj):
                        improved = True
                        break
                    else:
                        # Remove edge
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                if improved:
                    break
        return adj
    
    def random_maximal_tf():
        """Generate a random maximal triangle-free graph."""
        # Start with empty graph
        adj = [0] * n
        # Random ordering of vertices
        order = list(range(n))
        random.shuffle(order)
        
        # For each vertex in random order, decide which earlier vertices to connect to
        # while maintaining triangle-freeness
        processed = [False] * n
        for idx in range(n):
            v = order[idx]
            # Find which unprocessed vertices are "available" to connect to
            # A vertex w can connect to v if:
            # 1. w is unprocessed (or we haven't decided yet)
            # 2. No common neighbor between v and w (among already-processed vertices)
            
            # Actually, let's use a different approach:
            # Process vertices in random order. For vertex v, connect to a random
            # subset of already-processed vertices such that no triangle is formed.
            
            pass
        
        # Better approach: start with random edges, then greedily add
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.3:  # Initial random density
                    if is_triangle_free_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Now greedily try to add more edges
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if is_triangle_free_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        return adj
    
    def random_tf_with_density(density):
        """Generate a random triangle-free graph with target edge density."""
        adj = [0] * n
        # Try random edges in random order
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if random.random() < density:
                if is_triangle_free_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Greedily add more edges
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if is_triangle_free_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        return adj
    
    # Main search loop
    iteration = 0
    while (time.time() - start_time) < deadline:
        iteration += 1
        
        # Generate random triangle-free graph
        if iteration % 3 == 1:
            adj = random_tf_with_density(0.5)
        elif iteration % 3 == 2:
            adj = random_tf_with_density(0.7)
        else:
            adj = random_maximal_tf()
        
        # Check if it's not 3-colorable
        if is_3_colorable(adj):
            continue
        
        # Count edges
        edge_count = count_edges(adj)
        if edge_count <= best_count:
            continue
        
        # Try local improvement
        adj_improved = local_search(adj)
        edge_count_improved = count_edges(adj_improved)
        
        if edge_count_improved > best_count:
            best_count = edge_count_improved
            best_edges = adj_to_edges(adj_improved)
    
    # Final check: ensure best_edges is valid
    if best_edges is None:
        best_edges = clebsch_edges
    
    # Verify triangle-free
    adj_final = edges_to_adj(best_edges)
    if not is_triangle_free(adj_final):
        # Fallback to Clebsch
        best_edges = clebsch_edges
    
    return best_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
