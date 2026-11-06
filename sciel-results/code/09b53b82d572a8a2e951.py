import time

def construct():
    # Build the 4-iterated Mycielski graph of K2.
    # G0 = K2 (n=2, alpha=1)
    # G1 = M(G0) = C5 (n=5, alpha=2)
    # G2 = M(G1) (n=11, alpha=3)
    # G3 = M(G2) (n=23, alpha=4)
    # G4 = M(G3) (n=47, alpha=5)
    # This graph is triangle-free and has alpha=5 < 10.
    # We will take a subgraph of 40 vertices to maximize n within the limit n <= 41.
    # Since any subgraph of a triangle-free graph is triangle-free, and alpha(subgraph) <= alpha(graph) = 5 < 10,
    # this subgraph is valid.
    
    # Start with K2
    n = 2
    adj = [0] * n
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    # Apply Mycielski construction 4 times
    for _ in range(4):
        new_n = 2 * n + 1
        new_adj = [0] * new_n
        
        # 1. Preserve original edges
        for i in range(n):
            new_adj[i] = adj[i]
        
        # 2. Add copy vertices and their edges
        # Vertex u' (index n+u) is connected to all neighbors of u in the original graph
        for u in range(n):
            u_prime = n + u
            neighbors = adj[u]
            while neighbors:
                lsb = neighbors & -neighbors
                v = lsb.bit_length() - 1
                new_adj[u_prime] |= (1 << v)
                new_adj[v] |= (1 << u_prime)
                neighbors ^= lsb
                
        # 3. Add the root vertex and its edges
        # Root (index 2*n) is connected to all copy vertices
        root = 2 * n
        for u in range(n):
            u_prime = n + u
            new_adj[root] |= (1 << u_prime)
            new_adj[u_prime] |= (1 << root)
            
        adj = new_adj
        n = new_n
        
    # Now n=47. We need to reduce to n=40.
    # We keep vertices 0 to 39.
    target_n = 40
    if n > target_n:
        # Create a mapping from new indices to old indices
        # We keep the first target_n vertices
        kept_vertices = list(range(target_n))
        # Map old index to new index
        old_to_new = {v: i for i, v in enumerate(kept_vertices)}
        
        new_n = target_n
        new_adj = [0] * new_n
        
        # Copy edges among kept vertices
        for i in range(target_n):
            old_i = kept_vertices[i]
            # Iterate through neighbors of old_i in the full graph
            neighbors = adj[old_i]
            while neighbors:
                lsb = neighbors & -neighbors
                old_j = lsb.bit_length() - 1
                neighbors ^= lsb
                if old_j in old_to_new:
                    new_j = old_to_new[old_j]
                    if i < new_j:
                        new_adj[i] |= (1 << new_j)
                        new_adj[new_j] |= (1 << i)
        
        adj = new_adj
        n = new_n

    # Extract edges for the output format
    edges = []
    for i in range(n):
        a = adj[i]
        while a:
            lsb = a & -a
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            a ^= lsb
            
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    result = construct()
    print(result['n'])
