import numpy as np

def construct():
    """
    Construct a K4-free graph on 18 vertices that minimizes the size of the
    largest triangle-free induced subgraph.
    
    Strategy:
    1. Start with a high-symmetry initial graph (complement of Clebsch-like structure or similar).
    2. Use local search to minimize the maximum triangle-free induced subgraph size.
    3. Use spectral properties (second largest eigenvalue) as a tie-breaker or secondary objective.
    """
    n = 18
    vertices = list(range(n))
    
    # Helper: compute adjacency matrix from edge list
    def build_adj(edges):
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    # Helper: check if graph is K4-free
    def is_k4_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    for k in range(j+1, n):
                        if adj[i, k] == 1 and adj[j, k] == 1:
                            for l in range(k+1, n):
                                if adj[i, l] == 1 and adj[j, l] == 1 and adj[k, l] == 1:
                                    return False
        return True
    
    # Helper: compute the size of the largest triangle-free induced subgraph
    def max_triangle_free_size(adj):
        # A subgraph is triangle-free if it contains no triangles.
        # We want the maximum size of a vertex subset with no triangles.
        # This is equivalent to the independence number of the triangle graph.
        # Build the triangle graph: vertices are original vertices, edges connect pairs that form a triangle with a third vertex.
        # Actually, simpler: use branch and bound or bitmask DP.
        # Since n=18, we can use a bitmask approach.
        
        # Precompute for each vertex, the set of vertices that form a triangle with it.
        # Actually, let's precompute the set of triangles.
        
        # A subset S is triangle-free if for every triple (i,j,k) in S, at least one edge is missing.
        # Equivalently, S is an independent set in the "triangle hypergraph".
        
        # Let's use a different approach: 
        # The complement of the triangle-free subgraph problem is the clique cover problem for triangles? No.
        
        # Let's use the fact that a graph is triangle-free iff it has no 3-cliques.
        # We want the largest subset with no 3-cliques.
        
        # Use branch and bound with bitmasks.
        # n=18, so 2^18 = 262144 masks. We can iterate all masks and check if they are triangle-free.
        # But checking each mask for triangles is O(n^3) which is 18^3 = 5832 per mask, total 262144 * 5832 ~ 1.5e9, too slow.
        
        # Better: precompute which masks are triangle-free.
        # Or: use the fact that a mask is triangle-free iff it contains no triangle.
        # Precompute the list of all triangles (triples that form a clique).
        
        triangles = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    for k in range(j+1, n):
                        if adj[i, k] == 1 and adj[j, k] == 1:
                            triangles.append((i, j, k))
        
        if not triangles:
            return n
        
        # Now, we want the largest subset that contains no triangle.
        # This is the independence number of the 3-uniform hypergraph of triangles.
        # For n=18, we can use a greedy + local search or exact branch and bound.
        
        # Let's use a simple branch and bound with memoization on bitmasks.
        # State: (mask of remaining vertices, current size)
        # But this might be slow.
        
        # Alternative: since n=18 is small, we can iterate all masks and check if they are triangle-free.
        # To check if a mask is triangle-free, we can precompute for each triangle whether it's contained in the mask.
        # For each mask, check all triangles: if any triangle is fully contained, it's not triangle-free.
        # Number of triangles is at most C(18,3) = 816.
        # Number of masks is 2^18 = 262144.
        # Total operations: 262144 * 816 ~ 2.1e8, which might be borderline in Python.
        
        # Let's optimize: represent masks as integers. For each triangle (i,j,k), compute the bitmask.
        triangle_masks = []
        for i, j, k in triangles:
            mask = (1 << i) | (1 << j) | (1 << k)
            triangle_masks.append(mask)
        
        # Now, for each mask from 0 to 2^n - 1, check if it contains any triangle.
        # If not, it's triangle-free. Track the maximum size (popcount).
        
        # To speed up, we can use the fact that if a mask is triangle-free, all its subsets are triangle-free.
        # But we want the maximum size, so we can iterate masks in decreasing order of popcount and return the first triangle-free one.
        
        # However, iterating all masks in decreasing popcount order is not straightforward.
        
        # Let's just iterate all masks and keep track of the maximum popcount for triangle-free masks.
        # This is O(2^n * T) where T is the number of triangles.
        
        max_size = 0
        for mask in range(1 << n):
            # Check if mask is triangle-free
            is_tf = True
            for tm in triangle_masks:
                if (mask & tm) == tm:
                    is_tf = False
                    break
            if is_tf:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
                    if max_size == n:  # Can't get better
                        break
        
        return max_size
    
    # Helper: compute the second largest eigenvalue of the adjacency matrix
    def second_largest_eigenvalue(adj):
        eigenvalues = np.linalg.eigvalsh(adj)
        # eigenvalues are in ascending order
        # We want the second largest, which is eigenvalues[-2]
        if len(eigenvalues) >= 2:
            return eigenvalues[-2]
        return 0.0
    
    # Initial graph: use a random regular graph or a known strongly regular graph
    # Let's start with the complement of the Clebsch graph if possible, or a random 5-regular graph.
    # For n=18, a 5-regular graph is a good start.
    
    import random
    random.seed(42)
    
    # Start with a random 5-regular graph
    edges = []
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Generate a random 5-regular graph
    degrees = [0] * n
    while True:
        # Try to add edges to reach degree 5
        incomplete = [i for i in range(n) if degrees[i] < 5]
        if not incomplete:
            break
        random.shuffle(incomplete)
        for i in range(0, len(incomplete) - 1, 2):
            u, v = incomplete[i], incomplete[i+1]
            if adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
                degrees[u] += 1
                degrees[v] += 1
                edges.append([u, v])
    
    # Ensure K4-free
    while not is_k4_free(adj):
        # Remove a random edge from a K4
        # Find a K4
        found = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    for k in range(j+1, n):
                        if adj[i, k] == 1 and adj[j, k] == 1:
                            for l in range(k+1, n):
                                if adj[i, l] == 1 and adj[j, l] == 1 and adj[k, l] == 1:
                                    # Remove edge (i, j)
                                    adj[i, j] = 0
                                    adj[j, i] = 0
                                    edges.remove([i, j])
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    current_score = max_triangle_free_size(adj)
    current_lambda2 = second_largest_eigenvalue(adj)
    
    # Local search
    improved = True
    while improved:
        improved = False
        # Try flipping each edge
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Flip edge (u, v)
            old_val = adj[u, v]
            adj[u, v] = 1 - old_val
            adj[v, u] = 1 - old_val
            
            if old_val == 0:
                # Added edge, check K4-free
                if not is_k4_free(adj):
                    # Revert
                    adj[u, v] = old_val
                    adj[v, u] = old_val
                    continue
            
            new_score = max_triangle_free_size(adj)
            new_lambda2 = second_largest_eigenvalue(adj)
            
            # Accept if better score, or same score with better lambda2
            if new_score < current_score or (new_score == current_score and new_lambda2 < current_lambda2):
                # Update edges list
                if old_val == 0:
                    edges.append([u, v])
                else:
                    edges.remove([u, v])
                current_score = new_score
                current_lambda2 = new_lambda2
                improved = True
            else:
                # Revert
                adj[u, v] = old_val
                adj[v, u] = old_val
    
    return edges
