import time
import random
import numpy as np

def construct():
    n = 20
    # Initialize with the seed graph: 5 disjoint K4s
    # This ensures we have a valid baseline of 30 edges
    best_edges = 30
    best_A = np.zeros((n, n), dtype=int)
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u = 4 * k + a
                v = 4 * k + b
                best_A[u, v] = 1
                best_A[v, u] = 1

    start_time = time.time()
    timeout = 140  # Leave 10s for safety
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper to update common neighbors matrix K from adjacency matrix A
    # K[i, j] = sum(A[i, :] * A[j, :])
    # We can maintain K incrementally or recompute it.
    # Recalculating K is O(n^3) = 8000 ops, which is very fast.
    # Updating K incrementally is O(n) per edge change.
    
    def get_common_neighbors(A):
        return A @ A

    def is_k23_free(A):
        K = A @ A
        # Check off-diagonal elements
        # np.fill_diagonal(K, 0)
        return np.max(K - np.diag(np.diag(K))) < 3

    # Peeling algorithm
    def peel_random_graph(p):
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    A[i, j] = 1
                    A[j, i] = 1
        
        K = A @ A
        # We need to find pairs (u, v) with K[u, v] >= 3
        # And remove edges to reduce this
        
        # To speed up, we can just check and remove
        while True:
            # Find all violating pairs
            # K is symmetric
            violations = []
            for u in range(n):
                for v in range(u + 1, n):
                    if K[u, v] >= 3:
                        violations.append((u, v))
            
            if not violations:
                break
            
            # Pick a random violation
            u, v = random.choice(violations)
            # Common neighbors are indices w where A[u, w] == 1 and A[v, w] == 1
            # We can find them by looking at the intersection of neighbors
            # Or simply iterate
            common = []
            for w in range(n):
                if w != u and w != v and A[u, w] == 1 and A[v, w] == 1:
                    common.append(w)
            
            if not common:
                # Should not happen if K[u, v] >= 3
                break
                
            # Pick a random common neighbor w
            w = random.choice(common)
            # Pick a random endpoint x from {u, v} to disconnect from w
            x = random.choice([u, v])
            
            # Remove edge (x, w)
            if A[x, w] == 1:
                A[x, w] = 0
                A[w, x] = 0
                # Update K
                # K[i, j] changes if i==x or j==x or i==w or j==w
                # K[x, y] -= A[w, y] for all y
                # K[y, x] -= A[y, w] for all y
                # K[w, y] -= A[x, y] for all y
                # K[y, w] -= A[y, x] for all y
                # Actually, it's easier to just recompute K or update carefully.
                # Since n=20, recomputing K is cheap (20x20 matrix mult)
                K = A @ A
                
        return A

    # Run multiple restarts
    while time.time() - start_time < timeout:
        # Try different densities
        p = random.uniform(0.3, 0.7)
        A = peel_random_graph(p)
        
        # Verify K2,3 free
        if not is_k23_free(A):
            continue
            
        m = int(np.sum(A)) // 2
        if m > best_edges:
            best_edges = m
            best_A = A.copy()
            
    # Convert matrix to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    return edges
