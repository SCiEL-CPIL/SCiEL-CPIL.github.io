import numpy as np
import random

def construct():
    n = 20
    
    # 1. Spectral Initialization
    # Create a random symmetric matrix
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    
    # Get top eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    top_eigenvector = eigenvectors[:, -1]
    
    # Split vertices into two sets based on the median of the eigenvector values
    # This creates a balanced bipartition
    median_val = np.median(top_eigenvector)
    # Assign vertices with value > median to group 1, others to group 2
    # If ties, handle arbitrarily
    group = np.zeros(n, dtype=int)
    for i in range(n):
        if top_eigenvector[i] > median_val:
            group[i] = 1
        else:
            group[i] = 0
            
    # Ensure balance if necessary, though median usually gives ~10/10
    # If one group is too small, swap some vertices? 
    # For n=20, median split is 10/10 or close.
    # Let's enforce exactly 10/10 for symmetry if needed, but 9/11 is fine too.
    # The spectral method naturally finds the best cut.
    
    # Initialize adjacency matrix as Complete Bipartite Graph between group 0 and group 1
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if group[i] != group[j]:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Helper function to check if a specific triple has >= 3 common neighbors
    def count_common_neighbors(i, j, k):
        count = 0
        for l in range(n):
            if l == i or l == j or l == k:
                continue
            if adj[i, l] == 1 and adj[j, l] == 1 and adj[k, l] == 1:
                count += 1
        return count

    # Helper function to find a violating triple (i, j, k) and their common neighbors
    def find_violation():
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = []
                    for l in range(n):
                        if l != i and l != j and l != k:
                            if adj[i, l] == 1 and adj[j, l] == 1 and adj[k, l] == 1:
                                common.append(l)
                    if len(common) >= 3:
                        return (i, j, k), common
        return None, None

    # Repair Phase: Remove edges until no K3,3 exists
    # Limit iterations to prevent infinite loops, though it should converge
    for _ in range(1000):
        (i, j, k), common = find_violation()
        if common is None:
            break
        # Remove a random edge incident to i, j, or k and one of the common neighbors
        # Removing (i, common[0]) is sufficient to reduce the count for this triple
        # We pick a random edge from the K3,3 structure to remove
        # The K3,3 edges are (i, c), (j, c), (k, c) for c in common
        # Let's remove (i, random_common)
        victim_v = random.choice(common)
        adj[i, victim_v] = 0
        adj[victim_v, i] = 0
        
    # Improvement Phase: Try to add edges
    # We iterate multiple times to reach a local optimum
    improved = True
    while improved:
        improved = False
        # Shuffle non-edges to avoid bias
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 0:
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if adj[u, v] == 1:
                continue
            
            # Check if adding (u, v) creates a violation
            # Adding (u, v) makes v a neighbor of u and u a neighbor of v.
            # This affects triples containing u or v.
            # Specifically, for any pair (x, y) where v is a common neighbor of x and y (v~x and v~y),
            # v becomes a common neighbor of (u, x, y) IF u~x and u~y.
            # Wait, the condition is:
            # New common neighbors of (u, x, y) = Old common neighbors + {v} if v~u, v~x, v~y.
            # Since we are adding (u,v), v~u is true.
            # So if v~x and v~y, then v is added to CN(u, x, y).
            # We must ensure that for all x, y such that v~x and v~y (x!=u, y!=u),
            # the current count of CN(u, x, y) is < 2.
            # If current count is 2, adding v makes it 3 -> Violation.
            
            safe = True
            # Iterate over all x, y
            for x in range(n):
                if x == u or x == v:
                    continue
                if adj[v, x] == 0:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    if adj[v, y] == 0:
                        continue
                    
                    # Check current common neighbors of (u, x, y)
                    # We need to count l such that l~u, l~x, l~y
                    # Note: v is NOT currently a neighbor of u, so v is not in the current count.
                    count = 0
                    for l in range(n):
                        if l == u or l == x or l == y:
                            continue
                        if adj[u, l] == 1 and adj[x, l] == 1 and adj[y, l] == 1:
                            count += 1
                    if count >= 2:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe:
                adj[u, v] = 1
                adj[v, u] = 1
                improved = True
                
    # Convert to list of edges
    edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
    
    return edges
