import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Generate weight-4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate weight-1 vectors (value 2)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    # Conflict matrix
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Spectral Partitioning Initialization
    # Laplacian L = D - A
    degrees = conflicts.sum(axis=1)
    L = np.diag(degrees) - conflicts.astype(float)
    
    # Compute eigenvectors
    # We want the eigenvectors corresponding to the smallest eigenvalues
    # numpy.linalg.eigh is for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        
        # Use the first few eigenvectors to create a feature map
        # Project into 2D or 3D and cluster?
        # Simpler: Use the Fiedler vector (2nd eigenvector) to split
        # Then pick the larger independent set?
        # Or use a greedy approach on the sorted vertices?
        
        # Strategy: Rank vertices by the value in the Fiedler vector
        # Vertices with similar values are likely in the same cluster
        # If we pick vertices with very different values, they are less likely to conflict?
        # Actually, in a graph, edges connect similar nodes?
        # In a conflict graph, edges are conflicts.
        # We want to pick non-conflicting nodes.
        # Spectral clustering groups nodes that are densely connected.
        # If we pick one node from each cluster, we avoid conflicts within the cluster.
        
        # Let's try a different approach: 
        # Greedy MIS with random restarts is often good enough if the graph is not too pathological.
        # But let's try to improve the Local Search.
        
    except Exception:
        pass

    best_selected = np.zeros(N4, dtype=bool)
    best_count = 0
    rng = np.random.default_rng(42)
    
    max_restarts = 300
    
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random permutation greedy
        perm = rng.permutation(N4)
        selected = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        
        # Greedy
        for u in perm:
            if not alive[u]:
                continue
            selected[u] = True
            # Remove u and neighbors
            neighbors = np.where(conflicts[u])[0]
            to_remove = np.unique(np.concatenate(([u], neighbors)))
            alive[to_remove] = False
            
        # Local Search: 1-Add
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            unselected = np.where(~selected)[0]
            if len(unselected) == 0: break
            
            # Check if any unselected node has no neighbors in selected
            # neighbor_count = sum(conflicts[unselected] & selected, axis=1)
            # This is O(N^2)
            # Optimization: maintain a count of conflicts for each node
            # But for simplicity, do it in bulk
            
            # Only check if time permits
            if len(unselected) < 1000: # Heuristic
                neighbor_counts = (conflicts[unselected] & selected).sum(axis=1)
                addable = unselected[neighbor_counts == 0]
                if len(addable) > 0:
                    # Add one
                    u = addable[0]
                    selected[u] = True
                    improved = True
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
            
    result_v4 = vectors_v4[best_selected]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
