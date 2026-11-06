import numpy as np

def construct():
    n = 20
    
    def repair(adj, max_iters=500):
        """Remove edges to eliminate K3,3 violations using vectorized ops"""
        for _ in range(max_iters):
            found = False
            for i in range(n):
                if found:
                    break
                for j in range(i+1, n):
                    if found:
                        break
                    for k in range(j+1, n):
                        # Vectorized common neighbor count
                        mask = (adj[i] & adj[j] & adj[k]).astype(int)
                        mask[i] = 0
                        mask[j] = 0
                        mask[k] = 0
                        common_count = np.sum(mask)
                        if common_count >= 3:
                            # Remove edge (i, first common neighbor)
                            common_indices = np.where(mask > 0)[0]
                            l = common_indices[0]
                            adj[i, l] = 0
                            adj[l, i] = 0
                            found = True
            if not found:
                break
        return adj
    
    best_edges = None
    best_count = 0
    
    for trial in range(30):
        np.random.seed(trial)
        
        # Random symmetric matrix
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        # Top eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        top_eigenvector = eigenvectors[:, -1]
        
        # Try different thresholds
        for q in [0.35, 0.45, 0.55, 0.65]:
            threshold = np.quantile(top_eigenvector, q)
            adj = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i+1, n):
                    if top_eigenvector[i] > threshold and top_eigenvector[j] > threshold:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            # Repair
            adj = repair(adj)
            
            edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
    
    # Fallback
    if best_edges is None or best_count <= 36:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges
