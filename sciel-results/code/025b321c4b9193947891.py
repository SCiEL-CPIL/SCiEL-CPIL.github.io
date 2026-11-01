def _energy_and_grad(X):
    X = np.asarray(X, dtype=float)
    # Compute pairwise distances
    # d2[i,j] = ||x_i - x_j||^2
    d2 = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * (X @ X.T)
    np.fill_diagonal(d2, 1.0) # Avoid division by zero, value doesn't matter as W[i,i]=0
    d2 = np.maximum(d2, _FLOOR**2) # Floor for stability
    d = np.sqrt(d2)
    
    # Energy
    # Use np.sum for speed in optimization loop. Evaluator uses fsum, but we just need to minimize.
    E = -np.sum(np.log(d[np.triu_indices(N, k=1)]))
    
    # Gradient
    # W[i,j] = 1/d2[i,j] for i != j, 0 for i=j
    W = np.zeros_like(d2)
    mask = ~np.eye(N, dtype=bool)
    W[mask] = 1.0 / d2[mask]
    
    # S[i] = sum_j W[i,j]
    S = W.sum(axis=1)
    
    # G = (W - diag(S)) @ X
    # G_i = sum_j W[i,j] x_j - S[i] x_i
    G = W @ X - S[:, None] * X
    
    return float(E), G
