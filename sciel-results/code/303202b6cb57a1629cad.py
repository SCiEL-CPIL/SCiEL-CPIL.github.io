import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    N = 50
    
    # 1. Initial Hexagonal Grid
    # Approximate hexagonal packing for 50 circles
    # 7 rows: 8, 7, 8, 7, 8, 7, 8 = 53 -> too many
    # 7 rows: 8, 7, 8, 7, 8, 7, 5 = 50
    # Let's just generate a dense grid and take the first 50
    centers = []
    r = 0.07 # approx radius
    dx = 2 * r
    dy = r * np.sqrt(3)
    for i in range(10):
        for j in range(10):
            if len(centers) >= N:
                break
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            centers.append([x, y])
    centers = np.array(centers[:N])
    
    # Center and scale to fit in [-0.5, 0.5]
    centers -= np.mean(centers, axis=0)
    max_extent = np.max(np.abs(centers))
    if max_extent > 0.5:
        centers *= (0.5 / max_extent)

    # 2. Spectral Embedding Initialization (as per directive)
    # Build adjacency based on proximity in current config
    dist = cdist(centers, centers)
    np.fill_diagonal(dist, np.inf)
    # Connect 6 nearest neighbors
    adj = np.zeros((N, N))
    for i in range(N):
        idx = np.argsort(dist[i])[:6]
        for j in idx:
            adj[i, j] = 1
            adj[j, i] = 1
            
    # Laplacian
    degree = adj.sum(axis=1)
    L = csr_matrix(np.diag(degree) - adj)
    
    # Find 2 Fiedler vectors
    # eigsh finds smallest eigenvalues
    try:
        # sigma=0 to find smallest
        # k=2 for 2D embedding
        # Note: eigsh might fail if graph is disconnected, but hex grid is connected
        vals, vecs = eigsh(L, k=2, sigma=0, which='LM') 
        # The first vector is usually the constant vector (eigenvalue 0). 
        # We need the next two.
        # eigsh with sigma=0 finds eigenvalues closest to 0.
        # So vecs[:, 0] is eigenvector for lambda ~ 0 (constants).
        # vecs[:, 1] and vecs[:, 2] are the Fiedler vectors.
        
        # However, eigsh returns k vectors. If k=2, it returns the 2 smallest.
        # One is the trivial one. We need the next two?
        # Actually, for 2D embedding, we need the eigenvectors corresponding to the two smallest *non-zero* eigenvalues.
        # So we should ask for k=3, and take vecs[:, 1] and vecs[:, 2].
        
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        # vecs[:, 0] is the constant vector.
        # vecs[:, 1] and vecs[:, 2] are the Fiedler vectors.
        x_spec = vecs[:, 1]
        y_spec = vecs[:, 2]
        
        # Normalize
        x_spec = (x_spec - np.mean(x_spec)) / np.std(x_spec)
        y_spec = (y_spec - np.mean(y_spec)) / np.std(y_spec)
        
        # Scale to fit square
        scale = 0.4 / np.max(np.abs(np.vstack([x_spec, y_spec]).T))
        centers_spec = np.column_stack([x_spec * scale, y_spec * scale])
        
        # Use spectral embedding as initial guess
        centers = centers_spec
    except Exception:
        # Fallback to grid if spectral fails
        pass

    # 3. Optimization
    # Objective: Minimize -LogSumExp(-lambda * distances)
    # This approximates -min(distances)
    lam = 50.0
    
    def objective(x):
        x = x.reshape(N, 2)
        # Boundary constraints
        # Distance to walls: 0.5 - abs(x)
        # We want to maximize min(dist_inter, dist_wall)
        # Equivalent to minimizing -min(...)
        
        # Inter-point distances
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        # Combine all distances
        # We want to avoid very small distances.
        # LogSumExp trick: -1/lam * log(sum(exp(-lam * d)))
        # To avoid overflow, we subtract the max d? No, d is small, exp(-lam*d) is large.
        # Actually, we want to maximize the minimum.
        # Let g(d) = -1/lam * log(sum(exp(-lam * d))). 
        # If one d is very small, exp(-lam*d) is large, sum is large, log is large, -log is small (negative).
        # So minimizing g(d) pushes the smallest d up.
        
        # Flatten inter-point distances (upper triangle)
        tri_indices = np.triu_indices(N, 1)
        d_inter = d[tri_indices]
        
        all_d = np.concatenate([d_inter, walls])
        
        # Clip to avoid numerical issues
        all_d = np.clip(all_d, 1e-6, None)
        
        # LogSumExp
        # max_d = np.max(all_d)
        # log_sum = np.log(np.sum(np.exp(-lam * (all_d - max_d)))) + lam * max_d
        # This is for sum(exp(lam*d)). We have sum(exp(-lam*d)).
        # Let u = -lam * all_d.
        # max_u = np.max(u)
        # log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
        
        u = -lam * all_d
        max_u = np.max(u)
        log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
        
        return log_sum / lam # We want to MINIMIZE this. 
                             # Wait. 
                             # If d is small, u is large (less negative), max_u is large, log_sum is large.
                             # So minimizing log_sum/lam minimizes the "largest" exp term, i.e., the smallest distance.
                             # Correct.

    # Flatten centers
    x0 = centers.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    # Actually, to allow some flexibility, we can set bounds to [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimize
    # L-BFGS-B is good for this size
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
    
    centers_opt = res.x.reshape(N, 2)
    
    # Final safety clip
    centers_opt = np.clip(centers_opt, -0.5, 0.5)
    
    return centers_opt
