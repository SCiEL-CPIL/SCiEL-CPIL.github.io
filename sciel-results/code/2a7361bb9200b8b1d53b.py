import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    N = 50
    
    # 1. Define the topology of a hexagonal packing for 50 circles
    # Row counts: 8, 7, 8, 7, 8, 7, 5
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    
    # Build the adjacency matrix for the contact graph
    adj = np.zeros((N, N))
    idx = 0
    for i, count in enumerate(row_counts):
        start = idx
        end = idx + count
        # Horizontal connections
        for j in range(start, end - 1):
            adj[j, j+1] = 1
            adj[j+1, j] = 1
        
        # Vertical connections to next row
        if i < len(row_counts) - 1:
            next_count = row_counts[i+1]
            next_start = end
            next_end = next_start + next_count
            
            # Determine offset logic
            # If current row is 8 and next is 7: Next is shifted by 0.5
            # If current row is 7 and next is 8: Next is shifted by -0.5 (or 0.5 relative to other side)
            # Standard hex packing:
            # Row 0 (8): 0, 1, 2, 3, 4, 5, 6, 7
            # Row 1 (7): 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5
            # Connections: Row1[j] connects to Row0[j] and Row0[j+1]
            
            # Row 2 (8): 0, 1, ... 7 (aligned with Row 0)
            # Connections: Row2[j] connects to Row1[j-1] and Row1[j] (if valid)
            
            # General rule for hex packing with alternating counts:
            # If row i has count C and row i+1 has count C-1:
            #   Row i+1 index k connects to Row i index k and k+1.
            # If row i has count C and row i+1 has count C+1:
            #   Row i+1 index k connects to Row i index floor(k/2) ... wait, let's just use the geometric positions to define connections.
            
            # Let's just use the geometric positions to define the graph robustly.
            # We will generate the positions first, then find neighbors based on distance.
            pass
            
        idx = end
        
    # Let's revert to generating positions for the graph, as it's less error-prone than manual index mapping for varying row sizes.
    # We generate the "ideal" hex positions, then build the graph from those, then use spectral embedding on that graph.
    # This satisfies "fixing a known near-optimal contact graph topology" by deriving it from the known near-optimal geometry.
    
    points = []
    y_pos = 0.0
    for i, count in enumerate(row_counts):
        offset = 0.5 if i % 2 == 1 else 0.0
        for j in range(count):
            x = j + offset
            y = y_pos
            points.append([x, y])
        y_pos += np.sqrt(3) / 2
        
    centers_grid = np.array(points)
    centers_grid -= np.mean(centers_grid, axis=0)
    
    # Scale to fit roughly in [-0.45, 0.45]
    max_ext = np.max(np.abs(centers_grid))
    if max_ext > 0:
        scale = 0.45 / max_ext
        centers_grid_scaled = centers_grid * scale
    else:
        centers_grid_scaled = centers_grid

    # 2. Build Graph from this ideal layout
    dist = cdist(centers_grid, centers_grid)
    np.fill_diagonal(dist, np.inf)
    
    # Find nearest neighbors (should be 6 for interior, less for boundary)
    # In a perfect hex grid, nearest neighbors are at distance 1.
    # We use a tolerance to pick the correct neighbors.
    tol = 1.1
    adj = np.zeros((N, N))
    for i in range(N):
        neighbors = np.where(dist[i] < tol)[0]
        for j in neighbors:
            adj[i, j] = 1
            adj[j, i] = 1
            
    # 3. Spectral Embedding
    # Compute Laplacian
    degree = adj.sum(axis=1)
    L = csr_matrix(np.diag(degree) - adj)
    
    # Get Fiedler vectors (eigenvectors for smallest non-zero eigenvalues)
    # eigsh with sigma=0 finds eigenvalues closest to 0.
    # We want the first 2 non-trivial eigenvectors (indices 1 and 2, since 0 is trivial constant vector).
    try:
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        # The first column (idx 0) corresponds to the constant vector (eigenvalue 0).
        # The next two columns correspond to the x and y coordinates in the spectral embedding.
        x_spec = vecs[:, 1]
        y_spec = vecs[:, 2]
        
        # Center and scale
        x_spec -= np.mean(x_spec)
        y_spec -= np.mean(y_spec)
        
        max_ext_spec = np.max(np.abs(np.column_stack([x_spec, y_spec])))
        if max_ext_spec > 0:
            scale_spec = 0.45 / max_ext_spec
            centers_spec = np.column_stack([x_spec * scale_spec, y_spec * scale_spec])
        else:
            centers_spec = centers_grid_scaled.copy()
            
    except Exception:
        centers_spec = centers_grid_scaled.copy()

    # 4. Optimization Function
    lam = 100.0

    def objective(x):
        x = x.reshape(N, 2)
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        tri_indices = np.triu_indices(N, 1)
        d_inter = d[tri_indices]
        
        all_d = np.concatenate([d_inter, walls])
        all_d = np.clip(all_d, 1e-6, None)
        
        u = -lam * all_d
        max_u = np.max(u)
        log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
        return log_sum / lam

    def get_radius(centers):
        centers = centers.reshape(N, 2)
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        min_wall = np.min(np.vstack([wx, wy]).flatten())
        return min(min_dist / 2.0, min_wall)

    # 5. Multi-Start Optimization
    # Prioritize the spectral solution as it is theoretically grounded in the topology
    best_centers = centers_spec
    best_radius = get_radius(best_centers)
    
    np.random.seed(42)
    
    # Start with the spectral solution and slight perturbations
    inits = [
        centers_spec,
        centers_grid_scaled
    ]
    
    # Add jittered spectral starts
    for i in range(4):
        noise = np.random.normal(0, 0.01, (N, 2))
        inits.append(centers_spec + noise)
        
    # Add a random start
    inits.append(np.random.uniform(-0.4, 0.4, (N, 2)))

    bounds = [(-0.5, 0.5)] * (2 * N)

    for x0 in inits:
        x0_flat = x0.flatten()
        x0_flat = np.clip(x0_flat, -0.5, 0.5)
        
        try:
            res = minimize(objective, x0_flat, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            centers_opt = res.x.reshape(N, 2)
            centers_opt = np.clip(centers_opt, -0.5, 0.5)
            
            r = get_radius(centers_opt)
            if r > best_radius:
                best_radius = r
                best_centers = centers_opt
        except Exception:
            continue

    return best_centers
