import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    N = 50
    
    # 1. Generate a specific hexagonal grid layout for 50 circles
    # Row counts: 8, 7, 8, 7, 8, 7, 5 = 50
    points = []
    row_counts = [8, 7, 8, 7, 8, 7, 5]
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

    # 2. Spectral Embedding Initialization
    centers_spec = centers_grid_scaled.copy()
    try:
        dist = cdist(centers_grid, centers_grid)
        np.fill_diagonal(dist, np.inf)
        adj = np.zeros((N, N))
        tol = 1.1
        for i in range(N):
            neighbors = np.where(dist[i] < tol)[0]
            for j in neighbors:
                adj[i, j] = 1
                adj[j, i] = 1
                
        degree = adj.sum(axis=1)
        L = csr_matrix(np.diag(degree) - adj)
        
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        x_spec = vecs[:, 1]
        y_spec = vecs[:, 2]
        
        x_spec -= np.mean(x_spec)
        y_spec -= np.mean(y_spec)
        
        max_ext_spec = np.max(np.abs(np.column_stack([x_spec, y_spec])))
        if max_ext_spec > 0:
            scale_spec = 0.45 / max_ext_spec
            centers_spec = np.column_stack([x_spec * scale_spec, y_spec * scale_spec])
    except Exception:
        pass

    # 3. Optimization Function
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

    # 4. Multi-Start Optimization
    best_centers = centers_grid_scaled
    best_radius = get_radius(best_centers)
    
    np.random.seed(42)
    inits = [
        centers_grid_scaled,
        centers_spec
    ]
    
    # Add jittered versions of the grid
    for i in range(4):
        noise = np.random.normal(0, 0.01, (N, 2))
        inits.append(centers_grid_scaled + noise)
        
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
