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
    
    # Scale to fit roughly in [-0.49, 0.49] to start closer to boundary
    max_ext = np.max(np.abs(centers_grid))
    if max_ext > 0:
        scale = 0.49 / max_ext
        centers_grid_scaled = centers_grid * scale
    else:
        centers_grid_scaled = centers_grid

    # 2. Spectral Embedding Initialization (using k-NN graph)
    centers_spec = centers_grid_scaled.copy()
    try:
        dist = cdist(centers_grid, centers_grid)
        np.fill_diagonal(dist, np.inf)
        # Build k-NN graph (k=6) to approximate hexagonal contact graph
        k = 6
        adj = np.zeros((N, N))
        for i in range(N):
            # Find k nearest neighbors
            nearest = np.argsort(dist[i])[:k]
            for j in nearest:
                adj[i, j] = 1
                adj[j, i] = 1
                
        degree = adj.sum(axis=1)
        L = csr_matrix(np.diag(degree) - adj)
        
        # Get 2 smallest non-zero eigenvalues/vectors
        # sigma=0 for shift-invert, which='LM' for largest magnitude (which are smallest for L)
        # Actually, for Laplacian, we want smallest non-zero.
        # eigsh with sigma=0 finds eigenvalues near 0.
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        # Skip the first vector (constant)
        x_spec = vecs[:, 1]
        y_spec = vecs[:, 2]
        
        x_spec -= np.mean(x_spec)
        y_spec -= np.mean(y_spec)
        
        max_ext_spec = np.max(np.abs(np.column_stack([x_spec, y_spec])))
        if max_ext_spec > 0:
            scale_spec = 0.49 / max_ext_spec
            centers_spec = np.column_stack([x_spec * scale_spec, y_spec * scale_spec])
    except Exception:
        pass

    # 3. Optimization Functions
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

    def refine_active_set(centers):
        """
        Refine the packing by maximizing the minimum of the currently active constraints.
        This helps escape local minima where the optimizer isn't pushing the bottleneck hard enough.
        """
        centers = centers.reshape(N, 2)
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        r_min = min(np.min(d) / 2.0, np.min(walls))
        
        # Identify active constraints (within 1% of min)
        tol = r_min * 0.01 + 1e-8
        active_inter = np.where(d <= 2.0 * r_min + tol)[0]
        # Need to map back to pairs
        tri_indices = np.triu_indices(N, 1)
        active_pairs = []
        for idx in active_inter:
            i, j = tri_indices[0][idx], tri_indices[1][idx]
            active_pairs.append((i, j))
            
        active_walls = np.where(walls <= r_min + tol)[0]
        # Map wall indices to (point_idx, wall_side)
        # 0..N-1 are x-walls, N..2N-1 are y-walls
        active_wall_constraints = []
        for idx in active_walls:
            if idx < N:
                active_wall_constraints.append((idx, 0)) # x-wall
            else:
                active_wall_constraints.append((idx - N, 1)) # y-wall

        if not active_pairs and not active_wall_constraints:
            return centers

        def active_objective(x):
            x = x.reshape(N, 2)
            d_act = cdist(x, x)
            np.fill_diagonal(d_act, np.inf)
            
            vals = []
            for i, j in active_pairs:
                vals.append(d_act[i, j] / 2.0)
            
            wx = 0.5 - np.abs(x[:, 0])
            wy = 0.5 - np.abs(x[:, 1])
            for pt, side in active_wall_constraints:
                if side == 0:
                    vals.append(wx[pt])
                else:
                    vals.append(wy[pt])
            
            if not vals:
                return 0.0
            vals = np.array(vals)
            vals = np.clip(vals, 1e-6, None)
            # Maximize min -> Minimize -min
            # Use log-sum-exp for smoothness
            u = -lam * vals
            max_u = np.max(u)
            log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
            return log_sum / lam

        try:
            res = minimize(active_objective, centers.flatten(), method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 200, 'ftol': 1e-10})
            new_centers = res.x.reshape(N, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            return new_centers
        except Exception:
            return centers

    # 4. Multi-Start Optimization
    best_centers = centers_grid_scaled
    best_radius = get_radius(best_centers)
    
    np.random.seed(42)
    inits = [
        centers_grid_scaled,
        centers_spec
    ]
    
    # Add jittered versions of the grid
    for i in range(8):
        noise = np.random.normal(0, 0.01, (N, 2))
        inits.append(centers_grid_scaled + noise)
        
    # Add random starts
    for i in range(4):
        inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))

    bounds = [(-0.5, 0.5)] * (2 * N)

    for x0 in inits:
        x0_flat = x0.flatten()
        x0_flat = np.clip(x0_flat, -0.5, 0.5)
        
        try:
            res = minimize(objective, x0_flat, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            centers_opt = res.x.reshape(N, 2)
            centers_opt = np.clip(centers_opt, -0.5, 0.5)
            
            # Refine using active set
            centers_refined = refine_active_set(centers_opt)
            
            r = get_radius(centers_refined)
            if r > best_radius:
                best_radius = r
                best_centers = centers_refined
        except Exception:
            continue

    return best_centers
