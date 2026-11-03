import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    N = 50
    
    # 1. Define a specific near-optimal contact graph for 50 circles
    # Hexagonal packing structure: Rows of 6, 7, 8, 8, 8, 7, 6
    row_counts = [6, 7, 8, 8, 8, 7, 6]
    points = []
    for i, count in enumerate(row_counts):
        # Offset rows to create hexagonal structure
        offset = 0.5 if i % 2 == 1 else 0.0
        for j in range(count):
            x = j + offset
            y = i * np.sqrt(3) / 2
            points.append([x, y])
            
    centers_grid = np.array(points)
    # Center the grid
    centers_grid -= np.mean(centers_grid, axis=0)
    
    # Scale to fit in [-0.45, 0.45] roughly
    max_ext = np.max(np.abs(centers_grid))
    if max_ext > 0:
        scale = 0.45 / max_ext
        centers_grid = centers_grid * scale

    # 2. Construct Contact Graph (Adjacency Matrix)
    # Connect circles that are close in the grid layout
    adj = np.zeros((N, N))
    dist_grid = cdist(centers_grid, centers_grid)
    # Threshold for connectivity in the grid layout
    # The minimum distance in the grid is 1.0 (scaled). 
    # We want to connect immediate neighbors.
    # Let's find the k-nearest neighbors or use a distance threshold.
    # In a hexagonal grid, each interior point has 6 neighbors.
    # Let's use a threshold slightly larger than the unit distance in the grid.
    # The unit distance in the grid before scaling was 1.0.
    # After scaling, it is 1.0 * scale.
    # We connect if dist < 1.1 * (1.0 * scale)
    threshold = 1.1 * (1.0 * scale)
    
    for i in range(N):
        neighbors = np.where(dist_grid[i] < threshold)[0]
        for j in neighbors:
            if i != j:
                adj[i, j] = 1
    
    # 3. Spectral Embedding
    # Compute Laplacian L = D - A
    degree = adj.sum(axis=1)
    L = csr_matrix(np.diag(degree) - adj)
    
    try:
        # We want the eigenvectors corresponding to the smallest non-zero eigenvalues
        # eigsh with sigma=0 finds eigenvalues near 0
        # We need k=3 to get the 2D embedding (index 0 is constant, 1 and 2 are the axes)
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        # Use the 2nd and 3rd eigenvectors (indices 1 and 2) for x and y
        x_spec = vecs[:, 1]
        y_spec = vecs[:, 2]
        
        # Center and scale
        x_spec -= np.mean(x_spec)
        y_spec -= np.mean(y_spec)
        
        # Scale to fit in the square
        max_ext_spec = np.max(np.abs(np.column_stack([x_spec, y_spec])))
        if max_ext_spec > 0:
            scale_spec = 0.45 / max_ext_spec
            centers_spec = np.column_stack([x_spec * scale_spec, y_spec * scale_spec])
        else:
            centers_spec = centers_grid
    except Exception:
        centers_spec = centers_grid

    # 4. Optimization
    def objective(x, temp):
        x = x.reshape(N, 2)
        # Walls
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        # Pairwise distances
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        tri_indices = np.triu_indices(N, 1)
        d_inter = d[tri_indices]
        
        # Effective distances (radius = d/2 for pairs, d for walls)
        # We want to maximize r = min(min(d_inter)/2, min(walls))
        # Let margins = [d_inter/2, walls]
        margins = np.concatenate([d_inter / 2.0, walls])
        
        # Clip to avoid log(0) or negative values in exp
        margins = np.clip(margins, 1e-6, None)
        
        # Log-sum-exp approximation of min
        # min(x) = -lim_{t->inf} (1/t) log(sum(exp(-t*x)))
        # Here we want to minimize -min(x), which is equivalent to maximizing min(x).
        # So we minimize logsumexp(-margins/t) / t ? 
        # Actually, standard trick: minimize (1/temp) * logsumexp(-margins/temp)
        # As temp -> 0, this approaches min(margins).
        # We want to MAXIMIZE min(margins), so we MINIMIZE -min(margins).
        # Let's minimize: (1/temp) * logsumexp(-margins/temp)
        # Wait, if margins are large, -margins/temp is large negative, exp is small.
        # If we minimize the logsumexp, we are pushing the smallest margins up?
        # Let's check: f(m) = (1/t) log(sum exp(-m_i/t)).
        # If m_min increases, all m_i increase (assuming others fixed), -m_i/t decrease, exp decrease, sum decrease, log decrease.
        # So minimizing f(m) maximizes m_min. Correct.
        
        u = -margins / temp
        max_u = np.max(u)
        log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
        return log_sum / temp

    def get_radius(centers):
        centers = centers.reshape(N, 2)
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        min_wall = np.min(np.vstack([wx, wy]).flatten())
        return min(min_dist / 2.0, min_wall)

    # 5. Multi-Start with Annealing
    best_centers = centers_spec
    best_radius = get_radius(best_centers)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Start with the spectral embedding
    x0 = centers_spec.flatten()
    x0 = np.clip(x0, -0.5, 0.5)
    
    # Annealing schedule for temperature
    # Start high to explore, end low to refine
    temps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    current_x = x0
    for t in temps:
        try:
            res = minimize(lambda x: objective(x, t), current_x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            current_x = res.x
            current_x = np.clip(current_x, -0.5, 0.5)
            
            r = get_radius(current_x)
            if r > best_radius:
                best_radius = r
                best_centers = current_x.reshape(N, 2)
        except Exception:
            break

    # Also try the grid initialization just in case
    x0_grid = centers_grid.flatten()
    x0_grid = np.clip(x0_grid, -0.5, 0.5)
    current_x = x0_grid
    for t in temps:
        try:
            res = minimize(lambda x: objective(x, t), current_x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            current_x = res.x
            current_x = np.clip(current_x, -0.5, 0.5)
            
            r = get_radius(current_x)
            if r > best_radius:
                best_radius = r
                best_centers = current_x.reshape(N, 2)
        except Exception:
            break

    return best_centers
