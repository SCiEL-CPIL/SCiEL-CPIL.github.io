# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _generate_pool(n_pool=200):
    """
    Generate a diverse pool of candidate points from symmetric geometric structures.
    These structures naturally have high min/max distance ratios.
    """
    pool = []
    
    # 1. Icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    pool.extend(ico.tolist())
    
    # 2. Dodecahedron vertices (20 points)
    # Using standard coordinates
    a = 1/phi
    b = phi
    c = 1
    d = -1
    dec = np.array([
        [a, b, c], [a, b, -c], [a, -b, c], [a, -b, -c],
        [-a, b, c], [-a, b, -c], [-a, -b, c], [-a, -b, -c],
        [b, c, a], [b, -c, a], [b, c, -a], [b, -c, -a],
        [c, a, b], [c, -a, b], [c, a, -b], [c, -a, -b],
        [d, b, c], [d, b, -c], [d, -b, c], [d, -b, -c]
    ])
    # Correct Dodecahedron vertices
    dec = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [-phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, -1/phi]
    ])
    dec = dec / np.linalg.norm(dec, axis=1, keepdims=True)
    pool.extend(dec.tolist())
    
    # 3. Icosidodecahedron vertices (30 points) - Union of Icosahedron and Dodecahedron scaled appropriately
    # Actually, the icosidodecahedron vertices are the midpoints of the edges of the dodecahedron/icosahedron.
    # Let's just add more random points on the sphere and in the volume to allow for non-spherical optimal configurations.
    
    # 4. Random points on the sphere
    for _ in range(100):
        p = np.random.randn(3)
        p /= np.linalg.norm(p)
        pool.append(p)
        
    # 5. Random points in the cube [-1, 1]^3
    for _ in range(50):
        p = np.random.uniform(-1, 1, 3)
        pool.append(p)
        
    # 6. Scaled versions of the icosahedron to allow for non-unit radius
    for scale in [0.5, 0.8, 1.2, 1.5]:
        pool.extend((ico * scale).tolist())
        
    pool = np.array(pool)
    return pool

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Uses a combinatorial search strategy:
    1. Generate a pool of candidate points from symmetric polytopes and random distributions.
    2. Use Simulated Annealing to select the best subset of 14 points from the pool.
    This avoids the local minima traps of continuous coordinate optimization by searching
    over a discrete set of high-quality candidate positions.
    """
    n = 14
    d = 3
    deadline = time.time() + 590  # Safety margin
    
    # Generate candidate pool
    pool = _generate_pool(n_pool=300)
    N = len(pool)
    
    # Precompute all pairwise distances in the pool for fast evaluation
    # dist_matrix[i, j] is distance between pool[i] and pool[j]
    # Using pdist on the pool is O(N^2), which is 300^2 = 90,000, very fast.
    pool_dists = pdist(pool, metric='euclidean')
    # Reconstruct square matrix for easy indexing
    # pdist returns condensed form. We can use squareform or just compute on the fly if N is small.
    # N=300 is small enough to keep in memory as a square matrix.
    from scipy.spatial.distance import squareform
    dist_mat = squareform(pool_dists)
    
    def _score_indices(indices):
        """Score a subset of indices."""
        if len(indices) < 2:
            return 0.0
        # Extract submatrix
        sub_dist = dist_mat[np.ix_(indices, indices)]
        # Get upper triangle (excluding diagonal)
        iu = np.triu_indices_from(sub_dist, k=1)
        if len(iu[0]) == 0:
            return 0.0
        vals = sub_dist[iu]
        d_min = np.min(vals)
        d_max = np.max(vals)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Initial selection: Random 14 points
    current_indices = np.random.choice(N, n, replace=False)
    current_score = _score_indices(current_indices)
    
    best_indices = current_indices.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 1e-4
    alpha = 0.995
    max_iter = 20000
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Propose a move: replace one point in the subset with a new one from the pool
        idx_to_replace = rng.integers(n)
        new_point_idx = rng.integers(N)
        
        # Avoid picking the same point
        if new_point_idx in current_indices:
            continue
            
        old_point_idx = current_indices[idx_to_replace]
        
        # Create new indices
        new_indices = current_indices.copy()
        new_indices[idx_to_replace] = new_point_idx
        
        # Evaluate new score
        new_score = _score_indices(new_indices)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_indices = new_indices
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_indices = current_indices.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Extract the best points
    best_points = pool[best_indices]
    
    # Optional: Fine-tune the selected points with a few steps of gradient-free local optimization
    # to escape discrete grid limitations.
    # We can use a simple Nelder-Mead on the final 14 points if time permits.
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        def _minimize_neg_score(x):
            points = x.reshape(n, d)
            return -_score(points)
        
        x0 = best_points.flatten()
        try:
            result = minimize(_minimize_neg_score, x0, method='Nelder-Mead', 
                              options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < -best_score:
                opt_points = result.x.reshape(n, d)
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
        except Exception:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
