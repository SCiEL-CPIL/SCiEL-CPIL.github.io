import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance
import sklearn.cluster

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D.
    
    Strategy:
    1. Generate diverse random initializations.
    2. Use Spectral Clustering to identify potential structural clusters (k-means-like but graph-based).
    3. Use L-BFGS-B to optimize a smooth log-sum-exp surrogate of the objective function.
       The objective is to maximize (d_min / d_max)^2.
       We minimize -log(d_min) + log(d_max) which is equivalent to minimizing log(d_max/d_min).
       Using log-sum-exp makes the function differentiable and convex-ish in local regions.
    4. Refine the best candidate using a direct search (Nelder-Mead) on the true non-smooth objective.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = time.time() + 580  # 580 seconds to be safe against the 600s limit

    def get_distances(points):
        """Calculate pairwise distances for a (N, 2) array."""
        return scipy.spatial.distance.pdist(points)

    def true_objective(x):
        """
        The actual metric to maximize: R = (d_min / d_max)^2.
        We want to maximize this. 
        Returns the score.
        """
        points = x.reshape(n, d)
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def smooth_objective(x, beta=50.0):
        """
        Smooth surrogate objective to minimize.
        We want to maximize d_min / d_max.
        Equivalent to minimizing log(d_max) - log(d_min).
        
        log(d_max) approximated by logsumexp(beta * dists) / beta
        log(d_min) approximated by -logsumexp(-beta * dists) / beta
        
        Objective to minimize: 
        f(x) = (logsumexp(beta * d) - logsumexp(-beta * d)) / beta
        
        This is differentiable and smooth.
        """
        points = x.reshape(n, d)
        dists = get_distances(points)
        
        # Stable logsumexp
        def logsumexp(a, axis=None):
            m = np.max(a, axis=axis, keepdims=True)
            # Handle case where max is -inf (shouldn't happen with distances)
            m = np.where(np.isinf(m), 0, m)
            s = np.sum(np.exp(a - m), axis=axis, keepdims=True)
            return np.log(s) + np.squeeze(m, axis=axis)
        
        # Max term: log(d_max) ~ (1/beta) * logsumexp(beta * dists)
        # Min term: log(d_min) ~ -(1/beta) * logsumexp(-beta * dists)
        
        # We want to minimize: log(d_max) - log(d_min)
        # = (1/beta) * logsumexp(beta * dists) - (-(1/beta) * logsumexp(-beta * dists))
        # = (1/beta) * (logsumexp(beta * dists) + logsumexp(-beta * dists))
        
        term_max = logsumexp(beta * dists)
        term_min = logsumexp(-beta * dists)
        
        obj = (term_max + term_min) / beta
        
        # Add a small penalty if points are too close (singularities)
        # Actually, if d_min -> 0, log(d_min) -> -inf, so -log(d_min) -> +inf.
        # The formula above handles this: logsumexp(-beta * d) will be dominated by the smallest d.
        # If d_min is very small, -beta * d_min is large positive, logsumexp is large, term_min is large.
        # So the objective goes to +infinity, which is good for minimization.
        
        return obj

    def smooth_gradient(x, beta=50.0):
        """
        Numerical gradient for L-BFGS-B if analytical is too complex or unstable.
        L-BFGS-B supports numerical gradients, but providing an analytical one is faster.
        However, implementing the gradient of logsumexp(pdist) is tricky due to the structure of pdist.
        We will rely on L-BFGS-B's numerical gradient or use a finite difference wrapper.
        For speed, let's just use L-BFGS-B with default numerical gradient.
        """
        pass

    # Initial Candidates
    candidates = []
    
    # 1. Regular Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates.append(grid_pts)
    
    # 2. Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(circle_pts)
    
    # 3. Random Points (multiple seeds)
    for seed in [42, 123, 999, 2023]:
        np.random.seed(seed)
        rand_pts = np.random.randn(n, d)
        candidates.append(rand_pts)
        
    # 4. Spectral Clustering based initialization
    # Hypothesis: The optimal configuration might have some symmetry or clustering.
    # We generate random points, compute a similarity graph, and use spectral clustering
    # to find "centers" or structure, then place points near these structures.
    
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        # Generate a random cloud
        cloud = np.random.randn(n, d) * 2.0
        
        # Compute pairwise distances
        dists = get_distances(cloud)
        
        # Create a similarity matrix (inverse of distance)
        # Use a Gaussian kernel
        sigma = np.median(dists[dists > 0])
        sim = np.exp(-(dists**2) / (2 * sigma**2))
        # Fill diagonal
        sim_mat = np.zeros((n, n))
        idx = np.triu_indices(n, k=1)
        sim_mat[idx] = sim
        sim_mat[idx[1], idx[0]] = sim
        
        # Spectral Clustering (k=2, 3, 4) to see if there's a natural partition
        for k in [2, 3, 4]:
            try:
                # Simple spectral clustering implementation
                # Laplacian L = D - W
                degrees = np.sum(sim_mat, axis=1)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
                L = np.eye(n) - D_inv_sqrt @ sim_mat @ D_inv_sqrt
                
                # Eigendecomposition
                vals, vecs = np.linalg.eigh(L)
                # Take k smallest eigenvectors
                U = vecs[:, :k]
                
                # Normalize rows
                norms = np.linalg.norm(U, axis=1, keepdims=True)
                U = U / (norms + 1e-10)
                
                # K-Means on U
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
                labels = kmeans.fit_predict(U)
                
                # Calculate centroids in original space
                centroids = np.zeros((k, d))
                for i in range(k):
                    mask = labels == i
                    if np.sum(mask) > 0:
                        centroids[i] = np.mean(cloud[mask], axis=0)
                    else:
                        centroids[i] = np.random.randn(d) * 0.1
                
                # Construct a new candidate by placing points near these centroids
                # We distribute the n points among the k clusters proportional to cluster size
                new_pts = np.zeros((n, d))
                point_idx = 0
                for i in range(k):
                    count = np.sum(labels == i)
                    for j in range(count):
                        # Place near centroid with small noise
                        new_pts[point_idx] = centroids[i] + np.random.randn(d) * 0.1
                        point_idx += 1
                    # If count is 0, skip (shouldn't happen with kmeans)
                
                # Ensure we have exactly n points
                # If some clusters were empty in kmeans assignment, we might have fewer points.
                # But KMeans assigns all points.
                
                candidates.append(new_pts)
            except Exception:
                continue

    best_points = None
    best_score = -np.inf

    # Optimization Loop
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # 1. L-BFGS-B on Smooth Objective
        # We minimize the smooth surrogate.
        # The smooth objective is scale-invariant? 
        # R = (d_min/d_max)^2 is scale invariant. 
        # However, L-BFGS-B might drift in scale. 
        # We can normalize the points after each step or add a penalty for scale deviation.
        # Let's just run it and see.
        
        try:
            # Convert to flat array
            x0 = init_pts.flatten()
            
            # Optimize
            res = scipy.optimize.minimize(
                smooth_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            opt_pts = res.x.reshape(n, d)
            
            # Evaluate true score
            score = true_objective(opt_pts)
            
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
                
        except Exception as e:
            # print(f"Optimization failed for candidate {i}: {e}")
            pass

        # 2. Nelder-Mead Refinement on True Objective
        # If we have a good candidate, refine it with NM on the raw, non-smooth objective.
        # This helps to "snap" to the exact optimal configuration if we are close.
        
        if best_points is not None:
            try:
                # Start NM from the best L-BFGS-B result
                # NM is derivative-free and handles non-smooth functions better than gradient methods
                res_nm = scipy.optimize.minimize(
                    lambda x: -true_objective(x), # Maximize true_objective
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                nm_pts = res_nm.x.reshape(n, d)
                nm_score = true_objective(nm_pts)
                
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_pts.copy()
                    
            except Exception as e:
                # print(f"NM failed: {e}")
                pass

        # 3. Check the initial candidate itself (in case optimization moved it away from a good spot)
        init_score = true_objective(init_pts.flatten())
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()

    # Final Fallback
    if best_points is None:
        # Return the grid
        best_points = grid_pts

    # One last check and cleanup
    # Ensure the points are finite and shape is correct
    if not np.all(np.isfinite(best_points)):
        best_points = grid_pts
        
    return best_points

if __name__ == "__main__":
    pts = min_max_dist_dim2_16()
    print(f"Shape: {pts.shape}")
    print(f"Points:\n{pts}")
    dists = scipy.spatial.distance.pdist(pts)
    d_min = np.min(dists)
    d_max = np.max(dists)
    ratio = d_min / d_max
    score = ratio ** 2
    print(f"d_min: {d_min}, d_max: {d_max}, Ratio: {ratio}, Score: {score}")
