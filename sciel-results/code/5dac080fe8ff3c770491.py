"""Spectral graph optimization for circle packing in a unit square."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
import scipy.sparse.linalg as spla

N = 50


def construct():
    """
    Construct centres of N=50 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the common radius via spectral graph optimization.
    
    Approach:
    1. Start with a good initial configuration (slightly perturbed grid).
    2. Iteratively:
       a. Build a k-NN graph with weights ~ 1/d^2.
       b. Compute the graph Laplacian.
       c. Use the Fiedler vector (second eigenvector) to guide gradient ascent
          on a proxy for minimum separation.
       d. Apply a repulsive force based on Laplacian eigenvalue.
       e. Enforce boundary constraints.
    3. Return the optimized centres.
    """
    np.random.seed(42)
    
    # Initial configuration: grid with small random perturbation
    m = int(np.ceil(np.sqrt(N)))  # 8x8 grid, use first 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small perturbation to break symmetry
    centers += np.random.randn(N, 2) * 0.005
    
    # Ensure within bounds
    centers = np.clip(centers, -0.49, 0.49)
    
    k = 8  # k-nearest neighbors
    n_iter = 200
    learning_rate = 0.01
    
    def compute_knn_graph(centers, k):
        """Compute k-NN graph with weights inversely proportional to squared distance."""
        n = len(centers)
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Find k nearest neighbors for each point
        knn_indices = np.argsort(dist_sq, axis=1)[:, :k]
        
        # Build adjacency matrix with weights = 1/dist^2
        adj = np.zeros((n, n))
        for i in range(n):
            for j in knn_indices[i]:
                if dist_sq[i, j] > 1e-12:
                    adj[i, j] = 1.0 / dist_sq[i, j]
        
        return adj
    
    def laplacian_eigenvector(adj):
        """Compute the Fiedler vector (second smallest eigenvector of Laplacian)."""
        # Compute Laplacian: L = D - A
        degrees = adj.sum(axis=1)
        D = np.diag(degrees)
        L = D - adj
        
        # Make symmetric
        L = (L + L.T) / 2.0
        
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        
        # Ensure consistent sign
        if fiedler[np.argmax(np.abs(fiedler))] < 0:
            fiedler = -fiedler
            
        return fiedler, eigvals[idx[1]]
    
    def boundary_penalty(centers):
        """Penalty for being too close to boundaries."""
        margin = 0.01
        penalty = np.zeros_like(centers)
        
        # Left boundary
        close_left = centers[:, 0] < -0.5 + margin
        penalty[close_left, 0] += (-0.5 + margin - centers[close_left, 0]) * 10.0
        
        # Right boundary
        close_right = centers[:, 0] > 0.5 - margin
        penalty[close_right, 0] += (0.5 - margin - centers[close_right, 0]) * 10.0
        
        # Bottom boundary
        close_bottom = centers[:, 1] < -0.5 + margin
        penalty[close_bottom, 1] += (-0.5 + margin - centers[close_bottom, 1]) * 10.0
        
        # Top boundary
        close_top = centers[:, 1] > 0.5 - margin
        penalty[close_top, 1] += (0.5 - margin - centers[close_top, 1]) * 10.0
        
        return penalty
    
    # Main optimization loop
    for it in range(n_iter):
        # Build k-NN graph
        adj = compute_knn_graph(centers, k)
        
        # Compute Fiedler vector and second eigenvalue
        fiedler, lambda_2 = laplacian_eigenvector(adj)
        
        # Compute repulsive forces:
        # 1. Spectral gradient: move points apart in the direction of the Fiedler vector
        #    Points with high Fiedler value move in positive direction, low in negative
        spectral_force = fiedler[:, np.newaxis] * 0.01
        
        # 2. Direct pairwise repulsion for very close pairs
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Repulsive force: sum over j of (i - j) / dist^2 for close pairs
        repulsive = np.zeros_like(centers)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d2 = dist_sq[i, j]
                if d2 < 0.01:  # Only for close pairs
                    repulsive[i] += (centers[i] - centers[j]) / d2
        
        # Scale repulsive force
        repulsive *= 0.001
        
        # 3. Boundary penalty
        bpen = boundary_penalty(centers)
        
        # Combine forces
        force = spectral_force + repulsive + bpen
        
        # Update centers
        centers += learning_rate * force
        
        # Clip to valid region
        centers = np.clip(centers, -0.49, 0.49)
        
        # Adaptive learning rate decay
        learning_rate *= 0.995
    
    # Final refinement: local optimization to maximize minimum distance
    # Use a simple gradient ascent on the minimum distance
    for refine_iter in range(50):
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find minimum distance
        min_dist = np.min(dist)
        
        # Find the pair with minimum distance
        i_min, j_min = np.unravel_index(np.argmin(dist), dist.shape)
        
        # Compute gradient of minimum distance w.r.t. centers
        grad = np.zeros_like(centers)
        dvec = centers[i_min] - centers[j_min]
        dnorm = np.linalg.norm(dvec)
        if dnorm > 1e-10:
            grad[i_min] += dvec / dnorm
            grad[j_min] -= dvec / dnorm
        
        # Add boundary constraints
        for i in range(N):
            if centers[i, 0] < -0.49:
                grad[i, 0] += 1.0
            if centers[i, 0] > 0.49:
                grad[i, 0] -= 1.0
            if centers[i, 1] < -0.49:
                grad[i, 1] += 1.0
            if centers[i, 1] > 0.49:
                grad[i, 1] -= 1.0
        
        # Update
        centers += 0.005 * grad
        centers = np.clip(centers, -0.49, 0.49)
    
    # Ensure valid output
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
