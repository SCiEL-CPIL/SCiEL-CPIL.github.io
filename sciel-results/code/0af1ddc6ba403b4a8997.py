"""Packing 300 equal circles in a unit circle using Delaunay-based spectral optimization."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh
import time

N = 300

# EVOLVE-BLOCK-START

def _score(points):
    """Compute the packing score: min of pairwise distances/2 and container margins."""
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Container margins
    norms = np.sqrt(np.sum(points ** 2, axis=1))
    min_margin = np.min(1.0 - norms)
    return min(min_pair, min_margin)


def _build_delaunay(points):
    """Build Delaunay triangulation and return adjacency matrix."""
    try:
        tri = Delaunay(points)
        n = len(points)
        # Build adjacency from Delaunay
        adj = np.zeros((n, n))
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[simplex[i], simplex[j]] = 1.0
                    adj[simplex[j], simplex[i]] = 1.0
        return adj
    except Exception:
        # Fallback: connect nearest neighbors
        n = len(points)
        adj = np.zeros((n, n))
        for i in range(n):
            diffs = points - points[i]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            dists[i] = np.inf
            nn = np.argsort(dists)[:4]
            for j in nn:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj


def _spectral_connectivity(adj):
    """Compute algebraic connectivity (Fiedler value) of the graph."""
    n = adj.shape[0]
    if n < 3:
        return 0.0
    L = sparse_laplacian(adj)
    try:
        # Find smallest non-zero eigenvalue
        eigenvalues = eigsh(L, k=2, sigma=0, which='LM', return_eigenvectors=False)
        # eigsh with sigma=0 finds eigenvalues near 0
        # The algebraic connectivity is the second smallest eigenvalue
        eigvals = sorted(eigenvalues)
        # First should be ~0 (trivial), second is Fiedler
        if len(eigvals) >= 2:
            return max(0.0, eigvals[1])
        return 0.0
    except Exception:
        return 0.0


def _gradient_descent(points, adj, max_iter=500, lr=0.001, time_limit=80.0):
    """Optimize points to maximize spectral connectivity while maintaining validity."""
    start_time = time.time()
    n = len(points)
    points = points.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current score
        current_score = _score(points)
        
        # Compute gradient of spectral connectivity w.r.t. positions
        # Use finite differences for simplicity
        eps = 1e-5
        grad = np.zeros_like(points)
        
        # Sample a subset of points for gradient computation to save time
        if n > 100:
            idx = np.random.choice(n, min(50, n), replace=False)
        else:
            idx = np.arange(n)
        
        for i in idx:
            for d in range(2):
                orig = points[i, d]
                points[i, d] = orig + eps
                # Rebuild Delaunay and compute connectivity
                adj_plus = _build_delaunay(points)
                f_plus = _spectral_connectivity(adj_plus)
                points[i, d] = orig - eps
                adj_minus = _build_delaunay(points)
                f_minus = _spectral_connectivity(adj_minus)
                points[i, d] = orig
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Update points
        points += lr * grad
        
        # Project back inside unit circle
        norms = np.sqrt(np.sum(points ** 2, axis=1))
        mask = norms > 0.99
        if np.any(mask):
            points[mask] *= 0.99 / norms[mask, np.newaxis]
    
    return points


def construct():
    """Construct 300 circle centers in unit disk using spectral optimization."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Initial hexagonal lattice
    spacing = 0.054
    points_list = []
    k = int(1.0 / spacing) + 3
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points_list.append([x, y])
    
    points = np.array(points_list)
    
    # Select N points closest to center
    if len(points) > N:
        norms = np.sqrt(np.sum(points ** 2, axis=1))
        idx = np.argsort(norms, kind="stable")[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points if needed
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.95)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    # Add small random perturbation to break symmetry
    points += np.random.randn(*points.shape) * 0.001
    
    # Project inside unit circle
    norms = np.sqrt(np.sum(points ** 2, axis=1))
    mask = norms > 0.99
    if np.any(mask):
        points[mask] *= 0.99 / norms[mask, np.newaxis]
    
    # Initial score
    best_score = _score(points)
    best_points = points.copy()
    
    # Build initial Delaunay
    adj = _build_delaunay(points)
    
    # Spectral optimization
    if time.time() < deadline - 30:
        points = _gradient_descent(points, adj, max_iter=200, lr=0.0005, time_limit=60.0)
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Local refinement: push apart close pairs
    if time.time() < deadline - 20:
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            # Find closest pairs
            diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Get the 20 closest pairs
            flat_idx = np.argsort(dists.flatten())[:40]
            i_idx = flat_idx // N
            j_idx = flat_idx % N
            
            improved = False
            for i, j in zip(i_idx, j_idx):
                if i >= j:
                    continue
                d = dists[i, j]
                if d < 0.055:
                    # Push apart
                    dir_vec = best_points[i] - best_points[j]
                    dir_norm = np.linalg.norm(dir_vec)
                    if dir_norm > 1e-12:
                        dir_vec /= dir_norm
                        shift = (0.055 - d) / 2.0 + 0.001
                        best_points[i] += shift * dir_vec
                        best_points[j] -= shift * dir_vec
                        improved = True
            
            # Project inside unit circle
            norms = np.sqrt(np.sum(best_points ** 2, axis=1))
            mask = norms > 0.99
            if np.any(mask):
                best_points[mask] *= 0.99 / norms[mask, np.newaxis]
            
            score = _score(best_points)
            if score > best_score:
                best_score = score
            elif not improved:
                break
    
    # Final projection and validation
    norms = np.sqrt(np.sum(best_points ** 2, axis=1))
    mask = norms > 0.999
    if np.any(mask):
        best_points[mask] *= 0.999 / norms[mask, np.newaxis]
    
    return best_points

# EVOLVE-BLOCK-END
