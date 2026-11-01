"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
from scipy.spatial.distance import pdist

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    # Compute all pairwise distances
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # We optimize the Gram matrix G directly.
    # G is n x n symmetric PSD.
    # Rank constraint: rank(G) <= d = 3.
    # We enforce this via eigenvalue penalty.
    
    # Variables: upper triangle of G (n*(n+1)/2 elements)
    # G[i,j] = x[k] for appropriate k
    
    def _gram_from_vec(x):
        """Reconstruct symmetric Gram matrix from upper-triangle vector."""
        G = np.zeros((n, n))
        k = 0
        for i in range(n):
            for j in range(i, n):
                G[i, j] = x[k]
                G[j, i] = x[k]
                k += 1
        return G
    
    def _vec_from_gram(G):
        """Extract upper-triangle vector from symmetric Gram matrix."""
        x = np.zeros(n * (n + 1) // 2)
        k = 0
        for i in range(n):
            for j in range(i, n):
                x[k] = G[i, j]
                k += 1
        return x
    
    def _eigenvalue_penalty(G, rank=d):
        """Penalty for eigenvalues beyond the 'rank' largest ones."""
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1]  # descending
        # Penalty: sum of squares of eigenvalues beyond rank
        if len(eigvals) > rank:
            excess = eigvals[rank:]
            return np.sum(excess ** 2)
        return 0.0
    
    def _eigenvalue_penalty_grad(G, rank=d):
        """Gradient of eigenvalue penalty w.r.t. G."""
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Gradient: for each eigenvalue beyond rank, add 2*lambda * v v^T
        grad = np.zeros((n, n))
        if len(eigvals) > rank:
            for k in range(rank, len(eigvals)):
                if eigvals[k] > 1e-10:
                    grad += 2 * eigvals[k] * np.outer(eigvecs[:, k], eigvecs[:, k])
        return grad
    
    def _squared_distances_from_gram(G):
        """Compute squared pairwise distances from Gram matrix."""
        # ||x_i - x_j||^2 = G[i,i] + G[j,j] - 2*G[i,j]
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = G[i, i] + G[j, j] - 2 * G[i, j]
        return d_sq
    
    def objective_and_grad(x, penalty_weight=10.0):
        """Objective: ratio of max to min squared distance + eigenvalue penalty."""
        G = _gram_from_vec(x)
        
        # Ensure diagonal is non-negative
        G = np.maximum(G, 0)
        # Re-symmetrize
        G = (G + G.T) / 2
        
        d_sq = _squared_distances_from_gram(G)
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-10:
            return 1e10, _vec_from_gram(np.zeros((n, n)))
        
        ratio = max_d_sq / min_d_sq
        
        # Eigenvalue penalty
        pen = _eigenvalue_penalty(G, d)
        
        total = ratio + penalty_weight * pen
        
        # Gradient computation (numerical for reliability)
        eps = 1e-8
        grad_x = np.zeros_like(x)
        for idx in range(len(x)):
            x_plus = x.copy()
            x_plus[idx] += eps
            f_plus, _ = objective_and_grad(x_plus, penalty_weight)
            grad_x[idx] = (f_plus - total) / eps
        
        return total, grad_x
    
    def objective(x, penalty_weight=10.0):
        """Objective function for minimization."""
        val, _ = objective_and_grad(x, penalty_weight)
        return val
    
    # Initial configurations
    np.random.seed(42)
    
    # 1. Random PSD matrix: V @ V.T where V is n x d
    V_rand = np.random.randn(n, d)
    G_rand = V_rand @ V_rand.T
    x0_rand = _vec_from_gram(G_rand)
    
    # 2. Fibonacci sphere on unit sphere, then scale
    G_fib = np.zeros((n, n))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        G_fib[i, i] = 1.0  # unit norm
        for j in range(i):
            # Need to store previous points... let's just use random for now
            pass
    # Actually, let's construct properly
    pts_fib = np.zeros((n, d))
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        pts_fib[i] = [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    G_fib = pts_fib @ pts_fib.T
    x0_fib = _vec_from_gram(G_fib)
    
    # 3. Regular simplex-like: first d+1 vertices of regular simplex, rest random
    G_simplex = np.zeros((n, n))
    for i in range(d + 1):
        for j in range(d + 1):
            if i == j:
                G_simplex[i, j] = 1.0
            else:
                G_simplex[i, j] = -1.0 / d
    for i in range(d + 1, n):
        G_simplex[i, i] = 1.0
        for j in range(i):
            G_simplex[i, j] = np.random.uniform(-0.5, 0.5)
            G_simplex[j, i] = G_simplex[i, j]
    x0_simplex = _vec_from_gram(G_simplex)
    
    # 4. Grid-based
    side = int(np.ceil(n ** (1.0 / d)))
    grid_pts = list(itertools.product(range(side), repeat=d))[:n]
    pts_grid = np.array(grid_pts, dtype=float)
    G_grid = pts_grid @ pts_grid.T
    x0_grid = _vec_from_gram(G_grid)
    
    best_x = None
    best_val = 1e10
    
    # Try multiple initializations with decreasing penalty weight
    penalty_weights = [50.0, 10.0, 1.0, 0.1]
    
    for x0 in [x0_rand, x0_fib, x0_simplex, x0_grid]:
        for pw in penalty_weights:
            try:
                res = minimize(objective, x0, args=(pw,), method='L-BFGS-B',
                               options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Also try Nelder-Mead for local refinement
        try:
            res = minimize(objective, x0, args=(penalty_weights[-1],), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback
        pts = np.random.randn(n, d)
        return pts
    
    # Extract points from Gram matrix
    G = _gram_from_vec(best_x)
    G = (G + G.T) / 2
    
    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 0)
    G_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    G_psd = (G_psd + G_psd.T) / 2
    
    # Extract points via Cholesky or eigen-decomposition
    # G = V @ V.T, so V = eigvecs @ sqrt(eigvals)
    V = eigvecs @ np.diag(np.sqrt(eigvals))
    points = V
    
    # Refine with coordinate-based optimization
    def coord_objective(x_coords):
        pts = x_coords.reshape(n, d)
        dists = pdist(pts)
        if len(dists) == 0 or np.min(dists) < 1e-10:
            return 1e10
        return (np.max(dists) / np.min(dists)) ** 2
    
    try:
        res_coord = minimize(coord_objective, points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res_coord.fun < _compute_score(points):
            points = res_coord.x.reshape(n, d)
    except Exception:
        pass
    
    return points

# EVOLVE-BLOCK-END
