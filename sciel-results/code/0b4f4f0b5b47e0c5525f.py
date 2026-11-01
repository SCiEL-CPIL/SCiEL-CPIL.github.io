"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
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
    
    # Precompute pairwise indices
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(indices)
    
    def get_distance_vector(G):
        """Return vector of squared distances for all pairs from Gram matrix."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = diag[i] + diag[j] - 2 * G[i, j]
        return d_sq

    def smooth_ratio(d_sq, T=0.05):
        """Compute smooth approximation of max/min ratio using log-sum-exp."""
        d_sq_clipped = np.maximum(d_sq, 1e-10)
        # Soft max: T * log(sum(exp(x/T)))
        max_approx = T * np.log(np.sum(np.exp(d_sq_clipped / T)))
        # Soft min: -T * log(sum(exp(-x/T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq_clipped / T)))
        if min_approx < 1e-10:
            return 1e10
        return max_approx / min_approx

    def objective(G_flat):
        G = G_flat.reshape(n, n)
        G = (G + G.T) / 2
        
        # Eigenvalue decomposition for PSD and rank constraints
        eigvals = np.linalg.eigvalsh(G)
        
        # PSD penalty
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2)
        
        # Rank penalty: eigenvalues beyond rank d should be zero
        rank_penalty = np.sum(eigvals[d:] ** 2)
        
        # Compute squared distances
        d_sq = get_distance_vector(G)
        
        # If distances are significantly negative, G is not PSD
        if np.any(d_sq < -1e-4):
            return 1e10 + 10.0 * (psd_penalty + rank_penalty)
        
        # Smooth ratio objective
        ratio = smooth_ratio(d_sq, T=0.05)
        
        # Scale normalization: fix trace to prevent trivial scaling
        trace_G = np.trace(G)
        scale_penalty = (trace_G - n) ** 2
        
        # Total objective
        return ratio + 5.0 * rank_penalty + 5.0 * psd_penalty + 1.0 * scale_penalty

    def extract_points(G):
        """Extract points from Gram matrix via eigen-decomposition."""
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep top d eigenvectors
        V = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))
        return V

    def refine_points(points):
        """Direct coordinate refinement using L-BFGS-B."""
        def obj(x):
            V = x.reshape(n, d)
            d_sq = pdist(V) ** 2
            if np.any(d_sq < 1e-10):
                return 1e10
            return smooth_ratio(d_sq, T=0.03)
        
        try:
            res = minimize(obj, points.flatten(), method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < smooth_ratio(pdist(points) ** 2, T=0.03):
                return res.x.reshape(n, d)
        except:
            pass
        return points

    # Multiple initializations
    np.random.seed(42)
    initializations = []
    
    # 1. Random rank-d Gram matrix
    V_rand = np.random.randn(n, d)
    G_rand = V_rand @ V_rand.T
    initializations.append(G_rand.flatten())
    
    # 2. Regular simplex-like in high dim, projected to d
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V_reg = np.column_stack([np.cos(angles), np.sin(angles), np.ones(n) * 0.1])
    G_reg = V_reg @ V_reg.T
    initializations.append(G_reg.flatten())
    
    # 3. Tetrahedron-based: repeat a 4-point regular simplex
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    # Scale and tile
    V_tetra = np.tile(tetra, (n // 4 + 1, 1))[:n]
    G_tetra = V_tetra @ V_tetra.T
    initializations.append(G_tetra.flatten())
    
    # 4. Identity-scaled
    G_id = np.eye(n)
    initializations.append(G_id.flatten())
    
    best_points = None
    best_score = 1e10
    
    for G0_flat in initializations:
        try:
            res = minimize(objective, G0_flat, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-9})
            if res.fun < 1e8:  # Only accept if not diverged
                G_opt = res.x.reshape(n, n)
                G_opt = (G_opt + G_opt.T) / 2
                points = extract_points(G_opt)
                points = refine_points(points)
                score = _compute_score(points)
                if score < best_score:
                    best_score = score
                    best_points = points
        except Exception as e:
            pass
    
    if best_points is None:
        # Fallback: regular simplex in d dimensions, then add jitter
        if d == 3:
            # 4 regular tetrahedron vertices, then perturb
            tetra = np.array([
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1]
            ]) * 0.5
            points = np.tile(tetra, (n // 4 + 1, 1))[:n]
            points += np.random.randn(n, d) * 0.1
        else:
            points = np.random.randn(n, d)
        return points
    
    return best_points

# EVOLVE-BLOCK-END
