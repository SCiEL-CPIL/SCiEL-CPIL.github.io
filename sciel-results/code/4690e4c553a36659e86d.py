import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def _energy(X):
    """Calculate the repulsive energy of the configuration."""
    # X is (N, D)
    # Compute pairwise distances
    dists_sq = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * (X @ X.T)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Avoid division by zero or very small numbers
    dists_sq = np.maximum(dists_sq, 1e-10)
    
    # Repulsive potential: sum 1/d^2
    # We exclude the diagonal
    energy = np.sum(1.0 / dists_sq)
    return energy

def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    best_X = None
    best_score = np.inf
    
    # Initial guess: Icosahedron vertices + 2 points
    # Icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 more points. We can place them randomly or at poles if available.
    # The icosahedron has 5-fold symmetry. Adding 2 points is tricky.
    # Let's just use random initialization for robustness, or mix.
    
    initializations = []
    
    # 1. Icosahedron + 2 random points
    X_ico = ico.copy()
    X_rand = np.random.randn(2, 3)
    X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
    initializations.append(np.vstack([X_ico, X_rand]))
    
    # 2. Fully random
    X_rand_full = np.random.randn(N, 3)
    X_rand_full /= np.linalg.norm(X_rand_full, axis=1, keepdims=True)
    initializations.append(X_rand_full)
    
    # 3. Another random
    X_rand_full2 = np.random.randn(N, 3)
    X_rand_full2 /= np.linalg.norm(X_rand_full2, axis=1, keepdims=True)
    initializations.append(X_rand_full2)
    
    for X0 in initializations:
        if time.time() > deadline:
            break
            
        X0 = _project_to_sphere(X0)
        
        # Define objective to minimize
        def obj(X_flat):
            X = X_flat.reshape(N, D)
            X = _project_to_sphere(X)
            return _energy(X)
        
        # Use L-BFGS-B
        try:
            res = minimize(obj, X0.flatten(), method='L-BFGS-B', options={'maxiter': 1000})
            X_opt = _project_to_sphere(res.x.reshape(N, D))
            
            # Calculate actual score
            dists_sq = np.sum(X_opt**2, axis=1)[:, None] + np.sum(X_opt**2, axis=1)[None, :] - 2 * (X_opt @ X_opt.T)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(np.maximum(dists_sq, 0))
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            pairwise_dists = dists[mask]
            
            if len(pairwise_dists) == 0:
                continue
                
            d_min = np.min(pairwise_dists)
            d_max = np.max(pairwise_dists)
            
            if d_min < 1e-6:
                continue
                
            score = (d_max / d_min) ** 2
            
            if score < best_score:
                best_score = score
                best_X = X_opt
                
        except Exception:
            pass

    if best_X is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_X

# Need itertools for fallback
import itertools
