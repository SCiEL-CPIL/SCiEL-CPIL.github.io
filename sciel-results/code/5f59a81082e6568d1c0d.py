import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2
DEADLINE = 140.0

def _pairwise_dists(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]

def _exact_score(pts):
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)

def _objective(params):
    """Exact objective: minimize max/min ratio of squared distances."""
    pts = params.reshape(N, D)
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)

def _nelder_mead_optimize(init_pts, t0, deadline):
    """Nelder-Mead local optimization with time guard."""
    try:
        res = minimize(
            _objective,
            init_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        return res.x.reshape(N, D)
    except Exception:
        return init_pts

def construct():
    t0 = time.time()
    
    # Generate diverse initial candidates
    rng = np.random.RandomState(42)
    
    candidates = []
    
    # 1. 4x4 Grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([X.flatten(), Y.flatten()])
    candidates.append(grid_pts)
    
    # 2. Triangular lattice
    tri_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0.5, np.sqrt(3)/2], [1.5, np.sqrt(3)/2], [2.5, np.sqrt(3)/2], [3.5, np.sqrt(3)/2],
        [1, np.sqrt(3)], [2, np.sqrt(3)], [3, np.sqrt(3)], [4, np.sqrt(3)],
        [1.5, 1.5*np.sqrt(3)], [2.5, 1.5*np.sqrt(3)], [3.5, 1.5*np.sqrt(3)], [4.5, 1.5*np.sqrt(3)]
    ])
    candidates.append(tri_pts)
    
    # 3. Random perturbations of grid
    for scale in [0.1, 0.2, 0.3, 0.5]:
        perturbed = grid_pts + rng.randn(N, D) * scale
        candidates.append(perturbed)
    
    # 4. Random perturbations of triangular
    for scale in [0.1, 0.2, 0.3]:
        perturbed = tri_pts + rng.randn(N, D) * scale
        candidates.append(perturbed)
    
    # 5. Pure random points in [0, 5]x[0, 5]
    for _ in range(3):
        random_pts = rng.uniform(0, 5, size=(N, D))
        candidates.append(random_pts)
    
    best_pts = None
    best_score = np.inf
    
    # Phase 1: Differential Evolution for global search
    # Use a bounded box around the grid
    bounds = [(-2, 5)] * (N * D)
    
    if time.time() - t0 < DEADLINE - 30:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            de_pts = de_result.x.reshape(N, D)
            de_score = _exact_score(de_pts)
            
            if de_score < best_score:
                best_score = de_score
                best_pts = de_pts
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starting points
    # Combine DE result with original candidates
    all_starts = list(candidates)
    if best_pts is not None:
        all_starts.insert(0, best_pts)
    
    for i, init_pts in enumerate(all_starts):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        try:
            refined = _nelder_mead_optimize(init_pts, t0, DEADLINE)
            score = _exact_score(refined)
            
            if score < best_score:
                best_score = score
                best_pts = refined
        except Exception:
            continue
    
    # Phase 3: Final polish on best
    if best_pts is not None and time.time() - t0 < DEADLINE - 2:
        try:
            final_pts = _nelder_mead_optimize(best_pts, t0, DEADLINE)
            final_score = _exact_score(final_pts)
            if final_score < best_score:
                best_pts = final_pts
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = grid_pts
    
    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
