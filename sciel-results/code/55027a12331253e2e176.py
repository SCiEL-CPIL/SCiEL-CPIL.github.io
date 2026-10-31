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

def _soft_objective(params, temp=0.01):
    pts = params.reshape(N, D)
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-12)
    m = np.max(d2)
    soft_max = m + temp * np.log(np.sum(np.exp((d2 - m) / temp)))
    m2 = np.min(d2)
    soft_min = m2 - temp * np.log(np.sum(np.exp(-(d2 - m2) / temp)))
    return soft_max / soft_min

def _exact_score(pts):
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)

def construct():
    t0 = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Standard 4x4 Grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([X.flatten(), Y.flatten()])
    candidates.append(grid_pts)
    
    # 2. Compressed 4x4 Grid (corners moved in)
    # Move corners (0,0), (3,0), (0,3), (3,3) towards center (1.5, 1.5)
    comp_pts = grid_pts.copy()
    # Indices of corners
    # (0,0) is 0, (3,0) is 3, (0,3) is 12, (3,3) is 15
    center = np.array([1.5, 1.5])
    for idx in [0, 3, 12, 15]:
        comp_pts[idx] = comp_pts[idx] + 0.3 * (center - comp_pts[idx])
    candidates.append(comp_pts)
    
    # 3. Triangular Lattice Rhombus
    tri_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0.5, np.sqrt(3)/2], [1.5, np.sqrt(3)/2], [2.5, np.sqrt(3)/2], [3.5, np.sqrt(3)/2],
        [1, np.sqrt(3)], [2, np.sqrt(3)], [3, np.sqrt(3)], [4, np.sqrt(3)],
        [1.5, 1.5*np.sqrt(3)], [2.5, 1.5*np.sqrt(3)], [3.5, 1.5*np.sqrt(3)], [4.5, 1.5*np.sqrt(3)]
    ])
    candidates.append(tri_pts)
    
    # 4. Random perturbations of Grid
    rng = np.random.RandomState(42)
    for i in range(10):
        perturbed = grid_pts + rng.randn(N, D) * 0.1
        candidates.append(perturbed)

    best_pts = None
    best_score = np.inf

    for i, init_pts in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        try:
            # Phase 1: Coarse Optimization
            res1 = minimize(
                lambda p: _soft_objective(p, temp=0.1),
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Phase 2: Fine Optimization
            res2 = minimize(
                lambda p: _soft_objective(p, temp=0.01),
                res1.x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = res2.x.reshape(N, D)
            score = _exact_score(pts)
            
            if score < best_score:
                best_score = score
                best_pts = pts
                
        except Exception:
            continue

    # Final polish on the best
    if best_pts is not None and time.time() - t0 < DEADLINE - 2:
        try:
            res_final = minimize(
                lambda p: _soft_objective(p, temp=0.001),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            pts_final = res_final.x.reshape(N, D)
            score_final = _exact_score(pts_final)
            if score_final < best_score:
                best_pts = pts_final
        except Exception:
            pass

    if best_pts is None:
        best_pts = grid_pts

    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
