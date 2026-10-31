"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # safety margin
    
    def _pairwise_dists_sq(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        iu = np.triu_indices(N, k=1)
        return d2[iu]

    def _soft_objective(params, temp=0.01):
        pts = params.reshape(N, D)
        d2 = _pairwise_dists_sq(pts)
        d2 = np.maximum(d2, 1e-12)
        
        m_max = np.max(d2)
        # Soft max
        soft_max = m_max + temp * np.log(np.sum(np.exp((d2 - m_max) / temp)))
        
        m_min = np.min(d2)
        # Soft min
        soft_min = m_min - temp * np.log(np.sum(np.exp(-(d2 - m_min) / temp)))
        
        return soft_max / soft_min

    def _exact_score(pts):
        d2 = _pairwise_dists_sq(pts)
        min_d2 = np.min(d2)
        max_d2 = np.max(d2)
        if min_d2 < 1e-12:
            return np.inf
        return max_d2 / min_d2

    def _refine_from_init(init_pts):
        """
        Refine initial points using annealed soft objective.
        Returns (best_pts, best_score).
        """
        best_pts = init_pts.copy()
        best_score = _exact_score(init_pts)
        
        # Annealing schedule: start with high temp, decrease
        temps = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
        
        current_params = init_pts.flatten()
        
        for temp in temps:
            if time.time() - start_time > deadline - 5:
                break
            
            try:
                res = minimize(
                    lambda p, t=temp: _soft_objective(p, t),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_params = res.x
                pts = current_params.reshape(N, D)
                s = _exact_score(pts)
                if s < best_score:
                    best_score = s
                    best_pts = pts.copy()
            except Exception:
                continue
        
        return best_pts, best_score

    # Generate diverse initial configurations
    candidates = []
    
    # 1. Standard 4x4 Grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([X.flatten(), Y.flatten()])
    candidates.append(grid_pts)
    
    # 2. Compressed 4x4 Grid (corners moved in)
    comp_pts = grid_pts.copy()
    center = np.array([1.5, 1.5])
    for idx in [0, 3, 12, 15]:
        comp_pts[idx] = comp_pts[idx] + 0.3 * (center - comp_pts[idx])
    candidates.append(comp_pts)
    
    # 3. Hexagonal lattice subset
    # Generate a hexagonal lattice and take 16 points
    hex_pts = []
    # Center
    hex_pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for k in range(6):
        angle = np.pi / 3 * k
        hex_pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: Need 9 more. Generate all ring 2 and take 9.
    # Hexagonal coordinates (q, r)
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            if dist == 2.0:
                ring2.append((q, r))
    for q, r in ring2[:9]:
        x = q + r / 2.0
        y = r * np.sqrt(3) / 2.0
        hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    candidates.append(hex_pts)
    
    # 4. Random perturbations of Grid
    rng = np.random.RandomState(42)
    for i in range(10):
        perturbed = grid_pts + rng.randn(N, D) * 0.1
        candidates.append(perturbed)

    best_pts = None
    best_score = np.inf

    for i, init_pts in enumerate(candidates):
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            refined_pts, refined_score = _refine_from_init(init_pts)
            
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined_pts
                
        except Exception:
            continue

    # Final polish on the best
    if best_pts is not None and time.time() - start_time < deadline - 2:
        try:
            # One more refinement with very low temperature
            res_final = minimize(
                lambda p: _soft_objective(p, temp=0.0001),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-14, 'gtol': 1e-12}
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
# EVOLVE-BLOCK-END

from scipy.optimize import minimize
