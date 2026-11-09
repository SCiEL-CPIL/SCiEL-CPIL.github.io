import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time


def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 560  # safety margin
    n = 14

    def normalize_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        mask = norms < 1e-12
        if np.any(mask):
            norms[mask] = 1.0
            pts[mask, 0] = 1.0
            pts[mask, 1:] = 0.0
        return pts / norms

    def ratio(pts):
        d = pdist(pts)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return d_min / d_max

    def neg_ratio(x):
        pts = x.reshape(n, 3)
        pts = normalize_sphere(pts)
        return -ratio(pts)

    def soft_neg_ratio(x, beta=50.0):
        pts = x.reshape(n, 3)
        pts = normalize_sphere(pts)
        d = pdist(pts)
        d_min_soft = -np.logaddexp.reduce(-d / beta) * beta
        d_max_soft = np.logaddexp.reduce(d / beta) * beta
        if d_max_soft <= 1e-15:
            return 0.0
        return -(d_min_soft / d_max_soft)

    def simulated_annealing(pts, steps=3000, T_start=0.1, T_end=0.001, seed=42):
        """Simulated annealing on the sphere to improve the ratio."""
        rng = np.random.RandomState(seed)
        pts = pts.copy()
        current_score = ratio(pts)
        best_pts = pts.copy()
        best_score = current_score
        
        for step in range(steps):
            if time.time() > deadline - 30:
                break
            T = T_start * (T_end / T_start) ** (step / max(steps - 1, 1))
            
            # Propose a move: random perturbation of one or two points
            n_perturb = rng.choice([1, 2], p=[0.7, 0.3])
            indices = rng.choice(n, size=n_perturb, replace=False)
            
            new_pts = pts.copy()
            for idx in indices:
                # Random direction
                v = rng.randn(3)
                v = v / np.linalg.norm(v)
                # Small step
                step_size = 0.05 * T / T_start + 0.01
                new_pts[idx] = pts[idx] + step_size * v
                new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
            new_score = ratio(new_pts)
            
            # Acceptance criterion
            if new_score > current_score or rng.rand() < np.exp((new_score - current_score) / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
        
        return best_pts, best_score

    # Base configurations on sphere
    k = np.arange(n)
    phi = np.arccos(1 - 2 * (k + 0.5) / n)
    golden = np.pi * (1 + np.sqrt(5))
    fib = np.column_stack([
        np.sin(phi) * np.cos(golden * k),
        np.sin(phi) * np.sin(golden * k),
        np.cos(phi)
    ])

    # Icosahedron 12 vertices
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
        [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
        [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 poles to make 14
    ico14 = np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]])

    # Create diverse initial candidates on sphere
    np.random.seed(777)
    candidates = [
        fib,
        ico14,
    ]
    
    # Add perturbed versions
    for base in [fib, ico14]:
        for noise_level in [0.05, 0.1, 0.2, 0.3]:
            perturbed = base + noise_level * np.random.randn(n, 3)
            perturbed = normalize_sphere(perturbed)
            candidates.append(perturbed)
    
    # Add random sphere points
    for _ in range(8):
        v = np.random.randn(n, 3)
        v = normalize_sphere(v)
        candidates.append(v)

    best_pts = None
    best_r = -1.0

    # Phase 1: Simulated Annealing from multiple starts
    for i, cand in enumerate(candidates[:6]):  # Limit to 6 best starts for SA
        if time.time() > deadline - 200:
            break
        try:
            sa_pts, sa_r = simulated_annealing(cand, steps=2000, T_start=0.05, T_end=0.001, seed=42+i)
            if np.isfinite(sa_r) and sa_r > best_r:
                best_r = sa_r
                best_pts = sa_pts
        except Exception:
            pass

    # Phase 2: Powell optimization with exact ratio from SA results and original candidates
    if best_pts is not None:
        candidates_opt = [best_pts]
    else:
        candidates_opt = [fib, ico14]
    
    # Add a few more original candidates for Powell
    candidates_opt.extend(candidates[6:10])
    
    for i, cand in enumerate(candidates_opt):
        if time.time() > deadline - 120:
            break
        x0 = cand.flatten()
        try:
            res = minimize(neg_ratio, x0, method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            r = -float(res.fun)
            if np.isfinite(r) and r > best_r:
                best_r = r
                best_pts = res.x.reshape(n, 3)
        except Exception:
            pass

    # Phase 3: Refine best with soft objective for smoother landscape
    if best_pts is not None and time.time() < deadline - 80:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 60:
                break
            try:
                res_soft = minimize(lambda x: soft_neg_ratio(x, beta), 
                                   best_pts.flatten(), 
                                   method='Powell',
                                   options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11})
                r_soft = -float(res_soft.fun)
                # Evaluate with exact ratio
                pts_soft = normalize_sphere(res_soft.x.reshape(n, 3))
                r_exact = ratio(pts_soft)
                if np.isfinite(r_exact) and r_exact > best_r:
                    best_r = r_exact
                    best_pts = pts_soft
            except Exception:
                pass

    # Phase 4: Final high-precision polish with Nelder-Mead
    if best_pts is not None and time.time() < deadline - 30:
        try:
            res2 = minimize(neg_ratio, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            r2 = -float(res2.fun)
            if np.isfinite(r2) and r2 > best_r:
                best_r = r2
                best_pts = normalize_sphere(res2.x.reshape(n, 3))
        except Exception:
            pass

    # Phase 5: One more SA round from the best point found
    if best_pts is not None and time.time() < deadline - 10:
        try:
            sa_pts2, sa_r2 = simulated_annealing(best_pts, steps=1000, T_start=0.02, T_end=0.0005, seed=999)
            if np.isfinite(sa_r2) and sa_r2 > best_r:
                best_r = sa_r2
                best_pts = sa_pts2
        except Exception:
            pass

    if best_pts is None:
        best_pts = fib

    best_pts = np.nan_to_num(best_pts, nan=0.0, posinf=0.0, neginf=0.0)
    return best_pts
