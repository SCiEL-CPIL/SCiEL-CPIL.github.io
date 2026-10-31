"""Optimized 16-point placement: minimize (max_dist/min_dist)^2."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2
DEADLINE = 140.0  # safety margin before 150s kill


def _pairwise_dists(pts):
    """Compute all pairwise squared distances. pts: (16,2) -> (120,) array of d^2."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]


def _soft_objective(params, temp=0.01):
    """
    params: (32,) flattened points
    Returns a smooth surrogate for (max/min)^2 using log-sum-exp.
    """
    pts = params.reshape(N, D)
    d2 = _pairwise_dists(pts)
    # Guard against zero distances
    d2 = np.maximum(d2, 1e-12)
    
    # Soft max: log-sum-exp with temperature
    m = np.max(d2)
    soft_max = m + temp * np.log(np.sum(np.exp((d2 - m) / temp)))
    # Soft min: -log-sum-exp of negatives
    m2 = np.min(d2)
    soft_min = m2 - temp * np.log(np.sum(np.exp(-(d2 - m2) / temp)))
    ratio2 = soft_max / soft_min
    return ratio2


def _exact_score(pts):
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)


def _get_initial_candidates():
    """Generate diverse initial candidates for local optimization."""
    candidates = []
    
    # 4x4 grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    candidates.append(np.column_stack([X.flatten(), Y.flatten()]))
    
    # Circle arrangement
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0
    candidates.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    
    # Two concentric circles (8+8)
    ang1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ang2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    r1, r2 = 0.5, 1.5
    c1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    c2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    candidates.append(np.vstack([c1, c2]))
    
    # Hexagonal packing pattern
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + (row % 2) * 0.5
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    candidates.append(np.array(hex_pts))
    
    # Spiral
    t = np.linspace(0, 3*np.pi, N)
    r = t / (3*np.pi)
    candidates.append(np.column_stack([r * np.cos(t), r * np.sin(t)]))
    
    # Random restarts with various scales
    for seed in range(8):
        rng_local = np.random.RandomState(seed * 100 + 7)
        scale = 0.5 + seed * 0.2
        pts = rng_local.randn(N, D) * scale
        candidates.append(pts)
    
    # Perturbed grid
    for seed in range(4):
        rng_local = np.random.RandomState(seed * 200 + 13)
        gx = np.array([0, 1, 2, 3], dtype=float)
        gy = np.array([0, 1, 2, 3], dtype=float)
        X, Y = np.meshgrid(gx, gy)
        X = X.flatten() + rng_local.randn(N) * 0.3
        Y = Y.flatten() + rng_local.randn(N) * 0.3
        candidates.append(np.column_stack([X, Y]))
    
    return candidates


def _local_optimize(pts_init, temp=0.01, maxiter=5000):
    """Polish a candidate with L-BFGS-B on soft objective."""
    try:
        res = minimize(
            lambda p: _soft_objective(p, temp),
            pts_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return res.x.reshape(N, D)
    except Exception:
        return pts_init


def construct():
    t0 = time.time()
    
    # ---- Phase 1: Global search with Differential Evolution on low-dim params ----
    # Use a parameterization: 4x4 grid with sinusoidal deformation
    def _de_objective(p):
        a1, a2, b1, b2, c1, c2, ph1, ph2, scale = p
        gx = np.array([0, 1, 2, 3], dtype=float)
        gy = np.array([0, 1, 2, 3], dtype=float)
        X, Y = np.meshgrid(gx, gy)
        X = X.flatten()
        Y = Y.flatten()
        # Deformation
        X = X + a1 * np.sin(b1 * X + ph1) * scale + c1 * X
        Y = Y + a2 * np.sin(b2 * Y + ph2) * scale + c2 * Y
        pts = np.column_stack([X, Y])
        return _soft_objective(pts.flatten(), temp=0.01)
    
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0.5, 2.0), (0.5, 2.0),
              (-0.3, 0.3), (-0.3, 0.3), (0.0, 2*np.pi), (0.0, 2*np.pi),
              (0.0, 0.5)]
    
    best_de_pts = None
    try:
        result_de = differential_evolution(
            _de_objective, bounds, seed=42, maxiter=300, tol=1e-10,
            popsize=20, polish=True, init='sobol',
            time_limit=50
        )
        if time.time() - t0 < DEADLINE:
            a1, a2, b1, b2, c1, c2, ph1, ph2, scale = result_de.x
            gx = np.array([0, 1, 2, 3], dtype=float)
            gy = np.array([0, 1, 2, 3], dtype=float)
            X, Y = np.meshgrid(gx, gy)
            X = X.flatten() + a1 * np.sin(b1 * X + ph1) * scale + c1 * X
            Y = Y.flatten() + a2 * np.sin(b2 * Y + ph2) * scale + c2 * Y
            best_de_pts = np.column_stack([X, Y])
    except Exception:
        pass
    
    # ---- Phase 2: Collect candidates and polish ----
    candidates = _get_initial_candidates()
    if best_de_pts is not None:
        candidates.append(best_de_pts)
    
    # Also try direct 32D DE with smaller budget
    if time.time() - t0 < DEADLINE - 40:
        try:
            result_de2 = differential_evolution(
                lambda p: _soft_objective(p, temp=0.01),
                [(-3, 3)] * 32, seed=123, maxiter=150, tol=1e-8,
                popsize=15, polish=True, init='sobol',
                time_limit=35
            )
            candidates.append(result_de2.x.reshape(N, D))
        except Exception:
            pass
    
    # Polish each candidate with multiple temperatures
    best_pts = None
    best_score = np.inf
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 15:
            break
        
        # Try multiple temperature scales
        for temp in [0.001, 0.01, 0.1]:
            if time.time() - t0 > DEADLINE - 10:
                break
            try:
                polished = _local_optimize(init_pts, temp=temp, maxiter=3000)
                score = _exact_score(polished)
                if score < best_score:
                    best_score = score
                    best_pts = polished
            except Exception:
                continue
    
    # ---- Phase 3: Final high-precision polish on best point ----
    if best_pts is not None and time.time() - t0 < DEADLINE - 5:
        try:
            # Use very tight temperature for final polish
            res_final = minimize(
                lambda p: _soft_objective(p, temp=0.0001),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            pts_final = res_final.x.reshape(N, D)
            score_final = _exact_score(pts_final)
            if score_final < best_score:
                best_pts = pts_final
                best_score = score_final
        except Exception:
            pass
    
    # ---- Phase 4: Additional local refinement with different starting points ----
    if best_pts is not None and time.time() - t0 < DEADLINE - 3:
        # Try perturbing the best solution and re-optimizing
        rng = np.random.RandomState(999)
        for _ in range(5):
            if time.time() - t0 > DEADLINE - 2:
                break
            perturbed = best_pts + rng.randn(N, D) * 0.05
            try:
                polished = _local_optimize(perturbed, temp=0.001, maxiter=2000)
                score = _exact_score(polished)
                if score < best_score:
                    best_score = score
                    best_pts = polished
            except Exception:
                continue
    
    if best_pts is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts


if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
