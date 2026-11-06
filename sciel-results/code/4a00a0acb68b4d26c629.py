"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def objective(x):
        """Objective for optimization: minimize negative log score."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        # log(score) = 2*(sum(log(scaled_d)) - n*log(n))
        # scaled_d = d * 2/D
        # log(scaled_d) = log(d) + log(2) - log(D)
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Regular octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    reg_score = compute_score(reg_pts)

    best_score = reg_score
    best_pts = reg_pts.copy()

    def try_candidate(pts):
        nonlocal best_score, best_pts
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # Try structured configurations that break octagon symmetry
    # Two concentric squares rotated 45 degrees
    for r1 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for r2 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            if time.time() > deadline - 20.0:
                break
            angles_sq1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            angles_sq2 = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
            pts1 = np.column_stack([r1 * np.cos(angles_sq1), r1 * np.sin(angles_sq1)])
            pts2 = np.column_stack([r2 * np.cos(angles_sq2), r2 * np.sin(angles_sq2)])
            pts = np.vstack([pts1, pts2])
            try_candidate(pts)

    # Perturbed regular octagon with varying radii
    np.random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 20.0:
            break
        radii = 1.0 + 0.3 * np.random.randn(n)
        radii = np.abs(radii) + 0.1
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.1 * np.random.randn(n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        try_candidate(pts)

    # Use differential evolution for global search
    bounds = [(-3, 3)] * 16

    def de_objective(x):
        if time.time() > deadline - 5.0:
            return 1e10
        return objective(x)

    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=False,
            init='latinhypercube'
        )
        if time.time() < deadline - 10.0 and de_result.fun < objective(best_pts.flatten()):
            de_pts = de_result.x.reshape(n, 2)
            try_candidate(de_pts)
    except Exception:
        pass

    # Refine best candidate with L-BFGS-B
    if time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass

    # Final refinement with tighter tolerance
    if time.time() < deadline - 2.0:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass

    # Ensure distinct points and valid coordinates
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        # Fallback to regular octagon
        best_pts = reg_pts

    # Scale to ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
