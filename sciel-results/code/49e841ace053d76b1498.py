"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin

    def score_points(pts):
        d = pdist(pts)
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    # --- Strategy: Use a well-known good configuration as starting point ---
    # For 16 points, a good arrangement is a 4x4 grid slightly perturbed,
    # or points on a circle, or a hexagonal lattice subset.
    # Let's try multiple initial configurations and pick the best.

    configs = []

    # Config 1: 4x4 Grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    configs.append(grid)

    # Config 2: Hexagonal lattice subset (16 points)
    hex_pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    dists_from_origin = np.sum(hex_pts ** 2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    configs.append(hex_pts[sorted_indices[:16]])

    # Config 3: Two concentric squares (4 inner, 12 outer?) No, let's do 8 inner, 8 outer
    # Inner: 4 points on circle radius 0.5, Outer: 12 points on circle radius 1.5?
    # Let's try: 4 points at (0.5,0), (-0.5,0), (0,0.5), (0,-0.5)
    # 12 points on circle radius 1.2
    inner = np.array([[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]])
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer = np.column_stack([1.2 * np.cos(angles), 1.2 * np.sin(angles)])
    configs.append(np.vstack([inner, outer]))

    # Config 4: Regular 16-gon
    angles16 = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    configs.append(np.column_stack([np.cos(angles16), np.sin(angles16)]))

    # Config 5: Perturbed grid
    rng = np.random.default_rng(42)
    grid_perturbed = grid + rng.normal(0, 0.05, size=grid.shape)
    configs.append(grid_perturbed)

    # Config 6: Another hexagonal arrangement
    # Center + 6 around + 9 more?
    # Let's just use the best from the above configs.

    best_pts = None
    best_score = 1e10
    for cfg in configs:
        # Normalize scale doesn't matter for ratio, but let's keep it reasonable
        s = score_points(cfg)
        if s < best_score:
            best_score = s
            best_pts = cfg.copy()

    if best_pts is None:
        best_pts = configs[0]
        best_score = score_points(best_pts)

    # --- Optimization Phase ---
    # We will use L-BFGS-B on the raw coordinates.
    # To handle symmetries (translation, rotation, scaling), we can fix 2 points.
    # Fix point 0 at (0,0) and point 1 at (1,0).
    # This reduces parameters from 32 to 28.

    # Identify the pair with minimum distance in best_pts to fix?
    # Or just fix points 0 and 1 and optimize the rest.
    # If points 0 and 1 are not the closest, the optimizer might struggle.
    # Better: Find the closest pair in the best initial config, fix them.

    d_init = pdist(best_pts)
    min_idx = np.argmin(d_init)
    # pdist returns in order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    # We need to map linear index to pair (i,j)
    # i = int((n-1 - sqrt(1-8*min_idx))/2) ? No, standard formula.
    # Let's just iterate to find i,j
    idx = 0
    i0, i1 = 0, 1
    for i in range(N - 1):
        for j in range(i + 1, N):
            if idx == min_idx:
                break
            idx += 1
        if idx == min_idx:
            break

    # Reorder points so that i0 is 0 and i1 is 1
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining

    pts_perm = best_pts[perm]

    # Transform: translate by -pts_perm[0], rotate so pts_perm[1] is at (1,0)
    p0 = pts_perm[0]
    p1 = pts_perm[1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-8:
        return best_pts

    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    # Transform all points
    transformed = (pts_perm - p0) @ R_mat.T / norm_v

    # Now transformed[0] is (0,0) and transformed[1] is (1,0)
    init_params = transformed[2:].flatten()

    def get_dists(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        return pdist(pts)

    def objective(params):
        d = get_dists(params)
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    # Bounds: points should not be too far from the unit segment
    # A reasonable bound is [-10, 10]
    bnds = [(-10, 10)] * 28

    best_params = init_params.copy()
    best_opt_score = objective(best_params)

    # Multi-start optimization with L-BFGS-B
    # We will try a few random restarts if time permits

    restarts = 3
    for r in range(restarts):
        if time.time() > deadline - 20:
            break

        # For restarts > 0, add noise to initial params
        if r == 0:
            curr_params = best_params.copy()
        else:
            curr_params = best_params + np.random.normal(0, 0.1, size=28)

        try:
            res = minimize(objective, curr_params, method='L-BFGS-B',
                           bounds=bnds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < best_opt_score:
                best_opt_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                           bounds=bnds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_opt_score:
                best_opt_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Reconstruct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)

    # Inverse transform: scale by norm_v, rotate by R_mat^T, translate by p0
    final_pts = final_pts * norm_v @ R_mat.T + p0

    # Reorder back to original permutation? Not necessary, any order is fine.
    # But let's check the score.
    final_score = score_points(final_pts)

    # Compare with the best initial config (which might have been better if optimization failed)
    if final_score < best_score:
        return final_pts
    else:
        return best_pts

# EVOLVE-BLOCK-END
