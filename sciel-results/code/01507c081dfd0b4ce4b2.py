import numpy as np
from scipy.optimize import minimize
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    n = 8
    num_pairs = n * (n - 1) // 2  # 28

    # Free indices (0 and 4 are fixed at (0,0) and (2,0))
    fixed_idx = [0, 4]
    free_idx = [i for i in range(n) if i not in fixed_idx]
    n_free = len(free_idx)  # 6

    def unpack(x):
        p = np.zeros((n, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2 * i], x[2 * i + 1]]
        return p

    def objective(x, beta=200.0):
        """
        Minimize: -sum(log d_ij) + num_pairs * log(soft_max(d_ij, beta))
        Higher beta gives tighter enforcement of diameter constraint.
        """
        p = unpack(x)
        # Compute all pairwise distances
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        # Upper triangle only
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)

        sum_log_d = np.sum(np.log(d))

        # Soft-max via log-sum-exp: log(sum(exp(beta * d))) / beta
        max_d = np.max(d)
        scaled = beta * (d - max_d)
        scaled = np.clip(scaled, -500, 500)
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_max = max_d + log_sum / beta

        # Objective: minimize -sum_log_d + num_pairs * log(soft_max)
        val = -sum_log_d + num_pairs * np.log(soft_max)
        return val

    # Initial guess: Regular octagon with P0=(0,0), P4=(2,0)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_unit = np.column_stack((np.cos(angles), np.sin(angles)))
    init_pts = -pts_unit + np.array([1.0, 0.0])

    x0 = np.array([init_pts[i] for i in free_idx]).flatten()

    best_x = x0.copy()
    best_obj = objective(x0)

    # Phase 1: Local optimization from regular octagon with BFGS
    try:
        res = minimize(objective, x0, method='BFGS',
                       options={'maxiter': 500, 'gtol': 1e-10, 'disp': False})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Phase 2: Multi-restart with structured perturbations
    np.random.seed(42)
    random.seed(42)

    restarts = []

    # Structured perturbations preserving some symmetry
    # Perturb points in pairs that are symmetric about the x-axis
    for scale in [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        for trial in range(4):
            noise = np.random.normal(0, scale, size=(n, 2))
            noise[0] = 0
            noise[4] = 0
            # Enforce x-axis symmetry for pairs (1,7), (2,6), (3,5)
            for pair in [(1, 7), (2, 6), (3, 5)]:
                avg_noise = (noise[pair[0]] + noise[pair[1]]) / 2
                noise[pair[0]] = avg_noise
                noise[pair[1]] = avg_noise
                # Flip y-component for symmetry
                noise[pair[1]][1] = -noise[pair[0]][1]
            perturbed = init_pts + noise
            x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
            restarts.append(x0_pert)

    # General random perturbations
    for scale in [0.6, 0.8, 1.0, 1.2, 1.5]:
        for trial in range(3):
            noise = np.random.normal(0, scale, size=(n, 2))
            noise[0] = 0
            noise[4] = 0
            perturbed = init_pts + noise
            x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
            restarts.append(x0_pert)

    # Random starts in reasonable range
    for trial in range(15):
        x0_rand = np.random.uniform(-1, 3, size=n_free * 2)
        restarts.append(x0_rand)

    for x0_r in restarts:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(objective, x0_r, method='BFGS',
                           options={'maxiter': 300, 'gtol': 1e-9, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # Phase 3: High-precision polish with L-BFGS-B
    if time.time() < deadline - 8:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 4: Final polish with BFGS at very tight tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='BFGS',
                           options={'maxiter': 5000, 'gtol': 1e-15, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 5: One more L-BFGS-B polish
    if time.time() < deadline - 3:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 3000, 'gtol': 1e-15, 'ftol': 1e-16, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    p = unpack(best_x)

    # Ensure distinct points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-8:
                p[j] += np.array([1e-6, 0])

    # Scale to ensure diameter is exactly 2
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(p[i] - p[j])
            if d > max_d:
                max_d = d

    if max_d > 0 and abs(max_d - 2.0) > 1e-12:
        scale_factor = 2.0 / max_d
        p = p * scale_factor

    # Return as list of lists of floats
    return [[float(p[i][0]), float(p[i][1])] for i in range(n)]
