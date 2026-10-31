import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 142.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]

    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]

    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]

    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)

    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective using hard min (non-smooth, for Nelder-Mead only)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """Smooth surrogate for the hard min via log-sum-exp."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)

    if np.any(areas < 1e-15):
        return 1e10

    scaled = -beta * areas
    max_scaled = np.max(scaled)

    if max_scaled > 700:
        return -np.min(areas)

    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def epigraph_obj(x, lam=1e5):
    """
    Epigraph formulation: minimize -t subject to t <= all areas.
    x = [bary_flat (22), t (1)]
    This is the smooth, directly-modeled objective (measured improvement).
    """
    bary_flat = x[:N*2]
    t = x[N*2]

    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)

    # Penalize violations: t > areas[i]
    violations = np.maximum(0.0, t - areas)
    penalty = lam * np.sum(violations**2)

    return -t + penalty

def epigraph_value(bary):
    """Compute the achieved minimum area for a bary configuration."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return float(np.min(areas))

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf

    scales = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    offsets = [0.0, 0.05, 0.1]

    for scale in scales:
        for offset_u in offsets:
            for offset_v in offsets:
                pts = []
                max_k = int(1.0 / scale) + 2
                for k in range(max_k + 1):
                    w_val = k * scale
                    if w_val > 1.0:
                        break
                    num_pts_in_row = int((1 - w_val) / scale) + 1
                    for j in range(num_pts_in_row):
                        u_val = j * scale + offset_u
                        v_val = (1 - w_val) - u_val + offset_v
                        if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                            pts.append([u_val, v_val])

                if len(pts) >= N:
                    rng = np.random.default_rng(42 + int(scale*100) + int(offset_u*100) + int(offset_v*100))
                    for _ in range(3):
                        if len(pts) == N:
                            bary = np.array(pts)
                        else:
                            idx = rng.choice(len(pts), N, replace=False)
                            bary = np.array(pts)[idx]
                        val = epigraph_value(bary)
                        if val > best_val:
                            best_val = val
                            best_bary = bary.copy()

    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multi-stage optimization,
    anchored on the epigraph formulation for gradient methods.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)

    bary_current = bary_init.copy()
    current_val = epigraph_value(bary_current)

    # Stage 1: Epigraph L-BFGS-B (smooth, directly models max-min)
    try:
        t0 = max(1e-6, current_val * 0.5)
        x0 = np.concatenate([bary_current.flatten(), [t0]])
        x_bounds = bounds + [(0.0, 0.5)]
        result1 = minimize(
            epigraph_obj,
            x0,
            args=(1e5,),
            method='L-BFGS-B',
            bounds=x_bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        cand = clamp_bary(result1.x[:N*2].reshape(N, 2))
        val = epigraph_value(cand)
        if val > current_val:
            bary_current = cand
            current_val = val
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 2: Soft-min L-BFGS-B (smooth surrogate, pushes all areas up)
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        cand = clamp_bary(result2.x.reshape(N, 2))
        val = epigraph_value(cand)
        if val > current_val:
            bary_current = cand
            current_val = val
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 3: Nelder-Mead on hard min (non-smooth polish)
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
        )
        cand = clamp_bary(result3.x.reshape(N, 2))
        val = epigraph_value(cand)
        if val > current_val:
            bary_current = cand
            current_val = val
    except Exception:
        pass

    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = epigraph_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Hexagonal Init + refine
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)

    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False,
                mutation=(0.3, 1.0),
                recombination=0.7
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)

        remaining = deadline - 15 - time.time()
        if remaining < 4:
            break

        t_limit = min(6.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 10:
        try:
            t0 = max(1e-6, best_val * 0.9)
            x0 = np.concatenate([best_bary.flatten(), [t0]])
            x_bounds = [(0, 1)] * (N * 2) + [(0.0, 0.5)]
            result_e1 = minimize(
                epigraph_obj,
                x0,
                args=(1e6,),
                method='L-BFGS-B',
                bounds=x_bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            update_best(clamp_bary(result_e1.x[:N*2].reshape(N, 2)))
        except Exception:
            pass

        if best_bary is not None and time.time() < deadline - 5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

        if best_bary is not None and time.time() < deadline - 2:
            try:
                t0 = max(1e-6, best_val * 0.99)
                x0 = np.concatenate([best_bary.flatten(), [t0]])
                x_bounds = [(0, 1)] * (N * 2) + [(0.0, 0.5)]
                result_e2 = minimize(
                    epigraph_obj,
                    x0,
                    args=(1e7,),
                    method='L-BFGS-B',
                    bounds=x_bounds,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13}
                )
                update_best(clamp_bary(result_e2.x[:N*2].reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)

    return cart
