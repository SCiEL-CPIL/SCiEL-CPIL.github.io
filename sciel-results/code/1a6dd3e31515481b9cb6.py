import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-12:
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(x_flat):
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    shifted = np.clip(shifted, 0, 100)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(x_flat, temperature):
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(x_flat, temperature):
    s = _soft_score(x_flat, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Radial parameterization ----------
# Points on a circle: r_i * (cos(theta_i), sin(theta_i))
# This naturally produces convex configurations when angles are sorted

def _radial_to_cartesian(params):
    """params: (2N,) -> (N, 2)
    First N values are radii (log-scale), next N are angles (radians)
    """
    radii = np.exp(params[:N])  # positive radii
    angles = params[N:]
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _cartesian_to_radial(points):
    """(N, 2) -> (2N,)"""
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    radii = np.maximum(radii, 1e-10)
    angles = np.arctan2(points[:, 1], points[:, 0])
    return np.concatenate([np.log(radii), angles])


def _neg_score_radial(params):
    points = _radial_to_cartesian(params)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _neg_soft_score_radial(params, temperature):
    points = _radial_to_cartesian(params)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 1e10
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 1e10
    return -(soft_min / hull_area)


# ---------- Half-plane / support function parameterization ----------
# Convex polygon as intersection of half-planes with fixed normal angles
# 13 support lines with evenly spaced normals, optimize 13 offsets

def _support_to_points(offsets):
    """Convert 13 support offsets to 13 points (vertices of the convex polygon).
    The normals are fixed at angles 0, 2pi/13, 4pi/13, ...
    We find the intersection of consecutive half-planes to get vertices.
    """
    n = len(offsets)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # For each vertex, it's the intersection of two adjacent support lines
    # Line i: normal_i . x = offset_i
    # Vertex i is intersection of line (i-1) and line i
    vertices = []
    for i in range(n):
        i_prev = (i - 1) % n
        n1 = normals[i_prev]
        n2 = normals[i]
        o1 = offsets[i_prev]
        o2 = offsets[i]
        
        # Solve: n1.x = o1, n2.x = o2
        A = np.array([n1, n2])
        b = np.array([o1, o2])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            vertices.append(np.array([0.0, 0.0]))
        else:
            x = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
            y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
            vertices.append(np.array([x, y]))
    
    return np.array(vertices)


def _neg_score_support(offsets):
    points = _support_to_points(offsets)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _neg_soft_score_support(offsets, temperature):
    points = _support_to_points(offsets)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 1e10
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 1e10
    return -(soft_min / hull_area)


# ---------- Seed generators ----------

def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular():
    rng = np.random.RandomState(42)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.05 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_random_ring(rng):
    n_outer = int(rng.integers(5, 10))
    n_inner = N - n_outer
    if n_inner > 0:
        o_angles = np.sort(rng.uniform(0, 2 * np.pi, n_outer))
        i_angles = np.sort(rng.uniform(0, 2 * np.pi, n_inner))
        r_in = rng.uniform(0.3, 0.7)
        o = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
        i = np.column_stack([r_in * np.cos(i_angles), r_in * np.sin(i_angles)])
        return np.vstack([o, i])
    else:
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        return np.column_stack([np.cos(angles), np.sin(angles)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(x0, deadline, max_temp=0.05, min_temp=1e-4, n_steps=12):
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))

    for temp in temps:
        if time.time() >= deadline - 8:
            break

        if time.time() < deadline - 6:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        if time.time() < deadline - 4:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

    return best, best_s


def _annealed_soft_minimize_radial(x0, deadline, max_temp=0.05, min_temp=1e-4, n_steps=12):
    best = x0.copy()
    best_points = _radial_to_cartesian(best)
    best_s = _score(best_points)

    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))

    for temp in temps:
        if time.time() >= deadline - 8:
            break

        if time.time() < deadline - 6:
            try:
                res = minimize(
                    _neg_soft_score_radial,
                    best,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x
                cand_points = _radial_to_cartesian(cand)
                s = _score(cand_points)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                    best_points = cand_points
            except Exception:
                pass

        if time.time() < deadline - 4:
            try:
                res = minimize(
                    _neg_soft_score_radial,
                    best,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x
                cand_points = _radial_to_cartesian(cand)
                s = _score(cand_points)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                    best_points = cand_points
            except Exception:
                pass

    return best_points, best_s


def _polish_true_objective(x0, deadline):
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    if time.time() < deadline - 6:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Nelder-Mead',
                options={'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    return best, best_s


def _polish_true_objective_radial(x0_radial, deadline):
    best = x0_radial.copy()
    best_points = _radial_to_cartesian(best)
    best_s = _score(best_points)

    if time.time() < deadline - 6:
        try:
            res = minimize(
                _neg_score_radial,
                best,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x
            cand_points = _radial_to_cartesian(cand)
            s = _score(cand_points)
            if s > best_s:
                best_s = s
                best = cand.copy()
                best_points = cand_points
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score_radial,
                best,
                method='Nelder-Mead',
                options={'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x
            cand_points = _radial_to_cartesian(cand)
            s = _score(cand_points)
            if s > best_s:
                best_s = s
                best = cand.copy()
                best_points = cand_points
        except Exception:
            pass

    return best_points, best_s


def _polish_true_objective_support(x0_support, deadline):
    best = x0_support.copy()
    best_points = _support_to_points(best)
    best_s = _score(best_points)

    if time.time() < deadline - 6:
        try:
            res = minimize(
                _neg_score_support,
                best,
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
            )
            cand = res.x
            cand_points = _support_to_points(cand)
            s = _score(cand_points)
            if s > best_s:
                best_s = s
                best = cand.copy()
                best_points = cand_points
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score_support,
                best,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x
            cand_points = _support_to_points(cand)
            s = _score(cand_points)
            if s > best_s:
                best_s = s
                best = cand.copy()
                best_points = cand_points
        except Exception:
            pass

    return best_points, best_s


def _local_search(points, deadline, n_iter=300, sigma_start=0.02, sigma_end=0.001):
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)

    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break

        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress

        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)

        for scale in [1.0, 0.5, 0.2, 2.0]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction

            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break

    return best, best_s


def _coordinate_descent(points, deadline, n_rounds=50):
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(99)
    
    for rnd in range(n_rounds):
        if time.time() >= deadline - 5:
            break
        
        improved = False
        for idx in range(N):
            if time.time() >= deadline - 5:
                break
            
            for dim in range(2):
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    cand = best.copy()
                    cand[idx, dim] += delta
                    s = _score(cand)
                    if s > best_s + 1e-15:
                        best_s = s
                        best = cand.copy()
                        improved = True
                        break
        
        if not improved:
            break
    
    return best, best_s


def _simulated_annealing(points, deadline, T_start=0.05, T_end=0.001, n_iter=500):
    best = points.copy()
    best_s = _score(best)
    current = points.copy()
    current_s = _score(current)
    rng = np.random.RandomState(77)
    
    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        idx = rng.randint(N)
        delta = rng.randn(2) * 0.02
        cand = current.copy()
        cand[idx] += delta
        
        cand_s = _score(cand)
        
        if cand_s > current_s or rng.random() < np.exp((cand_s - current_s) / T):
            current = cand
            current_s = cand_s
            if cand_s > best_s:
                best_s = cand_s
                best = cand.copy()
    
    return best, best_s


def _de_optimize(x0, deadline, pop_size=40, max_iter=200, mutation=(0.5, 1.0), crossover=0.7):
    n = len(x0)
    rng = np.random.default_rng(7)
    
    pop = np.array([x0.copy()])
    for i in range(pop_size - 1):
        pop_i = x0 + rng.normal(0, 0.1, n)
        pop = np.vstack([pop, pop_i])
    
    fitnesses = np.array([_neg_score(p) for p in pop])
    best_idx = np.argmin(fitnesses)
    best_x = pop[best_idx].copy()
    best_f = fitnesses[best_idx]
    
    for it in range(max_iter):
        if time.time() >= deadline - 10:
            break
        
        new_pop = [pop[best_idx].copy()]
        new_fitnesses = [best_f]
        
        for i in range(pop_size - 1):
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while c == best_idx:
                c = rng.randint(pop_size)
            
            F = rng.uniform(*mutation)
            v = pop[a] + F * (pop[b] - pop[c])
            
            CR = crossover
            j_rand = rng.randint(n)
            trial = np.where(rng.random(n) < CR, v, pop[i])
            trial[j_rand] = v[j_rand]
            
            f_trial = _neg_score(trial)
            
            if f_trial <= fitnesses[i]:
                new_pop.append(trial)
                new_fitnesses.append(f_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    best_idx = len(new_pop) - 1
            else:
                new_pop.append(pop[i].copy())
                new_fitnesses.append(fitnesses[i])
        
        pop = np.array(new_pop)
        fitnesses = np.array(new_fitnesses)
    
    return best_x, -best_f


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = [
        _seed_regular(),
        _seed_two_ring(7, 0.55, np.pi / 6),
        _seed_two_ring(6, 0.5, np.pi / 5),
        _seed_two_ring(8, 0.4, np.pi / 4),
        _seed_three_ring(5, 4, 4, 0.65, 0.35, np.pi / 5, np.pi / 4),
        _seed_three_ring(6, 4, 3, 0.6, 0.3, np.pi / 3, np.pi / 6),
        _seed_quasi_regular(),
    ]

    for _ in range(8):
        seeds.append(_seed_random_ring(rng))

    # Evaluate all seeds
    for seed in seeds:
        _update_best(seed)

    # Annealed optimization from each seed (full 26D space)
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        try:
            x_opt, s_opt = _annealed_soft_minimize(
                seed.flatten(),
                deadline,
                max_temp=0.05,
                min_temp=1e-4,
                n_steps=10,
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass

    # Radial parameterization optimization (26D but structured)
    if best_points is not None and time.time() < deadline - 40:
        try:
            radial_params = _cartesian_to_radial(best_points)
            # Sort angles to maintain convexity
            order = np.argsort(radial_params[N:])
            radial_params = radial_params[order]
            # Reorder points accordingly
            rad_sorted = radial_params[:N]
            ang_sorted = radial_params[N:]
            sorted_points = np.column_stack([
                np.exp(rad_sorted) * np.cos(ang_sorted),
                np.exp(rad_sorted) * np.sin(ang_sorted)
            ])
            
            x_opt, s_opt = _annealed_soft_minimize_radial(
                radial_params,
                deadline,
                max_temp=0.05,
                min_temp=1e-4,
                n_steps=10,
            )
            _update_best(x_opt)
        except Exception:
            pass

    # Support function parameterization (13D)
    if best_points is not None and time.time() < deadline - 35:
        try:
            # Initialize support offsets from current best
            # Compute approximate support function values
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            center = best_points.mean(axis=0)
            offsets = np.dot(best_points - center, normals.T)
            
            x_opt, s_opt = _polish_true_objective_support(
                offsets,
                deadline,
            )
            _update_best(x_opt)
        except Exception:
            pass

    # Differential Evolution from best
    if best_points is not None and time.time() < deadline - 30:
        try:
            de_x, de_s = _de_optimize(best_points.flatten(), deadline, pop_size=30, max_iter=100)
            _update_best(de_x.reshape(N, 2))
        except Exception:
            pass

    # Polish the best found so far
    if best_points is not None and time.time() < deadline - 25:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    # Polish in radial space
    if best_points is not None and time.time() < deadline - 20:
        try:
            radial_params = _cartesian_to_radial(best_points)
            order = np.argsort(radial_params[N:])
            radial_params = radial_params[order]
            x_pol, s_pol = _polish_true_objective_radial(radial_params, deadline)
            _update_best(x_pol)
        except Exception:
            pass

    # Coordinate descent
    if best_points is not None and time.time() < deadline - 15:
        try:
            cd_points, cd_s = _coordinate_descent(best_points, deadline, n_rounds=30)
            _update_best(cd_points)
        except Exception:
            pass

    # Local search from best
    if best_points is not None and time.time() < deadline - 12:
        try:
            ls_points, s_ls = _local_search(best_points, deadline, n_iter=200)
            _update_best(ls_points)
        except Exception:
            pass

    # Simulated annealing
    if best_points is not None and time.time() < deadline - 10:
        try:
            sa_points, sa_s = _simulated_annealing(best_points, deadline, n_iter=300)
            _update_best(sa_points)
        except Exception:
            pass

    # Final hard-objective polish
    if best_points is not None and time.time() < deadline - 8:
        try:
            x_final, s_final = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_final.reshape(N, 2))
        except Exception:
            pass

    # More annealed restarts if time permits
    while time.time() < deadline - 8:
        seed_idx = int(rng.integers(0, len(seeds)))
        base_seed = seeds[seed_idx].copy()
        noise = rng.normal(0, 0.05, (N, 2))
        perturbed = base_seed + noise
        try:
            x_opt, s_opt = _annealed_soft_minimize(
                perturbed.flatten(),
                deadline,
                max_temp=0.03,
                min_temp=1e-4,
                n_steps=8,
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass

    # One more polish if we improved
    if best_points is not None and time.time() < deadline - 5:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
