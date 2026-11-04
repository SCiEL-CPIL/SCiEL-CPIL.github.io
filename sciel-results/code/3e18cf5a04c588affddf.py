"""Optimized program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area.

Strategy:
- Angular parameterization: points = (r_i * cos(theta_i), r_i * sin(theta_i)) with sorted theta_i
  This structurally enforces convex position (all points on convex hull when r_i > 0 and angles sorted).
- Optimize the ratio min_triangle_area / hull_area directly using smooth surrogates.
- Multi-phase: coarse SA in angular space → L-BFGS-B polish → targeted local search.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 13
TIME_LIMIT = 140.0

_TRIS = np.array(list(combinations(range(N), 3)))
_N_TRIS = _TRIS.shape[0]


def _triangle_areas(pts):
    """Vectorized computation of all C(N,3) triangle areas."""
    i, j, k = _TRIS[:, 0], _TRIS[:, 1], _TRIS[:, 2]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0


def _soft_min_value(areas, beta):
    """Soft-min via log-sum-exp."""
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - log_sum / beta


def _angular_to_points(angles, radii):
    """Convert sorted angles and radii to 2D points."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _angular_score(angles, radii, beta=None):
    """Score from angular parameterization. If beta given, use soft-min surrogate."""
    pts = _angular_to_points(angles, radii)
    if beta is not None:
        areas = _triangle_areas(pts)
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        soft_min_val = _soft_min_value(areas, beta)
        return soft_min_val / hull_area
    else:
        return _score(pts)


def _neg_angular_soft(angles, radii, beta):
    """Negative soft score for minimization."""
    return -_angular_score(angles, radii, beta)


def _neg_angular_exact(angles, radii):
    """Negative exact score for minimization."""
    return -_angular_score(angles, radii)


def _regular_angular():
    """Regular 13-gon in angular parameterization."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    return angles, radii


def _generate_angular_inits():
    """Generate diverse initial configurations in angular space."""
    rng = np.random.default_rng(42)
    inits = []

    # 1. Regular polygon
    a, r = _regular_angular()
    inits.append((a, r))

    # 2. Perturbed regular polygon angles
    for noise in [0.05, 0.1, 0.2, 0.3]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, noise, N)
        a = np.sort(a)
        r = np.ones(N) + rng.normal(0, 0.1, N)
        r = np.clip(r, 0.3, 2.0)
        inits.append((a, r))

    # 3. Halton sequence mapped to angular
    def _halton(n, base):
        result = []
        for i in range(1, n + 1):
            x = 0.0
            f = 1.0
            i_temp = i
            while i_temp > 0:
                f /= base
                x += f * (i_temp % base)
                i_temp //= base
            result.append(x)
        return np.array(result)

    hx = _halton(N, 2)
    hy = _halton(N, 3)
    # Map to angles and radii
    angles_h = np.sort(np.arctan2(hy, hx))
    radii_h = np.sqrt(hx**2 + hy**2)
    radii_h = np.clip(radii_h, 0.2, 1.5)
    inits.append((angles_h, radii_h))

    # 4. Concentric ring: alternating inner/outer
    angles_c = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_c = np.where(np.arange(N) % 2 == 0, 1.0, 0.5)
    inits.append((angles_c, radii_c))

    # 5. Random sorted angles, uniform radii
    for seed in [1, 2, 3]:
        rng_s = np.random.default_rng(seed)
        a = np.sort(rng_s.uniform(0, 2 * np.pi, N))
        r = rng_s.uniform(0.5, 1.5, N)
        inits.append((a, r))

    # 6. Grid-like angular: some clustered, some spread
    a6 = np.sort(np.concatenate([
        np.linspace(0, np.pi/3, 4),
        np.linspace(np.pi/3, 2*np.pi/3, 3),
        np.linspace(2*np.pi/3, np.pi, 3),
        np.linspace(np.pi, 5*np.pi/3, 3)
    ]))
    r6 = np.ones(N)
    inits.append((a6, r6))

    # 7. Near-regular with one point pulled inward
    a7, r7 = _regular_angular()
    r7[0] = 0.4
    inits.append((a7, r7))

    # 8. Two-cluster: 7 and 6 points
    a8 = np.sort(np.concatenate([
        np.linspace(0, np.pi, 7, endpoint=False),
        np.linspace(np.pi, 2*np.pi, 6, endpoint=False)
    ]))
    r8 = np.ones(N)
    inits.append((a8, r8))

    return inits


def _simulated_annealing_angular(start_time, time_budget=50.0, seed=42):
    """
    Simulated Annealing in angular parameterization (26D: 13 angles + 13 radii).
    Angles are kept sorted by re-sorting after each perturbation.
    """
    deadline = start_time + time_budget
    rng = np.random.default_rng(seed)

    angles, radii = _regular_angular()
    current_score = _angular_score(angles, radii)

    best_angles = angles.copy()
    best_radii = radii.copy()
    best_score = current_score

    n_steps = 80000
    T_start = 0.15
    T_end = 0.0005

    for i in range(n_steps):
        if time.time() > deadline - 3:
            break

        T = T_start * (T_end / T_start) ** (i / float(n_steps))

        # Choose which parameter to perturb
        param_type = rng.random()
        idx = rng.integers(0, N)

        if param_type < 0.5:
            # Perturb angle
            new_angles = angles.copy()
            new_angles[idx] += rng.normal(0, 0.05 + 0.1 * T / T_start)
            new_angles = np.sort(new_angles)
            new_radii = radii
        else:
            # Perturb radius
            new_radii = radii.copy()
            new_radii[idx] += rng.normal(0, 0.05 + 0.15 * T / T_start)
            new_radii[idx] = max(0.05, new_radii[idx])
            new_angles = angles

        new_score = _angular_score(new_angles, new_radii)
        delta = new_score - current_score

        if delta > 0 or (T > 1e-8 and rng.random() < np.exp(delta / T)):
            angles = new_angles
            radii = new_radii
            current_score = new_score

            if current_score > best_score:
                best_score = current_score
                best_angles = angles.copy()
                best_radii = radii.copy()

    return best_angles, best_radii, best_score


def _lbfgs_angular_polish(angles, radii, start_time, time_budget=15.0, betas=(30, 80, 200, 500)):
    """L-BFGS-B polish in angular space with increasing softness."""
    deadline = start_time + time_budget
    best_angles = angles.copy()
    best_radii = radii.copy()
    best_score = _angular_score(best_angles, best_radii)

    for beta in betas:
        if time.time() > deadline - 2:
            break

        def obj(x):
            a = np.sort(x[:N])
            r = np.clip(x[N:], 0.05, 3.0)
            return -_angular_score(a, r, beta)

        x0 = np.concatenate([angles, radii])

        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(0, 2*np.pi)] * N + [(0.05, 3.0)] * N,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_angles = np.sort(res.x[:N])
            new_radii = np.clip(res.x[N:], 0.05, 3.0)
            s = _angular_score(new_angles, new_radii)
            if s > best_score:
                best_score = s
                best_angles = new_angles.copy()
                best_radii = new_radii.copy()
        except Exception:
            break

    return best_angles, best_radii, best_score


def _nelder_mead_angular(angles, radii, start_time, time_budget=10.0):
    """Nelder-Mead on exact score in angular space."""
    deadline = start_time + time_budget

    def obj(x):
        a = np.sort(x[:N])
        r = np.clip(x[N:], 0.05, 3.0)
        return -_angular_score(a, r)

    x0 = np.concatenate([angles, radii])

    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_angles = np.sort(res.x[:N])
        new_radii = np.clip(res.x[N:], 0.05, 3.0)
        s = _angular_score(new_angles, new_radii)
        return new_angles, new_radii, s
    except Exception:
        return angles, radii, _angular_score(angles, radii)


def _targeted_point_polish(angles, radii, start_time, time_budget=15.0):
    """Optimize individual points (angle+radius) using Nelder-Mead on 2D subproblem."""
    deadline = start_time + time_budget
    best_angles = angles.copy()
    best_radii = radii.copy()
    best_score = _angular_score(best_angles, best_radii)

    pts = _angular_to_points(best_angles, best_radii)
    areas = _triangle_areas(pts)
    k_worst = min(8, len(areas))
    worst_indices = np.argsort(areas)[:k_worst]

    # Get point indices involved in worst triangles
    involved = set()
    for wi in worst_indices:
        i, j, k = _TRIS[wi]
        involved.add(i)
        involved.add(j)
        involved.add(k)

    for pt_idx in sorted(involved):
        if time.time() > deadline - 2:
            break

        def obj_2d(x, idx=pt_idx):
            a = best_angles.copy()
            r = best_radii.copy()
            a[idx] = x[0]
            r[idx] = max(0.05, x[1])
            a = np.sort(a)
            # After sorting, the point may have moved to a different index.
            # We need to track which sorted position this point ends up in.
            # For simplicity, just evaluate with the unsorted assignment and sort.
            # Actually, let's just place it and sort the whole thing.
            pts_trial = _angular_to_points(a, r)
            return -_score(pts_trial)

        x0 = np.array([best_angles[pt_idx], best_radii[pt_idx]])

        try:
            res = minimize(
                obj_2d, x0, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            # Apply: set angle and radius, then sort
            new_a = best_angles.copy()
            new_r = best_radii.copy()
            new_a[pt_idx] = res.x[0]
            new_r[pt_idx] = max(0.05, res.x[1])
            # Sort by angle
            order = np.argsort(new_a)
            new_a = new_a[order]
            new_r = new_r[order]

            s = _angular_score(new_a, new_r)
            if s > best_score:
                best_score = s
                best_angles = new_a.copy()
                best_radii = new_r.copy()
        except Exception:
            pass

    return best_angles, best_radii, best_score


def _de_angular_search(start_time, time_budget=20.0):
    """Differential evolution in angular space (26D)."""
    deadline = start_time + time_budget

    def de_obj(x):
        a = np.sort(x[:N])
        r = np.clip(x[N:], 0.05, 2.5)
        return -_angular_score(a, r, beta=40)

    try:
        res = differential_evolution(
            de_obj,
            bounds=[(0, 2*np.pi)] * N + [(0.1, 2.0)] * N,
            seed=42,
            maxiter=40,
            popsize=12,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            init='latinhypercube',
            updating='immediate',
            callback=lambda xk, conv: time.time() > deadline - 3
        )
        a = np.sort(res.x[:N])
        r = np.clip(res.x[N:], 0.05, 2.5)
        s = _angular_score(a, r)
        return a, r, s
    except Exception:
        return None, None, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    inits = _generate_angular_inits()

    best_angles = None
    best_radii = None
    best_score = -1

    # Evaluate all initializations
    for a, r in inits:
        if time.time() > deadline - 60:
            break
        s = _angular_score(a, r)
        if s > best_score:
            best_score = s
            best_angles = a.copy()
            best_radii = r.copy()

    if best_angles is None:
        best_angles, best_radii = _regular_angular()
        best_score = _angular_score(best_angles, best_radii)

    # Phase 1: Simulated Annealing (main global search)
    remaining = deadline - 40 - time.time()
    if remaining > 15:
        sa_a, sa_r, sa_s = _simulated_annealing_angular(
            start_time, time_budget=min(remaining, 45.0), seed=42
        )
        if sa_s > best_score:
            best_score = sa_s
            best_angles = sa_a
            best_radii = sa_r

        # Second SA run with different seed
        remaining2 = deadline - 35 - time.time()
        if remaining2 > 15:
            sa_a2, sa_r2, sa_s2 = _simulated_annealing_angular(
                start_time, time_budget=min(remaining2, 30.0), seed=123
            )
            if sa_s2 > best_score:
                best_score = sa_s2
                best_angles = sa_a2
                best_radii = sa_r2

    # Phase 2: L-BFGS-B polish on best + top inits
    candidates = [(best_angles, best_radii)]
    scored_inits = []
    for a, r in inits:
        s = _angular_score(a, r)
        scored_inits.append((s, a, r))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    for s, a, r in scored_inits[:4]:
        if s > best_score * 0.5:
            candidates.append((a, r))

    for a_c, r_c in candidates:
        if time.time() > deadline - 25:
            break
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        pa, pr, ps = _lbfgs_angular_polish(
            a_c, r_c, start_time,
            time_budget=min(remaining, 12.0),
            betas=(30, 80, 200, 500)
        )
        if ps > best_score:
            best_score = ps
            best_angles = pa
            best_radii = pr

    # Phase 3: Differential evolution (if time allows)
    remaining = deadline - 20 - time.time()
    if remaining > 12:
        de_a, de_r, de_s = _de_angular_search(start_time, time_budget=min(remaining, 15.0))
        if de_a is not None and de_s > best_score:
            best_score = de_s
            best_angles = de_a
            best_radii = de_r

    # Phase 4: Targeted point polish
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        pa, pr, ps = _targeted_point_polish(
            best_angles, best_radii, start_time,
            time_budget=min(remaining, 10.0)
        )
        if ps > best_score:
            best_score = ps
            best_angles = pa
            best_radii = pr

    # Phase 5: Final Nelder-Mead polish
    remaining = deadline - 3 - time.time()
    if remaining > 3:
        pa, pr, ps = _nelder_mead_angular(
            best_angles, best_radii, start_time,
            time_budget=min(remaining, 5.0)
        )
        if ps > best_score:
            best_score = ps
            best_angles = pa
            best_radii = pr

    # Final: one more L-BFGS-B with high beta
    remaining = deadline - 1 - time.time()
    if remaining > 2:
        pa, pr, ps = _lbfgs_angular_polish(
            best_angles, best_radii, start_time,
            time_budget=min(remaining, 3.0),
            betas=(300, 800)
        )
        if ps > best_score:
            best_score = ps
            best_angles = pa
            best_radii = pr

    # Convert to Cartesian
    if time.time() > deadline - 0.5:
        pts = _angular_to_points(best_angles, best_radii)
        return pts

    pts = _angular_to_points(best_angles, best_radii)
    final_score = _score(pts)
    if final_score > best_score:
        best_score = final_score

    return pts
