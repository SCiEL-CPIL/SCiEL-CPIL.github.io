"""
Logarithmic energy minimization of 150 points on the unit sphere.

Strategy:
  - Multi-start local optimization (Fibonacci spiral, geodesic dome, random).
  - Basin hopping with Riemannian (exponential-map / rotation) perturbations
    that respect the sphere geometry and converge faster than Euclidean
    projection perturbations.
  - Riemannian L-BFGS polish: optimize in tangent space (a change of
    variables that removes the sphere constraint), then map back.
  - Final high-precision polish pass.
  - Additional: structured spherical grid parameterization for better
    initial configurations, and Barzilai-Borwein adaptive step sizes for
    gradient-based refinement.

All time is bounded by an explicit wall-clock deadline with a safety margin.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Energy and gradient
# ---------------------------------------------------------------------------
def _energy_and_grad(pts):
    """
    Logarithmic energy and tangent-space gradient.
    pts: (N, 3) unit vectors.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i (Cartesian) = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2, projected to tangent.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts - pts * row_sums[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute only the energy (fsum for precision)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


# ---------------------------------------------------------------------------
# Riemannian (exponential map) utilities
# ---------------------------------------------------------------------------
def _exp_map_tangent(pts, tangent):
    """
    Exponential map on the sphere: move pts along tangent vectors.
    For a unit vector x and tangent v, the exponential map is
        exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    which is equivalent to rotating x by angle |v| about the axis v/|v|.
    This respects the sphere geometry exactly.
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    # Where norm ~ 0, c ~ 1, s ~ 0, so result ~ x
    out = c * pts + s * u
    return _project_to_sphere(out)


def _log_map_tangent(pts, target):
    """
    Logarithmic map: tangent vector at pts that maps to target on the sphere.
    v = theta * (target - cos(theta) x) / sin(theta), theta = arccos(x . target).
    """
    cos_theta = np.sum(pts * target, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    # Avoid division by zero
    safe_sin = np.where(sin_theta > 1e-12, sin_theta, 1.0)
    v = (theta / safe_sin) * (target - cos_theta * pts)
    # Zero out where theta ~ 0
    v = np.where((theta > 1e-12)[:, np.newaxis], v, 0.0)
    return v


# ---------------------------------------------------------------------------
# Riemannian L-BFGS polish (optimizes in tangent space)
# ---------------------------------------------------------------------------
def _riemannian_lbfgs_polish(pts, deadline, maxiter=400, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates.
    We represent the configuration as a flat tangent vector t (size 3N) and
    define f(t) = energy(exp(pts, t)), grad_f(t) = grad of energy at exp(pts,t)
    pulled back (which for the sphere exponential map with the metric is just
    the tangent gradient, since the exponential map is an isometry in the
    infinitesimal sense for our purposes here).

    We use a trust-region-like approach: start from t=0 (current config),
    optimize t, and if it improves, set pts = exp(pts, t_best) and repeat.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(t):
        # t is flat tangent vector (3N,). Interpret as (N,3) tangent.
        tv = t.reshape(N, 3)
        # Make sure tangent is actually tangent (subtract radial part)
        dot = np.sum(tv * pts, axis=1, keepdims=True)
        tv = tv - dot * pts
        new_pts = _exp_map_tangent(pts, tv)
        e, g = _energy_and_grad(new_pts)
        # Gradient of f w.r.t. t: since exp is local isometry, g (tangent) is
        # the gradient in t-space directly.
        return e, g.flatten()

    t0 = np.zeros(3 * N)
    try:
        res = minimize(fun, t0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        if time.time() < deadline:
            tv = res.x.reshape(N, 3)
            dot = np.sum(tv * pts, axis=1, keepdims=True)
            tv = tv - dot * pts
            new_pts = _exp_map_tangent(pts, tv)
            new_energy = _energy_only(new_pts)
            if new_energy < best_energy and np.all(np.isfinite(new_pts)):
                best_energy = new_energy
                best_pts = new_pts.copy()
    except Exception:
        pass

    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Cartesian L-BFGS polish (fallback / final)
# ---------------------------------------------------------------------------
def _lbfgs_polish(pts, deadline, maxiter=500, ftol=1e-16, gtol=1e-14):
    """L-BFGS-B polish using flat Cartesian coordinates with projection."""
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_res = _energy_only(p_res)
        if e_res < best_energy and np.all(np.isfinite(p_res)):
            best_energy = e_res
            best_pts = p_res.copy()
    except Exception:
        pass
    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Barzilai-Borwein adaptive gradient step in tangent space
# ---------------------------------------------------------------------------
def _bb_gradient_descent(pts, deadline, maxiter=2000):
    """
    Barzilai-Borwein step-size gradient descent on the sphere.
    Uses the exponential map for steps and BB step sizes for fast
    convergence near minima.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    prev_grad = None
    prev_tangent = None
    alpha = 1e-3  # initial step size

    for it in range(maxiter):
        if time.time() > deadline - 0.5:
            break

        e, grad = _energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()

        # Check convergence
        if np.linalg.norm(grad) < 1e-14:
            break

        # Compute BB step size
        if prev_grad is not None and prev_tangent is not None:
            y = grad - prev_grad
            s = -prev_tangent  # previous step direction (negative gradient)
            ys = np.sum(y * s)
            gy = np.sum(grad * y)
            gs = np.sum(grad * s)
            if abs(ys) > 1e-15:
                alpha_y = ys / gy if abs(gy) > 1e-15 else 1e-3
                alpha_s = -gs / ys if abs(ys) > 1e-15 else 1e-3
                # Alternate between alpha_y and alpha_s
                if it % 2 == 0:
                    alpha = alpha_y
                else:
                    alpha = alpha_s
                alpha = np.clip(alpha, 1e-6, 1.0)

        # Gradient step in tangent space
        tangent = -alpha * grad
        new_pts = _exp_map_tangent(pts, tangent)
        new_pts = _project_to_sphere(new_pts)

        new_e, new_grad = _energy_and_grad(new_pts)

        # Line search if needed (simple backtracking)
        if new_e > e + 1e-10:
            # Try smaller step
            for factor in [0.5, 0.25, 0.1]:
                small_tangent = -alpha * factor * grad
                small_pts = _exp_map_tangent(pts, small_tangent)
                small_pts = _project_to_sphere(small_pts)
                small_e, _ = _energy_and_grad(small_pts)
                if small_e < e:
                    new_pts = small_pts
                    new_e = small_e
                    new_grad = _energy_and_grad(small_pts)[1]
                    break
            else:
                break

        prev_tangent = -grad
        prev_grad = grad
        pts = new_pts

    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Structured spherical grid initialization
# ---------------------------------------------------------------------------
def _spherical_grid_init(n, seed=0):
    """
    Generate points on a structured spherical grid using latitude/longitude
    with non-uniform spacing to avoid clustering at poles.
    """
    rng = np.random.RandomState(seed)
    # Use a modified Fibonacci spiral with slight perturbation for diversity
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    # Add slight non-uniformity
    z = z + 0.02 * np.sin(i * 0.7)
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.randn(n) * 0.01
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------
def _fibonacci_spiral(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron.
    Provides a different structure than the Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    edges = [(0, 11), (0, 5), (0, 1), (0, 7), (0, 10), (1, 5), (5, 11),
             (11, 10), (10, 7), (7, 1), (1, 6), (6, 10), (2, 11), (2, 5),
             (2, 9), (2, 4), (2, 8), (3, 5), (3, 11), (3, 4), (3, 9),
             (3, 6), (4, 9), (5, 4), (11, 4), (10, 7), (6, 1), (8, 4),
             (9, 8), (8, 5), (6, 9), (7, 2), (8, 3)]

    # Subdivide until we have enough points
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
        # Rebuild edges: each old edge (i,j) becomes (i, mid) and (mid, j)
        # mid index = len(verts) - len(new_verts) + k
        new_edges = []
        base = len(verts) - len(new_verts)
        for k, (i, j) in enumerate(edges):
            m = base + k
            new_edges.append((i, m))
            new_edges.append((m, j))
        edges = new_edges

    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    return _project_to_sphere(verts)


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


# ---------------------------------------------------------------------------
# Basin hopping with Riemannian perturbations
# ---------------------------------------------------------------------------
def _basin_hopping(pts0, deadline, rng, max_iter=200):
    """
    Basin Hopping with Riemannian (exponential-map) perturbations.
    1. Perturb a random subset of points using the exponential map.
    2. Polish with Riemannian L-BFGS.
    3. Accept if better (or Metropolis accept).
    """
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy

    T = 0.4
    T_min = 1e-7
    cooling = 0.96

    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break

        # Perturb a random subset using the exponential map
        n_perturb = rng.randint(5, 18)
        idxs = rng.choice(N, n_perturb, replace=False)

        tangent = np.zeros_like(current)
        for idx in idxs:
            rand_vec = rng.randn(3)
            tv = rand_vec - np.dot(rand_vec, current[idx]) * current[idx]
            norm = np.linalg.norm(tv)
            if norm < 1e-15:
                continue
            tv /= norm
            # Random angle scaled by temperature
            angle = T * abs(rng.randn())
            tangent[idx] = angle * tv

        new_pts = _exp_map_tangent(current, tangent)
        new_pts = _project_to_sphere(new_pts)

        # Polish with Riemannian L-BFGS
        new_pts, new_energy = _riemannian_lbfgs_polish(new_pts, deadline, maxiter=80)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > T_min and rng.rand() < math.exp(-delta_e / T)):
            current = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()

        T *= cooling
        if T < T_min:
            break

    return best, best_energy


# ---------------------------------------------------------------------------
# Multi-phase optimization combining L-BFGS and BB gradient descent
# ---------------------------------------------------------------------------
def _multi_phase_polish(pts, deadline, maxiter_lbfgs=300, maxiter_bb=500):
    """
    Combine L-BFGS and BB gradient descent for robust convergence.
    Phase 1: L-BFGS for fast initial convergence.
    Phase 2: BB gradient descent for fine-tuning.
    Phase 3: L-BFGS again for final polish.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    # Phase 1: L-BFGS
    t1 = time.time()
    if t1 < deadline - 2.0:
        pts, e1 = _riemannian_lbfgs_polish(pts, deadline - 1.0, maxiter=maxiter_lbfgs)
        if e1 < best_energy:
            best_energy = e1
            best_pts = pts.copy()

    # Phase 2: BB gradient descent
    t2 = time.time()
    if t2 < deadline - 1.0:
        pts, e2 = _bb_gradient_descent(pts, deadline - 0.5, maxiter=maxiter_bb)
        if e2 < best_energy:
            best_energy = e2
            best_pts = pts.copy()

    # Phase 3: Final L-BFGS polish
    t3 = time.time()
    if t3 < deadline - 0.5:
        pts, e3 = _riemannian_lbfgs_polish(pts, deadline - 0.1, maxiter=200, ftol=1e-17, gtol=1e-14)
        if e3 < best_energy:
            best_energy = e3
            best_pts = pts.copy()

    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    total_time_budget = 145.0  # 5s safety margin before 150s kill
    deadline_global = start_time + total_time_budget

    rng = np.random.RandomState(777)

    # Initial configurations - include structured grid for diversity
    starts = [
        _fibonacci_spiral(N),
        _geodesic_dome(N, seed=0),
        _spherical_grid_init(N, seed=0),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
    ]

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Local optimization from multiple starts (~25% of time)
    phase1_deadline = start_time + 0.25 * total_time_budget
    for pts0 in starts:
        if time.time() > phase1_deadline:
            break
        pts_opt, e = _riemannian_lbfgs_polish(pts0, phase1_deadline, maxiter=150)
        if e < best_energy:
            best_energy = e
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: Basin hopping (~50% of time)
    phase2_deadline = start_time + 0.75 * total_time_budget
    bh_pts, bh_energy = _basin_hopping(best_pts, phase2_deadline, rng, max_iter=200)
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_pts = bh_pts.copy()

    # Phase 3: Final multi-phase high-precision polish (~25% of time)
    if time.time() < deadline_global - 2.0:
        best_pts, e = _multi_phase_polish(best_pts, deadline_global - 1.0,
                                          maxiter_lbfgs=400, maxiter_bb=300)
        if e < best_energy:
            best_energy = e
            best_pts = best_pts.copy()

    # Safety check
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)

    best_pts = _project_to_sphere(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
