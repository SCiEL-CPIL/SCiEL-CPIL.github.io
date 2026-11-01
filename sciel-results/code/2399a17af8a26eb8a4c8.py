"""Optimized program: 14 points in 3D minimizing (max/min pairwise dist)^2."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 14
D = 3
_DEADLINE = time.time() + 135.0  # safety margin before 150s kill

# Precompute pairwise indices
_IDX_I, _IDX_J = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_IDX_I)


def _pairwise_dists2(pts):
    """Compute squared pairwise distances for N points in D dims."""
    pts = np.asarray(pts, dtype=float)
    diffs = pts[_IDX_I] - pts[_IDX_J]
    return np.sum(diffs * diffs, axis=1)


def _score(pts):
    """Return (max_d/min_d)^2."""
    d2 = _pairwise_dists2(pts)
    d2 = np.maximum(d2, 1e-24)
    return d2.max() / d2.min()


def _logsumexp_max(x, tau):
    """Smooth max: (1/tau) * logsumexp(tau * x)."""
    m = x.max()
    return m + np.log(np.sum(np.exp(tau * (x - m)))) / tau


def _logsumexp_min(x, tau):
    """Smooth min: -(1/tau) * logsumexp(-tau * x)."""
    m = x.min()
    return m - np.log(np.sum(np.exp(-tau * (x - m)))) / tau


def _objective_and_grad(x, tau):
    """
    Objective: 0.5*(log(max_d2) - log(min_d2)) using smooth surrogates.
    x: flat array of shape (N*D,)
    Returns (obj, grad_flat)
    """
    pts = x.reshape(N, D)
    d2 = _pairwise_dists2(pts)
    d2 = np.maximum(d2, 1e-24)

    max_d2 = _logsumexp_max(d2, tau)
    min_d2 = _logsumexp_min(d2, tau)

    obj = 0.5 * (np.log(max_d2) - np.log(min_d2))

    m_max = d2.max()
    exp_max = np.exp(tau * (d2 - m_max))
    w_max = exp_max / exp_max.sum()

    m_min = d2.min()
    exp_min = np.exp(-tau * (d2 - m_min))
    w_min = exp_min / exp_min.sum()

    grad_d2 = 0.5 * (w_max / max_d2 - w_min / min_d2)

    diffs = pts[_IDX_I] - pts[_IDX_J]
    grad_pts = np.zeros_like(pts)
    np.add.at(grad_pts, _IDX_I, 2.0 * grad_d2[:, None] * diffs)
    np.add.at(grad_pts, _IDX_J, -2.0 * grad_d2[:, None] * diffs)

    return obj, grad_pts.flatten()


def _build_constraints():
    """
    Fix point 0 at origin, point 1 at (0,0,1) to remove symmetries.
    Use LinearConstraint for efficiency.
    """
    A = np.zeros((6, N * D))
    A[0, 0] = 1  # x[0] = 0
    A[1, 1] = 1  # x[1] = 0
    A[2, 2] = 1  # x[2] = 0
    A[3, 3] = 1  # x[3] = 0
    A[4, 4] = 1  # x[4] = 0
    A[5, 5] = 1  # x[5] = 1
    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    return [LinearConstraint(A, b, b)]


def _apply_symmetry(pts):
    """Translate so point 0 is at origin, rotate so point 1 is on +z axis."""
    pts = pts - pts[0]
    v = pts[1].copy()
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-14:
        return pts
    v = v / norm_v
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    c = np.dot(v, z)
    if s < 1e-12:
        if c > 0:
            return pts
        else:
            R = np.diag([1.0, -1.0, -1.0])
            return pts @ R.T
    axis = axis / s
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    theta = np.arccos(np.clip(c, -1.0, 1.0))
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return pts @ R.T


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    golden = (1 + np.sqrt(5)) / 2
    pts = np.zeros((n, 3))
    for k in range(n):
        z = 1 - 2 * (k + 0.5) / n
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * k / golden
        pts[k] = [r * np.cos(theta), r * np.sin(theta), z]
    return pts


def _icosahedron_and_center():
    """
    Generate 12 icosahedron vertices + 2 additional points.
    """
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    fib_pts = _fibonacci_sphere(14)
    result = np.vstack([vertices, fib_pts[12:14]])
    return result


def _generate_initial_points(rng, n_starts=20):
    """Generate diverse initial configurations."""
    inits = []
    
    # Start 0: Fibonacci sphere
    pts = _fibonacci_sphere(N)
    pts = _apply_symmetry(pts)
    inits.append(pts.flatten())
    
    # Start 1: Icosahedron + 2 extra
    pts = _icosahedron_and_center()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts = _apply_symmetry(pts)
    inits.append(pts.flatten())
    
    # Starts 2-19: Random with symmetry applied
    for i in range(n_starts - 2):
        pts = rng.randn(N, D) * 0.8
        pts[0] = 0.0
        pts[1] = [0.0, 0.0, 1.0]
        inits.append(pts.flatten())
    
    return inits


def _refine(pts, constraints, tau, maxiter, ftol):
    """Run SLSQP refinement on a set of points."""
    x0 = pts.flatten()
    res = minimize(
        _objective_and_grad,
        x0,
        args=(tau,),
        method='SLSQP',
        jac=True,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': ftol},
    )
    if res.success and res.x is not None:
        return res.x.reshape(N, D)
    return None


def construct():
    global _DEADLINE
    start_time = time.time()
    rng = np.random.RandomState(42)

    best_score = np.inf
    best_pts = None

    constraints = _build_constraints()

    # Phase 1: Multi-start SLSQP with moderate tau
    n_starts = 15
    tau_initial = 500.0

    inits = _generate_initial_points(rng, n_starts)

    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 30:
            break

        try:
            res = minimize(
                _objective_and_grad,
                x0,
                args=(tau_initial,),
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12},
            )
            if res.success and res.x is not None:
                pts = res.x.reshape(N, D)
                s = _score(pts)
                if s < best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    # Phase 2: Polish best candidate with high tau and more iterations
    if best_pts is not None:
        for tau_polish in [2000.0, 5000.0]:
            if time.time() > _DEADLINE - 10:
                break
            try:
                x0 = best_pts.flatten()
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(tau_polish,),
                    method='SLSQP',
                    jac=True,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-14},
                )
                if res.success and res.x is not None:
                    pts = res.x.reshape(N, D)
                    s = _score(pts)
                    if s < best_score:
                        best_score = s
                        best_pts = pts.copy()
            except Exception:
                pass

    # Phase 3: If time remains, do a few more random restarts from best
    if best_pts is not None and time.time() < _DEADLINE - 15:
        for _ in range(5):
            if time.time() > _DEADLINE - 5:
                break
            try:
                noise = rng.randn(N, D) * 0.05
                x0 = (best_pts + noise).flatten()
                x0[:3] = 0.0
                x0[3:6] = [0.0, 0.0, 1.0]
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(5000.0,),
                    method='SLSQP',
                    jac=True,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-13},
                )
                if res.success and res.x is not None:
                    pts = res.x.reshape(N, D)
                    s = _score(pts)
                    if s < best_score:
                        best_score = s
                        best_pts = pts.copy()
            except Exception:
                pass

    if best_pts is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)

    return best_pts
