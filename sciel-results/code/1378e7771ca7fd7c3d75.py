"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 135.0  # safety margin before the 150 s wall-clock kill


def _to_sphere(u):
    """Map box variable u (|u|<=1) to unit vector via x = 2*arcsin(u/2)."""
    # u is 1D of size 3*N
    u = u.reshape(-1, 3)
    x = 2.0 * np.arcsin(np.clip(u / 2.0, -1.0, 1.0))
    return np.sin(x).reshape(-1)  # unit vector components flattened


def _covering_radius_softmax(P, beta):
    """Smooth surrogate: soft-max of angular distances from P to its own points."""
    # P: (N,3) unit vectors. Compute angular distance from each point to nearest.
    P = P.reshape(-1, 3)
    dots = np.clip(P @ P.T, -1.0, 1.0)
    D = np.arccos(dots)  # (N,N) angular distances
    np.fill_diagonal(D, 0.0)
    # d_i = min_{j != i} D[i,j]
    d = D.min(axis=1)  # distance from each point to its nearest neighbor
    # soft-max of d
    m = d.max()
    s = m + np.log(np.sum(np.exp(beta * (d - m)))) / beta
    return s


def _objective(u, beta):
    P = _to_sphere(u)
    return _covering_radius_softmax(P, beta)


def _true_covering_radius(P):
    """Exact covering radius via convex hull circumradii (for final scoring)."""
    from scipy.spatial import ConvexHull
    P = P.reshape(-1, 3)
    try:
        hull = ConvexHull(P)
    except Exception:
        return np.pi  # fallback
    max_r = 0.0
    for simplex in hull.simplices:
        pts = P[simplex]  # (d+1, 3)
        # circumradius of the spherical simplex
        M = np.column_stack([pts, -np.ones(len(pts))])
        U, s, Vt = np.linalg.svd(M)
        null_vec = Vt[-1]
        c = null_vec[:3]
        r0 = null_vec[3]
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            continue
        c = c / norm_c
        r = np.arccos(np.clip(r0, -1.0, 1.0))
        max_r = max(max_r, r)
    return max_r


def construct():
    t0 = time.time()
    best_P = None
    best_score = np.pi

    def _elapsed():
        return time.time() - t0

    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    P0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to u-space
    # P = sin(x) where x = 2*arcsin(u/2) => u = 2*sin(x/2) where x = arcsin(P)
    # Note: arcsin(P) is component-wise.
    P0_clipped = np.clip(P0, -1.0, 1.0)
    x0 = np.arcsin(P0_clipped)
    u0 = 2.0 * np.sin(x0 / 2.0)
    u0 = u0.flatten() # Ensure 1D

    # Evaluate initial
    P = P0
    cr = _true_covering_radius(P)
    best_score = cr
    best_P = P.copy()

    # Multi-stage optimization
    # Higher betas for sharper focus on the max-min distance
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    maxiter_per_stage = 100

    for beta in betas:
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Use L-BFGS-B with numerical gradient (jac=None)
        # This is safer than a custom python-loop gradient which was too slow
        result = minimize(
            _objective,
            u0,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-1.0, 1.0)] * (N * 3),
            options={'maxiter': maxiter_per_stage, 'ftol': 1e-12, 'gtol': 1e-10},
        )
        u_opt = np.clip(result.x, -1.0, 1.0)
        P_opt = _to_sphere(u_opt).reshape(-1, 3)
        cr = _true_covering_radius(P_opt)
        if cr < best_score:
            best_score = cr
            best_P = P_opt.copy()
            u0 = u_opt
        else:
            # Check if the surrogate improved even if true CR didn't (might be stuck in local min of true CR but better surrogate)
            if result.fun < _objective(u0, beta):
                u0 = u_opt

    # Final polish
    if _elapsed() < _DEADLINE - 10:
        P = best_P.copy()
        # Use a few iterations of local search on true CR
        for _ in range(20):
            if _elapsed() > _DEADLINE - 10:
                break
            improved = False
            for i in range(N):
                best_local = P[i].copy()
                best_local_cr = best_score
                # Try random perturbations
                for _ in range(10):
                    rand_dir = np.random.randn(3)
                    rand_dir -= np.dot(rand_dir, P[i]) * P[i]
                    norm_dir = np.linalg.norm(rand_dir)
                    if norm_dir < 1e-10:
                        continue
                    rand_dir /= norm_dir
                    for step in [0.02, 0.01, 0.005]:
                        P_new = P.copy()
                        P_new[i] = P[i] + step * rand_dir
                        P_new[i] /= np.linalg.norm(P_new[i])
                        cr = _true_covering_radius(P_new)
                        if cr < best_local_cr - 1e-12:
                            best_local_cr = cr
                            best_local = P_new[i].copy()
                            improved = True
                P[i] = best_local
            if not improved:
                break
            cr = _true_covering_radius(P)
            if cr < best_score:
                best_score = cr
                best_P = P.copy()

    return best_P
# EVOLVE-BLOCK-END
