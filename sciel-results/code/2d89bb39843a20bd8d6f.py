"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 135.0  # safety margin before the 150 s wall-clock kill


def _to_sphere(u):
    """Map box variable u (|u|<=1) to unit vector via x = 2*arcsin(u/2)."""
    x = 2.0 * np.arcsin(np.clip(u / 2.0, -1.0, 1.0))
    return np.sin(x)  # unit vector components


def _covering_radius_softmax(P, beta):
    """Smooth surrogate: soft-max of angular distances from P to its own points."""
    # P: (N,3) unit vectors. Compute angular distance from each point to nearest.
    # dist(i,j) = arccos(P_i . P_j)
    dots = np.clip(P @ P.T, -1.0, 1.0)
    D = np.arccos(dots)  # (N,N) angular distances
    np.fill_diagonal(D, 0.0)
    # For each row i, nearest is min over j!=i; but for covering radius we want
    # the max over sphere points. Use the points themselves as a proxy + facets.
    # Actually covering radius = max circumradius of facets. Use soft-max over
    # all pairwise distances as a smooth upper-bound proxy, then refine.
    # Better: for each point i, the farthest point on the sphere from the set is
    # at the circumcenter of the facet. Approximate by max over the N points of
    # (distance to nearest other point) is a lower bound. We'll use the true
    # covering radius via convex hull in the final eval, but for the smooth
    # objective use soft-max of the "max distance to nearest neighbor" which
    # correlates well.
    # d_i = min_{j != i} D[i,j]
    d = D.min(axis=1)  # distance from each point to its nearest neighbor
    # soft-max of d
    m = d.max()
    s = m + np.log(np.sum(np.exp(beta * (d - m)))) / beta
    return s


def _objective(u, beta):
    P = _to_sphere(u)
    return _covering_radius_softmax(P, beta)


def _grad(u, beta):
    """Numerical gradient (central differences) of the soft-max objective."""
    P = _to_sphere(u)
    # analytic-ish: use finite differences for reliability
    eps = 1e-6
    g = np.zeros_like(u)
    for k in range(u.size):
        up = u.copy(); up[k] += eps
        um = u.copy(); um[k] -= eps
        g[k] = (_objective(up, beta) - _objective(um, beta)) / (2 * eps)
    return g


def _true_covering_radius(P):
    """Exact covering radius via convex hull circumradii (for final scoring)."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(P)
    except Exception:
        return np.pi  # fallback
    max_r = 0.0
    for simplex in hull.simplices:
        pts = P[simplex]  # (d+1, 3)
        # circumradius of the spherical simplex: solve for center c on unit sphere
        # such that |c - p_i| = r for all i, with c = sum w_i p_i / |sum w_i p_i|
        # Use the formula: circumradius = arccos( (sum of pairwise dot products of adjacent) / (2*prod of sines) )
        # Simpler: find c minimizing max_i |c - p_i| on the sphere.
        # For a triangle on the sphere, circumradius via:
        #   cos(circumradius) = (a^2 + b^2 + c^2 - 2 a b c cos(A)) / (2 (a^2 b^2 + b^2 c^2 + c^2 a^2 - a^2 b^2 c^2))
        # where a,b,c are side lengths (angular) and A is the angle opposite a.
        # Use a robust numerical approach: least squares for the center.
        # c = M^+ 1 / |M^+ 1| where M = pts, solve M c = 1 (c on sphere, |c|=1, c.p_i = cos(r))
        # Actually: c . p_i = cos(r) for all i. So M c = cos(r) * 1.
        # Eliminate cos(r): M c = r0 * 1 where r0 = cos(r). So c is in the null space direction.
        # Solve: find c, r0 such that M c = r0 * ones, |c| = 1.
        # This is: (M - r0 * ones * M^T M^+ ...) — use the standard formula.
        # For a simplex of d+1 points in R^3, the circumcenter is:
        #   c = (M^T M)^{-1} M^T (M^T M)^{-1} 1 / |(M^T M)^{-1} M^T (M^T M)^{-1} 1|  -- not quite.
        # Use: the center c satisfies M c = cos(r) * 1. Let A = M (shape (d+1, 3)).
        # We want A c = r0 * 1, |c| = 1.
        # From A c = r0 * 1: c = A^+ (r0 * 1) + null(A) * z. Since rank(A) <= 3 and d+1 >= 2,
        # for a non-degenerate simplex in R^3, rank(A) = 3 (if d+1 >= 3) or 2 (if d+1 = 2).
        # For d=2 (triangle): rank(A) = 3 if non-coplanar with origin... actually 3 points in R^3
        # always span at most 3D, and if they don't contain the origin, rank = 3.
        # Solve A c = r0 * 1. This is an underdetermined system (3 equations, 4 unknowns c, r0).
        # Write: A c - r0 * 1 = 0. Stack: [A, -1] [c; r0] = 0.
        # Null space of [A, -1] gives the solution.
        M = np.column_stack([pts, -np.ones(len(pts))])
        # Find null space
        U, s, Vt = np.linalg.svd(M)
        null_vec = Vt[-1]
        c = null_vec[:3]
        r0 = null_vec[3]
        c = c / np.linalg.norm(c)
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
    u0 = np.sin(np.arcsin(np.clip(P0, -1.0, 1.0)) / 2.0) * 2.0
    # Actually: P = sin(x) where x = 2*arcsin(u/2), so u = 2*sin(arcsin(P)/2) = 2*sin(x/2)
    # x = arcsin(P) component-wise, so u = 2*sin(arcsin(P)/2)
    u0 = 2.0 * np.sin(np.arcsin(np.clip(P0, -1.0, 1.0)) / 2.0)

    # Evaluate initial
    P = P0
    cr = _true_covering_radius(P)
    best_score = cr
    best_P = P.copy()

    # Multi-stage optimization with decreasing beta (increasing temperature)
    # Start with high beta (smooth), go to low beta (sharp)
    betas = [10.0, 20.0, 40.0, 80.0, 160.0]
    maxiter_per_stage = 200

    for beta in betas:
        if _elapsed() > _DEADLINE - 15:
            break
        result = minimize(
            _objective,
            u0,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-1.0, 1.0)] * (N * 3),
            options={'maxiter': maxiter_per_stage, 'ftol': 1e-14, 'gtol': 1e-12},
        )
        u_opt = np.clip(result.x, -1.0, 1.0)
        P_opt = _to_sphere(u_opt)
        cr = _true_covering_radius(P_opt)
        if cr < best_score:
            best_score = cr
            best_P = P_opt.copy()
            u0 = u_opt
        else:
            # Keep the better of the two
            if result.fun < _objective(u0, beta):
                u0 = u_opt

    # Final polish with very tight tolerances on the true objective (non-smooth)
    # Use a few local steps with coordinate descent on the true covering radius
    if _elapsed() < _DEADLINE - 10:
        P = best_P.copy()
        for _ in range(50):
            if _elapsed() > _DEADLINE - 10:
                break
            improved = False
            for i in range(N):
                # Try small perturbations of point i
                best_local = P[i].copy()
                best_local_cr = best_score
                for _ in range(20):
                    # Random direction on the tangent plane
                    rand_dir = np.random.randn(3)
                    rand_dir -= np.dot(rand_dir, P[i]) * P[i]
                    rand_dir /= (np.linalg.norm(rand_dir) + 1e-15)
                    for step in [0.01, 0.005, 0.002]:
                        P_new = P.copy()
                        P_new[i] = P[i] + step * rand_dir
                        P_new[i] /= np.linalg.norm(P_new[i])
                        cr = _true_covering_radius(P_new)
                        if cr < best_local_cr - 1e-10:
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
