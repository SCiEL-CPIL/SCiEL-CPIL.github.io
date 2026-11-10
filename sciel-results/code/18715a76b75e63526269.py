"""Repulsive-dynamics + soft-max smoothing for 50-point sphere covering."""
import time

N = 50
import numpy as np

_DEADLINE = time.time() + 130.0  # safety margin below 150 s


def _time_left():
    return _DEADLINE - time.time()


def _normalize(P):
    P = P.copy()
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return P / norms


def _covering_radius_deg(P):
    """Exact covering radius in degrees: max over y of angular distance to nearest point.
    This equals the largest circumradius of the convex-hull facets (Voronoi cells)."""
    P = _normalize(P)
    # For each point i, its Voronoi cell on the sphere: the farthest point in its cell
    # is determined by the facet of the convex hull. The covering radius is the max
    # angular distance from a point to its nearest site, which equals the max over
    # each site of the circumradius of its cell.
    # Compute via: for each pair of nearby sites, the cell boundary.
    # Simpler exact approach: the covering radius = max over all points of the
    # largest empty cap centered on a Voronoi vertex.
    # We compute the convex hull of P in R^3, take each facet, and the facet's
    # circumradius (angular) is the distance from the origin direction to the
    # facet plane, i.e. arcsin of the distance from origin to the plane.
    from scipy.spatial import ConvexHull
    hull = ConvexHull(P)
    max_r = 0.0
    for simplex in hull.simplices:
        tri = P[simplex]
        # Circumradius of a spherical triangle = the angular distance from the
        # origin to the plane through the three points, measured from the
        # triangle's centroid direction. The farthest point in this cell from
        # all three is the circumcenter; its angular distance to each vertex
        # is the circumradius.
        # Compute the plane normal and distance from origin.
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        d = np.dot(n, tri[0])  # distance from origin to plane
        d = np.clip(d, -1.0, 1.0)
        # The cell extends from the triangle out to the plane; the farthest
        # point is at angular distance asin(|d|) from the triangle centroid
        # direction... Actually the circumradius is the angle between a vertex
        # and the circumcenter. For a facet with plane distance d from origin,
        # the circumradius (angular) = arcsin(1/d) is wrong; correct: the
        # circumcenter is at distance 1/sqrt(1-d^2) along n, normalized to
        # unit sphere. The angular distance from vertex to circumcenter =
        # arccos(d * sqrt(1-d^2)/1) ... let's just compute directly.
        # Circumcenter on sphere: direction n (or -n), pick the one on the
        # same side as the triangle (dot with tri[0] > 0).
        if np.dot(n, tri[0]) < 0:
            n = -n
        # Angular distance from a vertex to n:
        cosang = np.clip(np.dot(tri[0], n), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        if ang > max_r:
            max_r = ang
    return max_r


def _soft_covering_radius(P, beta):
    """Smooth surrogate: soft-max of per-site circumradii.
    For each site i, compute a soft measure of its cell size, then soft-max."""
    P = _normalize(P)
    # Pairwise angular distances (in radians), N x N
    G = np.clip(np.dot(P, P.T), -1.0, 1.0)
    D = np.arccos(G)  # angular distance matrix, radians
    # For each site i, its "local covering radius" approx = max_j D[i,j] over
    # its neighbors, but we use a soft version. The covering radius is the max
    # over the sphere; a good smooth proxy is the soft-max over i of the
    # soft-max over j of D[i,j] weighted by nearest-neighbor structure.
    # Use: for each i, softmax_j D[i,j] with beta, then softmax_i of that.
    # To keep it cheap, use a moderate beta.
    inv = D * beta
    # soft-max over j for each i
    m = np.max(inv, axis=1, keepdims=True)
    e = np.exp(inv - m)
    sm_i = (m + np.log(np.sum(e, axis=1, keepdims=True)))[:, 0] / beta
    # soft-max over i
    m2 = np.max(sm_i, keepdims=True)
    e2 = np.exp((sm_i - m2) * beta)
    soft = (m2 + np.log(np.sum(e2)) / beta)
    return np.degrees(soft[0])


def _repulsive_forces(P, beta):
    """Compute a repulsive force on each point. Points in sparse (empty) regions
    experience stronger repulsion so they flow into gaps.

    Force on i = sum_j w_ij * (P_i - P_j) / |P_i - P_j|^2  (standard repulsion)
    where w_ij is scaled by inverse local density of j (empty regions push more).
    """
    P = _normalize(P)
    G = np.clip(np.dot(P, P.T), -1.0, 1.0)
    D = np.arccos(G)
    # Local density of each point: number of neighbors within a small angle.
    # Use a soft count.
    rho = np.exp(-D * beta)  # N x N, rho[i,j]
    density = np.sum(rho, axis=1)  # local density per point
    # Inverse density: points in empty regions get weight > 1
    inv_density = 1.0 / np.maximum(density, 1e-6)
    # Repulsive force: for each i, sum over j of (P_i - P_j) * strength
    # strength scaled by inv_density[j] (push from sparse areas) and by 1/(1+D^2)
    forces = np.zeros_like(P)
    for i in range(N):
        # D[i,:] is distance from i to all j
        d2 = D[i, :] ** 2
        d2[i] = 1e6  # avoid self
        w = inv_density / (1.0 + d2)
        w[i] = 0.0
        # force direction: P_i - P_j for each j
        diff = P[i][None, :] - P  # N x 3
        forces[i] = np.sum(w[:, None] * diff, axis=0)
    return forces


def _project_to_sphere(P, forces, step):
    """Move points along force, reproject to sphere, keep in a valid hemisphere
    check later."""
    P = _normalize(P)
    P_new = P + step * forces
    P_new = _normalize(P_new)
    return P_new


def _optimize_from_start(P0, beta=20.0, max_iter=4000):
    """Repulsive dynamics + soft-max objective refinement."""
    P = _normalize(P0)
    best = P.copy()
    best_val = _covering_radius_deg(P)
    step = 0.02
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        F = _repulsive_forces(P, beta)
        P = _project_to_sphere(P, F, step)
        # Occasionally evaluate exact objective and keep best
        if it % 50 == 0:
            val = _covering_radius_deg(P)
            if val < best_val - 1e-9:
                best_val = val
                best = P.copy()
        # Anneal step size
        step = 0.02 * (0.999 ** it)
        if step < 1e-5:
            break
    # Final polish: high-precision local refinement
    best = _polish(best)
    return best, _covering_radius_deg(best)


def _polish(P, max_iter=2000):
    """High-precision polish: gradient descent on the exact covering radius
    using the facet circumradius. We use a simple projected gradient step on
    the worst facet's circumcenter direction."""
    P = _normalize(P)
    from scipy.spatial import ConvexHull
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        try:
            hull = ConvexHull(P)
        except Exception:
            break
        # Find the facet with largest circumradius
        worst = -1
        worst_r = 0.0
        worst_n = None
        worst_tri = None
        for simplex in hull.simplices:
            tri = P[simplex]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            if np.dot(n, tri[0]) < 0:
                n = -n
            cosang = np.clip(np.dot(tri[0], n), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            if ang > worst_r:
                worst_r = ang
                worst = simplex[0]
                worst_n = n
                worst_tri = simplex
        if worst < 0:
            break
        # Move the worst facet's vertices toward the circumcenter to shrink it
        # (the circumcenter is the farthest point; moving vertices toward it
        # reduces the cell's circumradius).
        step = 0.01 * (0.995 ** it)
        for idx in worst_tri:
            P[idx] = P[idx] + step * worst_n
        P = _normalize(P)
    return P


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(rng):
    v = rng.normal(size=(N, 3))
    return _normalize(v)


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 130.0
    rng = np.random.default_rng(12345)
    best_P = None
    best_val = np.inf

    # Start 1: Fibonacci spiral
    P0 = _fibonacci_start()
    P, val = _optimize_from_start(P0, beta=20.0, max_iter=4000)
    if val < best_val:
        best_val = val
        best_P = P

    if _time_left() < 20.0:
        return best_P

    # Start 2-4: random restarts
    for k in range(3):
        if _time_left() < 15.0:
            break
        P0 = _random_start(rng)
        P, val = _optimize_from_start(P0, beta=20.0, max_iter=3000)
        if val < best_val:
            best_val = val
            best_P = P

    if _time_left() < 5.0:
        return best_P

    # Final polish on the best
    best_P = _polish(best_P, max_iter=1500)
    return best_P
