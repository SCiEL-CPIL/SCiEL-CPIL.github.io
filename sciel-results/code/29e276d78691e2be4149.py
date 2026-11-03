"""Rewritten program: multi-stage sphere covering with 50 points.
Uses Riesz energy minimization, smooth LSE covering radius refinement,
Lloyd iterations, and simulated annealing for local search."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 50
WALL_CLOCK_LIMIT = 145.0  # safety margin below 150s


def _normalize(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees using convex hull facets.
    The covering radius equals the maximum circumradius of the facets
    of the convex hull (each facet's circumcenter gives the farthest
    point in that facet's cap).
    """
    points = np.asarray(points, dtype=np.float64)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute-force on a fine grid
        theta = np.linspace(0, np.pi, 300)
        phi = np.linspace(0, 2 * np.pi, 600)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH),
            np.sin(TH) * np.sin(PH),
            np.cos(TH)
        ]).reshape(-1, 3)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        YP = Y @ points.T
        min_angles = np.arccos(np.clip(YP, -1.0, 1.0))
        max_angle = np.max(np.min(min_angles, axis=1))
        return np.degrees(max_angle)

    facets = hull.simplices
    max_ang = 0.0
    for tri in facets:
        a = points[tri[0]]
        b = points[tri[1]]
        c = points[tri[2]]
        # Solve for circumcenter O = alpha*a + beta*b + gamma*c
        # such that |O - a|^2 = |O - b|^2 = |O - c|^2 = 0 (O on unit sphere)
        # Equivalently: 2*O.a = 2*O.b = 2*O.c = 1 + |O|^2 - 1 = ... 
        # Actually: |O-a|^2 = 0 means O.a = (|O|^2 + |a|^2 - |a|^2)/2 = |O|^2/2
        # Since |a|=|b|=|c|=1 and |O|=1: O.a = O.b = O.c = 1/2... no.
        # Let's use the standard approach:
        # |O - a|^2 = |O|^2 - 2 O.a + |a|^2 = 1 - 2 O.a + 1 = 2 - 2 O.a
        # We want the circumcenter of triangle abc on the sphere, which is
        # the point O on the sphere equidistant from a, b, c.
        # |O-a|^2 = |O-b|^2 => O.a = O.b
        # |O-a|^2 = |O-c|^2 => O.a = O.c
        # And O = alpha*a + beta*b + gamma*c with alpha+beta+gamma not constrained to 1
        # since O is on the sphere but not necessarily in the plane of abc... 
        # Actually the circumcenter IS in the plane of abc (for a spherical triangle,
        # the circumcenter lies on the great circle... no, it's the intersection of
        # perpendicular bisectors which are great circles).
        
        # Standard method: solve the linear system
        # O.a = O.b  =>  O.(a-b) = 0
        # O.a = O.c  =>  O.(a-c) = 0
        # O.O = 1
        # This is nonlinear. Instead, use the fact that O = alpha*a + beta*b + gamma*c
        # and impose |O-a|^2 = |O-b|^2 = |O-c|^2.
        
        # |O-a|^2 = |O|^2 - 2O.a + 1
        # |O-b|^2 = |O|^2 - 2O.b + 1
        # Equality: O.a = O.b = O.c
        
        # Let O = alpha*a + beta*b + gamma*c
        # O.a = alpha + beta*(a.b) + gamma*(a.c)
        # O.b = alpha*(a.b) + beta + gamma*(b.c)
        # O.c = alpha*(a.c) + beta*(b.c) + gamma
        
        # O.a = O.b: alpha(1-ab) + beta(ab-1) + gamma(ac-bc) = 0
        # O.a = O.c: alpha(1-ac) + beta(ab-ac) + gamma(ac-1) = 0
        
        ab = a @ b
        ac = a @ c
        bc = b @ c
        
        M = np.array([
            [1.0 - ab, ab - 1.0, ac - bc],
            [1.0 - ac, ab - ac, ac - 1.0]
        ])
        # We have 2 equations, 3 unknowns. The solution is determined up to scale.
        # We need |O| = 1 to fix the scale.
        # Find the null space direction or solve with a constraint.
        # Use SVD to find the direction.
        U, s_vals, Vt = np.linalg.svd(M)
        # The null space is the last row of Vt
        if s_vals[1] < 1e-12:
            continue
        # Actually for 2x3, the null space is 1D
        # The direction satisfying M @ [alpha,beta,gamma]^T = 0
        # is given by the last column of V
        V = Vt.T
        direction = V[:, -1]
        
        O = direction[0] * a + direction[1] * b + direction[2] * c
        O_norm = np.linalg.norm(O)
        if O_norm < 1e-15:
            continue
        O = O / O_norm
        
        # Check that O is on the correct side (inside the cap)
        # The covering radius for this facet is the angular distance from O to a (or b or c)
        cos_ang = np.clip(a @ O, -1.0, 1.0)
        ang = np.arccos(cos_ang)
        if ang > max_ang:
            max_ang = ang

    return np.degrees(max_ang)


def _riesz_energy_grad(points, s=12.0):
    """
    Compute the Riesz s-energy and its gradient for N points on the unit sphere.
    Energy: sum_{i != j} |p_i - p_j|^{-s}
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    mask = ~np.eye(n, dtype=bool)
    energy = np.sum(dist[mask] ** (-s))

    inv_dist_s = dist ** (-s)
    np.fill_diagonal(inv_dist_s, 0.0)
    inv_dist_s2 = inv_dist_s / dist_sq
    np.fill_diagonal(inv_dist_s2, 0.0)

    grad = -s * np.sum(inv_dist_s2[:, :, np.newaxis] * diff, axis=1)

    return energy, grad


def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    """
    n_theta = 60
    n_phi = 120
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH)
    ]).reshape(-1, 3)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    YP = Y @ points.T
    YP = np.clip(YP, -1.0, 1.0)
    angles = np.arccos(YP)

    min_angles = np.min(angles, axis=1)

    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_cr = log_sum_exp / beta

    w_i = np.exp(scaled - max_scaled)
    w_i = w_i / np.sum(w_i)

    gamma = 30.0
    neg_angles = -gamma * angles
    max_neg = np.max(neg_angles, axis=1, keepdims=True)
    exp_neg = np.exp(neg_angles - max_neg)
    v_ij = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)

    denom = np.sqrt(np.maximum(1.0 - YP * YP, 1e-12))
    weights_ij = w_i[:, np.newaxis] * v_ij / denom

    grad = -(1.0 / beta) * (weights_ij.T @ Y)

    return smooth_cr, grad


def _lbfgs_riesz(points, s=12.0, deadline=None, max_iter=3000):
    """Run L-BFGS-B optimization of the Riesz energy on the sphere."""
    x0 = points.flatten()

    def objective(x):
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        energy, grad = _riesz_energy_grad(pts, s=s)
        grad_proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return energy, grad_proj.flatten()

    if deadline is not None:
        remaining = deadline - time.time()
        max_iter = int(min(max_iter, max(200, remaining * 80)))

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )

    x_opt = result.x.reshape(N, 3)
    return _normalize(x_opt)


def _lbfgs_smooth(points, beta=50.0, deadline=None, max_iter=2000):
    """Run L-BFGS-B optimization of the smooth LSE covering radius."""
    x0 = points.flatten()

    def objective(x):
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        cr, grad = _smooth_covering_radius_grad(pts, beta=beta)
        grad_proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return cr, grad_proj.flatten()

    if deadline is not None:
        remaining = deadline - time.time()
        max_iter = int(min(max_iter, max(200, remaining * 60)))

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )

    x_opt = result.x.reshape(N, 3)
    return _normalize(x_opt)


def _spherical_lloyd(points, n_iters=20):
    """
    Perform Lloyd's algorithm on the sphere: for each point, move to the
    centroid of its Voronoi cell approximated by inverse-distance weighting.
    """
    points = points.copy()
    for _ in range(n_iters):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)

        weights = 1.0 / dist_sq
        np.fill_diagonal(weights, 0.0)

        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.maximum(weight_sums, 1e-15)
        new_points = (weights[:, :, np.newaxis] * points[np.newaxis, :, :]).sum(axis=1) / weight_sums
        points = _normalize(new_points)
    return points


def _simulated_annealing(points, deadline, n_iter=None, T0=0.05, T1=0.001):
    """
    Simulated annealing on the sphere to escape local minima.
    Moves one point at a time by a random perturbation scaled by temperature.
    Accepts moves that improve the covering radius with probability exp(-delta/T).
    """
    points = points.copy()
    current_cr = _covering_radius(points)
    best_cr = current_cr
    best_points = points.copy()

    if n_iter is None:
        n_iter = 5000

    T = T0
    cooling = (T1 / T0) ** (1.0 / max(n_iter, 1))
    rng = np.random.default_rng(12345)

    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        T *= cooling

        idx = rng.integers(0, N)
        # Random perturbation on the sphere
        noise = rng.normal(size=3) * T
        new_point = points[idx] + noise
        new_point = new_point / np.linalg.norm(new_point)

        trial_points = points.copy()
        trial_points[idx] = new_point
        trial_cr = _covering_radius(trial_points)

        delta = trial_cr - current_cr
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            points = trial_points
            current_cr = trial_cr
            if current_cr < best_cr:
                best_cr = current_cr
                best_points = points.copy()

    return best_points


def _fibonacci_points(n=N, jitter=0.0, seed=42):
    """Generate Fibonacci spiral points with optional jitter."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        noise = rng.normal(size=(n, 3)) * jitter
        pts = pts + noise
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _random_points(n=N, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based_points(n=N):
    """Generate points based on icosahedron symmetry, then add more points."""
    # Start with icosahedron vertices (12 points)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=np.float64)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)

    # Add face centers (20 faces)
    from scipy.spatial import Delaunay
    try:
        # Use the convex hull to find faces
        hull = ConvexHull(ico)
        faces = hull.simplices
        face_centers = []
        for face in faces:
            fc = ico[face].mean(axis=0)
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)
        face_centers = np.array(face_centers)
    except Exception:
        face_centers = ico  # fallback

    # Combine icosahedron vertices and face centers
    pts = np.vstack([ico, face_centers])  # 32 points

    # If we need more, add edge midpoints
    if n > len(pts):
        edges = set()
        for face in faces:
            for i in range(3):
                a, b = face[i], face[(i+1)%3]
                edge = (min(a,b), max(a,b))
                edges.add(edge)
        edge_centers = []
        for a, b in edges:
            ec = (ico[a] + ico[b]) / 2.0
            ec = ec / np.linalg.norm(ec)
            edge_centers.append(ec)
        edge_centers = np.array(edge_centers)
        pts = np.vstack([pts, edge_centers])

    # If still need more, add random points
    if n > len(pts):
        rng = np.random.default_rng(99)
        extra = rng.normal(size=(n - len(pts), 3))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        pts = np.vstack([pts, extra])

    return pts[:n]


def _refine(points, deadline):
    """
    Multi-stage refinement: Riesz -> Lloyd -> LSE smooth -> Riesz polish -> LSE polish.
    """
    best_points = points.copy()
    best_cr = _covering_radius(points)

    # Stage 1: Riesz energy minimization (s=12)
    if time.time() < deadline - 20.0:
        try:
            rpts = _lbfgs_riesz(points, s=12.0, deadline=deadline)
            cr = _covering_radius(rpts)
            if cr < best_cr:
                best_cr = cr
                best_points = rpts
        except Exception:
            pass

    # Stage 2: Lloyd iteration
    if time.time() < deadline - 15.0:
        try:
            lpts = _spherical_lloyd(best_points, n_iters=20)
            cr = _covering_radius(lpts)
            if cr < best_cr:
                best_cr = cr
                best_points = lpts
        except Exception:
            pass

    # Stage 3: Smooth LSE covering radius refinement
    if time.time() < deadline - 10.0:
        try:
            spoints = _lbfgs_smooth(best_points, beta=50.0, deadline=deadline)
            cr = _covering_radius(spoints)
            if cr < best_cr:
                best_cr = cr
                best_points = spoints
        except Exception:
            pass

    # Stage 4: Riesz polish with higher s for packing-like behavior
    if time.time() < deadline - 8.0:
        try:
            rpts2 = _lbfgs_riesz(best_points, s=20.0, deadline=deadline)
            cr = _covering_radius(rpts2)
            if cr < best_cr:
                best_cr = cr
                best_points = rpts2
        except Exception:
            pass

    # Stage 5: Another LSE pass with higher beta
    if time.time() < deadline - 5.0:
        try:
            spoints2 = _lbfgs_smooth(best_points, beta=100.0, deadline=deadline)
            cr = _covering_radius(spoints2)
            if cr < best_cr:
                best_cr = cr
                best_points = spoints2
        except Exception:
            pass

    # Stage 6: Final Lloyd + Riesz
    if time.time() < deadline - 3.0:
        try:
            lpts2 = _spherical_lloyd(best_points, n_iters=10)
            cr = _covering_radius(lpts2)
            if cr < best_cr:
                best_cr = cr
                best_points = lpts2
        except Exception:
            pass

    if time.time() < deadline - 2.0:
        try:
            rpts3 = _lbfgs_riesz(best_points, s=12.0, deadline=deadline)
            cr = _covering_radius(rpts3)
            if cr < best_cr:
                best_cr = cr
                best_points = rpts3
        except Exception:
            pass

    return best_points, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + WALL_CLOCK_LIMIT

    # Generate diverse initial configurations
    configs = []

    # 1. Fibonacci spiral (no jitter)
    configs.append(_fibonacci_points(N, jitter=0.0))

    # 2. Jittered Fibonacci
    for jit in [0.05, 0.1, 0.15, 0.2]:
        configs.append(_fibonacci_points(N, jitter=jit, seed=42 + int(jit * 100)))

    # 3. Random configurations
    for seed in [0, 1, 2, 3, 4, 5]:
        configs.append(_random_points(N, seed=seed))

    # 4. Icosahedron-based
    configs.append(_icosahedron_based_points(N))

    best_points = None
    best_radius = np.inf

    # Phase 1: Riesz energy optimization from multiple starts
    for idx, init in enumerate(configs):
        if time.time() > deadline - 30.0:
            break

        try:
            optimized = _lbfgs_riesz(init, s=12.0, deadline=deadline)
            radius = _covering_radius(optimized)
            if radius < best_radius:
                best_radius = radius
                best_points = optimized.copy()
        except Exception:
            continue

    if best_points is None:
        return _fibonacci_points(N)

    # Phase 2: Full refinement on the best point
    best_points, best_radius = _refine(best_points, deadline)

    # Phase 3: Try simulated annealing if time permits
    if time.time() < deadline - 5.0:
        try:
            sa_points = _simulated_annealing(best_points, deadline, n_iter=2000)
            sa_cr = _covering_radius(sa_points)
            if sa_cr < best_radius:
                best_radius = sa_cr
                best_points = sa_points
                # Refine the SA result
                if time.time() < deadline - 3.0:
                    best_points, best_radius = _refine(best_points, deadline)
        except Exception:
            pass

    # Final: ensure we return a valid configuration
    best_points = _normalize(best_points)
    return best_points
