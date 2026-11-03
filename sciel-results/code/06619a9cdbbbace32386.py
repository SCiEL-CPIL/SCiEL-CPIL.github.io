"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.optimize import minimize

N = 130
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _angular_distance_matrix(P):
    """Compute pairwise angular distances (in radians) for points on unit sphere."""
    # Normalize just in case
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    P = P / norms
    # Dot products
    dots = np.clip(P @ P.T, -1.0, 1.0)
    return np.arccos(dots)

def _covering_radius(P):
    """Compute the covering radius in degrees."""
    # The covering radius is the maximum over the sphere of min angular distance to any point.
    # This equals the maximum circumradius of the convex hull facets (for points in general position on sphere).
    # Alternative: use the fact that covering radius = max over Voronoi vertices of distance to nearest site.
    # More robust: compute convex hull and find max circumradius of facets.
    try:
        hull = ConvexHull(P)
        facets = hull.simplices  # shape (M, 3) for 3D
        max_r = 0.0
        for simplex in facets:
            a, b, c = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            # Circumradius of triangle abc inscribed in unit sphere
            # The circumcenter of the triangle projected onto the sphere gives the farthest point in this facet
            # Circumradius formula: R = abc / (4 * area)
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ca = np.linalg.norm(a - c)
            # Area via cross product
            cross = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(cross)
            if area < 1e-15:
                continue
            R = (ab * bc * ca) / (4.0 * area)
            # Angular circumradius = arcsin(R) for triangle on unit sphere? No.
            # For a triangle on the unit sphere, the circumradius (angular) is the angle from the circumcenter to a vertex.
            # The circumcenter of the planar triangle projected onto the sphere:
            # Actually, for points on the unit sphere, the circumcircle of the triangle on the sphere
            # has angular radius theta where sin(theta) = R_planar / 1 ... no.
            # Let's use: the farthest point in the spherical triangle from all three vertices is the circumcenter.
            # The angular distance from circumcenter to any vertex is the circumradius.
            # For a triangle with side lengths (angular) a,b,c, circumradius R satisfies:
            # sin(R) = (side_a * side_b * side_c) / (4 * area_of_spherical_triangle) -- this is complex.
            # Simpler: the circumcenter of the planar triangle, normalized, gives the point on sphere.
            # The angular distance from that point to any vertex is the circumradius.
            # Planar circumcenter:
            d = 2.0 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
            if abs(d) < 1e-15:
                continue
            ux = ((a[0]**2 + a[1]**2 + a[2]**2)*(b[1]-c[1]) + (b[0]**2+b[1]**2+b[2]**2)*(c[1]-a[1]) + (c[0]**2+c[1]**2+c[2]**2)*(a[1]-b[1])) / d
            uy = ((a[0]**2 + a[1]**2 + a[2]**2)*(c[0]-a[0]) + (b[0]**2+b[1]**2+b[2]**2)*(a[0]-c[0]) + (c[0]**2+c[1]**2+c[2]**2)*(b[0]-a[0])) / d
            uz = ux * 0 + uy * 0  # need full 3D
            # Better approach: use the formula for circumcenter in 3D
            # C = a + (b-a) * s + (c-a) * t
            # where s,t solve: |a + (b-a)s + (c-a)t - a|^2 = |... - b|^2 = |... - c|^2
            # Simplified: circumcenter of triangle abc:
            A = np.array([a, b, c])
            # Use the formula: circumcenter = (|a|^2(b-c) + |b|^2(c-a) + |c|^2(a-b)) / (2 * (a x b + b x c + c x a)) ... not quite
            # Let's just compute it properly
            b_minus_a = b - a
            c_minus_a = c - a
            # Solve for s, t such that:
            # 2 * s * (b-a).(b-a) + 2 * t * (c-a).(b-a) = |b|^2 - |a|^2
            # 2 * s * (b-a).(c-a) + 2 * t * (c-a).(c-a) = |c|^2 - |a|^2
            A_mat = np.array([[2*(b_minus_a @ b_minus_a), 2*(c_minus_a @ b_minus_a)],
                              [2*(b_minus_a @ c_minus_a), 2*(c_minus_a @ c_minus_a)]])
            b_vec = np.array([b @ b - a @ a, c @ c - a @ a])
            try:
                st = np.linalg.solve(A_mat, b_vec)
            except:
                continue
            circum = a + st[0] * b_minus_a + st[1] * c_minus_a
            norm_circum = np.linalg.norm(circum)
            if norm_circum < 1e-15:
                continue
            circum = circum / norm_circum
            # Angular distance from circum to a
            dot_val = np.clip(circum @ a, -1.0, 1.0)
            ang_r = np.arccos(dot_val)
            if ang_r > max_r:
                max_r = ang_r
        return np.degrees(max_r)
    except:
        # Fallback: use k-d tree and sample
        tree = cKDTree(P)
        # Sample many points on sphere
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(10000, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.degrees(np.max(dists))

def _lloyd_iteration(P, n_iter=50):
    """Perform Lloyd's algorithm iterations for spherical Voronoi relaxation."""
    P = P.copy()
    for _ in range(n_iter):
        # Compute Voronoi cells on the sphere using Delaunay triangulation
        try:
            # Use spherical Voronoi via Delaunay
            # For points on unit sphere, we can use the dual of Delaunay
            # Simpler: use the fact that Voronoi cell centroid = average of facet circumcenters
            # Or use scipy's Delaunay on 3D points
            hull = ConvexHull(P)
            facets = hull.simplices
            # For each point, find its Voronoi cell (dual of Delaunay)
            # The Voronoi cell of point i consists of all facets of the convex hull
            # that are "visible" from point i... actually, in 3D convex hull,
            # each vertex's Voronoi cell is the set of facets not containing it? No.
            # Better approach: use the fact that for points in convex position,
            # the Voronoi diagram on the sphere is dual to the Delaunay triangulation.
            # For now, use a simpler heuristic: move each point toward the average
            # of its neighbors in the Delaunay graph.
            delaunay = Delaunay(P)
            # Build adjacency
            adj = {i: set() for i in range(N)}
            for simplex in delaunay.simplices:
                for i in simplex:
                    for j in simplex:
                        if i != j:
                            adj[i].add(j)
            # Move each point toward centroid of its neighbors
            new_P = P.copy()
            for i in range(N):
                if len(adj[i]) > 0:
                    neighbors = P[list(adj[i])]
                    centroid = np.mean(neighbors, axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-10:
                        centroid = centroid / norm
                        new_P[i] = centroid
                else:
                    new_P[i] = P[i]
            P = new_P
        except:
            break
    return P

def construct():
    start_time = time.time()
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_P = P.copy()
    best_cr = _covering_radius(P)
    
    # Lloyd's algorithm with multiple restarts
    rng = np.random.default_rng(123)
    
    def _score_and_update(P_candidate):
        nonlocal best_P, best_cr
        cr = _covering_radius(P_candidate)
        if cr < best_cr:
            best_cr = cr
            best_P = P_candidate.copy()
        return cr
    
    # First, run Lloyd from the initial configuration
    P_lloyd = _lloyd_iteration(P, n_iter=100)
    _score_and_update(P_lloyd)
    
    # Try multiple random restarts with perturbation + Lloyd
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Perturb the best configuration
        P_perturb = best_P + rng.normal(scale=0.05, size=(N, 3))
        P_perturb /= np.linalg.norm(P_perturb, axis=1, keepdims=True)
        
        # Run Lloyd
        P_lloyd = _lloyd_iteration(P_perturb, n_iter=80)
        _score_and_update(P_lloyd)
        
        # Also try smaller perturbations
        if time.time() - start_time > TIME_LIMIT:
            break
        P_perturb2 = best_P + rng.normal(scale=0.01, size=(N, 3))
        P_perturb2 /= np.linalg.norm(P_perturb2, axis=1, keepdims=True)
        P_lloyd2 = _lloyd_iteration(P_perturb2, n_iter=60)
        _score_and_update(P_lloyd2)
    
    # Final polish: local optimization on the best configuration
    if time.time() - start_time < TIME_LIMIT - 30:
        P_opt = best_P.copy()
        # Use gradient-free local optimization
        from scipy.optimize import differential_evolution
        
        def _objective(flat_params):
            P_temp = flat_params.reshape(N, 3)
            norms = np.linalg.norm(P_temp, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            P_temp = P_temp / norms
            return _covering_radius(P_temp)
        
        # Use a few iterations of Nelder-Mead on the best
        try:
            res = minimize(_objective, P_opt.flatten(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_cr:
                P_final = res.x.reshape(N, 3)
                P_final /= np.linalg.norm(P_final, axis=1, keepdims=True)
                best_P = P_final
                best_cr = res.fun
        except:
            pass
    
    # Ensure output is valid
    best_P = np.asarray(best_P, dtype=np.float64)
    norms = np.linalg.norm(best_P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_P = best_P / norms
    
    return best_P
