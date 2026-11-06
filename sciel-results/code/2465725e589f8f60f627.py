"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before 150s kill
    
    # Generate a dense set of probe points on the sphere to approximate the covering radius.
    # Using a Fibonacci sphere with ~20000 points provides a good balance between accuracy and speed.
    M = 20000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    probe_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def compute_covering_radius(pts):
        """Compute the exact covering radius using the convex hull (Delaunay triangulation)."""
        try:
            # Check for valid hull
            hull = ConvexHull(pts)
            max_r = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                # The circumcenter is the point c on the sphere such that |c-p0| = |c-p1| = |c-p2|
                # This is equivalent to c being in the plane of p0, p1, p2 and equidistant.
                # For a triangle on the unit sphere, the circumcenter can be found by solving:
                # c . p0 = c . p1 = c . p2 = k
                # c = a*p0 + b*p1 + d*p2
                # This leads to a linear system. A simpler way for unit vectors:
                # The circumcenter is the normalized vector:
                # c = (sin(A)*p0 + sin(B)*p1 + sin(C)*p2) / norm(...)
                # Where A, B, C are the angles opposite to sides a, b, c.
                
                # Alternative: The circumcenter is the intersection of the plane of the triangle 
                # and the line perpendicular to the plane passing through the origin? No, that's for Euclidean.
                # On the sphere, the circumcenter is the pole of the great circle passing through the vertices? No.
                
                # Let's use the property that the circumcenter c satisfies:
                # (c - p0) . (p1 - p2) = 0  => c . (p1 - p2) = p0 . (p1 - p2)
                # (c - p1) . (p2 - p0) = 0  => c . (p2 - p0) = p1 . (p2 - p0)
                # c is in the span of p0, p1, p2.
                
                # Actually, a robust method:
                # The circumcenter is the normalized solution to:
                # c = (p0 * sin(2*A) + p1 * sin(2*B) + p2 * sin(2*C)) / norm(...)
                # But computing angles is expensive.
                
                # Simpler: The circumcenter is the point on the sphere that is equidistant from p0, p1, p2.
                # This is the intersection of the sphere with the line defined by the cross product of the edges?
                
                # Let's use the linear system approach:
                # Let M be the matrix with rows p0, p1, p2.
                # We want c = M.T @ w such that c . p_i = k for all i.
                # M @ w = k * ones(3)
                # This doesn't work directly because c is in the span of p0, p1, p2.
                # c = a*p0 + b*p1 + d*p2
                # c . p0 = a + b*(p1.p0) + d*(p2.p0) = k
                # c . p1 = a*(p0.p1) + b + d*(p2.p1) = k
                # c . p2 = a*(p0.p2) + b*(p1.p2) + d = k
                
                G = np.array([
                    [1.0, p0 @ p1, p0 @ p2],
                    [p0 @ p1, 1.0, p1 @ p2],
                    [p0 @ p2, p1 @ p2, 1.0]
                ])
                # We want G @ [a, b, d].T = k * [1, 1, 1].T
                # This is underdetermined if we don't fix k.
                # However, the circumcenter is unique (for non-degenerate triangles).
                # The vector c is proportional to the solution of G @ x = 1.
                try:
                    x = np.linalg.solve(G, np.ones(3))
                    c = x[0]*p0 + x[1]*p1 + x[2]*p2
                    norm_c = np.linalg.norm(c)
                    if norm_c < 1e-10:
                        continue
                    c /= norm_c
                    # The distance from c to any vertex is the circumradius.
                    # We need to ensure c is the "farthest" point in the cell.
                    # For a Delaunay triangle, the circumcenter is the farthest point in the Voronoi cell 
                    # IF the circumcenter lies inside the triangle. If not, the farthest point is on the boundary.
                    # However, the covering radius is the max over all cells.
                    # The circumradius of the Delaunay triangles gives the upper bound.
                    # Actually, the covering radius is exactly the max circumradius of the Delaunay triangles
                    # if we consider the spherical Delaunay.
                    
                    dist = np.arccos(np.clip(c @ p0, -1.0, 1.0))
                    if dist > max_r:
                        max_r = dist
                except np.linalg.LinAlgError:
                    pass
            return np.degrees(max_r)
        except Exception:
            return 180.0

    def soft_max_covering_radius(pts, beta=50.0):
        """
        Approximate the covering radius using a soft-max over probe points.
        For each probe point y, d(y) = min_i arccos(y . p_i).
        We want to minimize max_y d(y).
        Soft-max: (1/beta) * log(sum(exp(beta * d(y))))
        """
        # pts: (N, 3)
        # probe_pts: (M, 3)
        
        # Compute dot products: (M, N)
        dots = probe_pts @ pts.T
        # Clip to [-1, 1] for arccos stability
        dots = np.clip(dots, -1.0, 1.0)
        
        # Compute angular distances: (M, N)
        dists = np.arccos(dots)
        
        # For each probe point, find the min distance to any config point
        min_dists = np.min(dists, axis=1)  # (M,)
        
        # Soft-max
        # To avoid overflow, subtract max
        max_d = np.max(min_dists)
        exp_d = np.exp(beta * (min_dists - max_d))
        soft_max = (1.0 / beta) * (max_d + np.log(np.sum(exp_d)))
        
        return soft_max

    def energy(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        return soft_max_covering_radius(pts, beta=50.0)

    # Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Initialize with a Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_energy = energy(pts.flatten())
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Annealing parameters
    T_start = 0.5
    T_end = 0.001
    steps = 10000
    alpha = (T_end / T_start) ** (1.0 / steps)
    T = T_start
    
    # Precompute probe points for speed? No, they are fixed.
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        T *= alpha
        
        # Generate a random perturbation
        # Move a random point by a small random vector
        idx = rng.integers(0, N)
        delta = rng.normal(size=3) * 0.05
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_energy = energy(new_pts.flatten())
        
        # Acceptance criterion
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            prob = np.exp(-(new_energy - current_energy) / T)
            if rng.random() < prob:
                pts = new_pts
                current_energy = new_energy
        
    # Final refinement: use L-BFGS-B on the best found configuration to polish it
    # This helps converge to the local minimum of the smooth energy
    from scipy.optimize import minimize
    
    res = minimize(energy, best_pts.flatten(), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
    if res.success or res.fun < best_energy:
        final_pts = res.x.reshape(-1, 3)
        final_pts /= np.linalg.norm(final_pts, axis=1, keepdims=True)
        # Check if it's valid and better
        # We don't strictly need to check CR here if we trust the energy, 
        # but let's stick to the best energy configuration.
        # Actually, the soft-max might not perfectly align with the true CR.
        # Let's compare the true CR of the SA result and the L-BFGS result.
        
        cr_sa = compute_covering_radius(best_pts)
        cr_lbfgs = compute_covering_radius(final_pts)
        
        if cr_lbfgs < cr_sa:
            best_pts = final_pts
        else:
            best_pts = best_pts

    return best_pts
# EVOLVE-BLOCK-END
