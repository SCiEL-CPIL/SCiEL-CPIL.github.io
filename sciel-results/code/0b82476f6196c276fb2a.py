import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # seconds, leave margin

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Initial points
    points = fibonacci_spiral(N)

    # Dense spherical grid for objective evaluation
    # Use a two-level grid: coarse for fast objective, fine for final check
    n_coarse = 1500
    n_fine = 6000
    grid_coarse = fibonacci_spiral(n_coarse)
    grid_fine = fibonacci_spiral(n_fine)

    def compute_covering_radius_grid(pts, grid):
        """Compute approximate covering radius using grid."""
        dots = np.dot(grid, pts.T)  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)  # max dot = min distance
        max_ang = np.arccos(np.clip(1.0 - min_dots, -1.0, 1.0))
        return float(np.max(max_ang))

    def compute_facet_circumradius(pts):
        """Compute exact covering radius as max circumradius of convex hull facets."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            max_r = 0.0
            for i in range(len(faces)):
                ai, bi, ci = a[i], b[i], c[i]
                # Circumcenter of triangle on sphere
                # Solve for point equidistant from a, b, c
                # Using the formula: circumcenter = (a + b + c) / |a + b + c| only works for equilateral
                # General: find point p such that p.a = p.b = p.c
                # p = alpha*a + beta*b + gamma*c with constraints
                # Simpler: the circumcenter direction is perpendicular to (b-a) and (c-a)
                # and lies on the plane through a, b, c
                # Actually for sphere, the circumcenter of the spherical triangle
                # is in the direction of the cross product approach
                # Let's use: the center of the cap covering the facet
                # The circumradius is the max angular distance from the circumcenter to the vertices
                # For a spherical triangle, the circumcenter is the intersection of perpendicular bisectors
                # Alternative: the cap that covers a, b, c has center p where
                # p.a = p.b = p.c = cos(r)
                # p = (a + b + c) normalized doesn't work in general
                # Use the formula: circumcenter = (|b-c|^2 * a + |c-a|^2 * b + |a-b|^2 * c) / |...|
                # Actually that's the circumcenter in 3D, which for points on unit sphere
                # gives the direction of the circumcenter on the sphere
                bc_sq = np.sum((b - c) ** 2)
                ca_sq = np.sum((c - a) ** 2)
                ab_sq = np.sum((a - b) ** 2)
                # Circumcenter of triangle in 3D
                # C = (a * |b-c|^2 + b * |c-a|^2 + c * |a-b|^2) / (2 * area * 2) ... 
                # Simpler formula:
                # The circumcenter of points a, b, c on the unit sphere
                # is in the direction: a * (|b|^2 |c|^2 - ...) ... 
                # Let me use a direct approach:
                # The circumcenter p satisfies p·a = p·b = p·c
                # p is in the span of a, b, c
                # p = x*a + y*b + z*c
                # p·a = x + y(b·a) + z(c·a) = r
                # p·b = x(a·b) + y + z(c·b) = r
                # p·c = x(a·c) + y(b·c) + z = r
                # And |p| = 1 (since it's on the unit sphere)
                # Actually p doesn't need to be on the unit sphere, just the direction matters
                # The angular distance from p to a is arccos(p·a / |p|)
                # Let's solve the linear system:
                # [1, a·b, a·c] [x,y,z]^T = r
                # [a·b, 1, b·c] [x,y,z]^T = r
                # [a·c, b·c, 1] [x,y,z]^T = r
                # This is singular (all rows sum to same), so we need to add constraint
                # x + y + z = 1 (p is in the affine plane)
                # Actually, let's use a different approach:
                # The circumcenter is the point on the sphere equidistant from a, b, c
                # It's the intersection of the sphere with the perpendicular bisector planes
                # For numerical stability, let's compute the circumradius directly
                # from the side lengths using the spherical law of cosines
                # Side lengths:
                c_side = np.arccos(np.clip(np.dot(a, b), -1, 1))
                b_side = np.arccos(np.clip(np.dot(a, c), -1, 1))
                a_side = np.arccos(np.clip(np.dot(b, c), -1, 1))
                # Spherical circumradius:
                # R = arcsin(sin(A) * sin(B) * sin(C) / (2 * sin(a) * sin(b) * sin(c)))... 
                # Actually, use: cos(R) = (cos(a) + cos(b) + cos(c) - 1) / (2 * cos(A) * cos(B) * cos(C))... 
                # This is getting complicated. Let me use a simpler numerical approach:
                # The circumcenter of a spherical triangle can be found by:
                # p = (sin(a_side) * a + sin(b_side) * b + sin(c_side) * c) / |...|
                # Wait, that's not right either.
                # Let me use the vector approach:
                # The circumcenter is in the direction:
                # n = (a × b) × (b × c) ... no
                # Simplest: the circumcenter is the normalized sum weighted by opposite side sines
                # Actually, for a triangle on the unit sphere, the circumcenter direction is:
                # p ∝ sin(a_side) * a + sin(b_side) * b + sin(c_side) * c
                # where a_side is opposite to vertex a, etc.
                # Let me verify: for an equilateral triangle, this gives p ∝ a + b + c, correct.
                sa = np.sin(a_side)
                sb = np.sin(b_side)
                sc = np.sin(c_side)
                p = sa * a + sb * b + sc * c
                p_norm = np.linalg.norm(p)
                if p_norm > 1e-10:
                    p = p / p_norm
                    # Circumradius is max angular distance from p to vertices
                    r_a = np.arccos(np.clip(np.dot(p, a), -1, 1))
                    r_b = np.arccos(np.clip(np.dot(p, b), -1, 1))
                    r_c = np.arccos(np.clip(np.dot(p, c), -1, 1))
                    r = max(r_a, r_b, r_c)
                    max_r = max(max_r, r)
                else:
                    # Degenerate: use max pairwise distance
                    d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                    d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                    d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                    max_r = max(max_r, max(d_ab, d_bc, d_ca))
            return max_r
        except Exception:
            return compute_covering_radius_grid(pts, grid_fine)

    def lloyd_refine(pts, n_iter=50):
        """Lloyd/Voronoi centroid refinement for uniform distribution."""
        pts = pts.copy()
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            # For each point, find its Voronoi cell centroid on the sphere
            # Use a grid-based approximation
            dots = np.dot(grid_coarse, pts.T)  # (M, N)
            nearest = np.argmax(dots, axis=1)  # (M,)
            # For each point, average the grid points in its cell
            new_pts = np.zeros_like(pts)
            counts = np.zeros(N)
            for j in range(N):
                mask = nearest == j
                if np.sum(mask) > 0:
                    new_pts[j] = np.sum(grid_coarse[mask], axis=0)
                    counts[j] = np.sum(mask)
            # Normalize
            for j in range(N):
                if counts[j] > 0:
                    new_pts[j] /= np.linalg.norm(new_pts[j])
            pts = new_pts
        return pts

    def gradient_descent_coverage(pts, beta_min, beta_max, lr, decay, max_iter):
        """Gradient descent on soft-min/soft-max coverage objective."""
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Compute dots: (M, N)
            dots = np.dot(grid_coarse, pts.T)
            dots = np.clip(dots, -1.0, 1.0)
            ang_dists = np.arccos(dots)  # (M, N)
            
            # Soft-min over points for each grid point
            # s_j = -1/T_min * log(sum_k exp(-T_min * d_jk))
            T_min = beta_min
            # For numerical stability
            max_d = np.max(ang_dists, axis=1, keepdims=True)
            exp_neg = np.exp(-T_min * (ang_dists - max_d))
            sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
            soft_min_dists = -1.0 / T_min * np.log(sum_exp) + max_d.squeeze()
            
            # Weights for soft-min
            w_jk = exp_neg / sum_exp  # (M, N)
            
            # Soft-max over grid points
            T_max = beta_max
            max_sm = np.max(soft_min_dists, keepdims=True)
            exp_sm = np.exp(T_max * (soft_min_dists - max_sm))
            sum_exp_sm = np.sum(exp_sm)
            softmax_max = exp_sm / sum_exp_sm  # (M,)
            
            # Gradient of soft_min_dists[j] w.r.t pts[k]:
            # d(s_j)/d(pts[k]) = sum_k' w_jk' * d(d_jk')/d(pts[k'])
            # d(d_jk)/d(pts[k]) = -grid[j] / sqrt(1 - (grid[j]·pts[k])^2)
            # where d_jk = arccos(grid[j]·pts[k])
            # d(arccos(u))/du = -1/sqrt(1-u^2)
            # du/d(pts[k]) = -grid[j]
            # So d(d_jk)/d(pts[k]) = grid[j] / sqrt(1 - (grid[j]·pts[k])^2)
            
            # Compute gradient efficiently
            # For each grid point j, the contribution to pts[k] is:
            # w_jk * softmax_max[j] * grid[j] / sqrt(1 - dots[j,k]^2)
            
            # sqrt(1 - dots^2) = sin(arccos(dots)) = sin(ang_dists)
            sin_dists = np.sin(ang_dists)  # (M, N)
            sin_dists = np.maximum(sin_dists, 1e-10)
            
            # grad_pts[k] = sum_j softmax_max[j] * w_jk * grid[j] / sin_dists[j,k]
            # = sum_j (softmax_max[j] * w_jk / sin_dists[j,k]) * grid[j]
            
            weights = softmax_max[:, np.newaxis] * w_jk / sin_dists  # (M, N)
            grad_pts = grid_coarse.T @ weights  # (N, 3)
            
            # Update
            pts = pts + lr * grad_pts  # move toward reducing max distance
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts = pts / norms
            
            lr *= decay
        
        return pts

    # Stage 1: Lloyd refinement for good global distribution
    points = lloyd_refine(points, n_iter=30)

    # Stage 2: Gradient descent on coverage field
    # Use moderate beta values
    points = gradient_descent_coverage(points, beta_min=30.0, beta_max=5.0, lr=0.05, decay=0.999, max_iter=500)

    # Stage 3: Fine-tune with higher beta
    points = gradient_descent_coverage(points, beta_min=60.0, beta_max=10.0, lr=0.02, decay=0.999, max_iter=300)

    # Stage 4: Target worst facets
    if time.time() < deadline - 20.0:
        # Find the worst facet and nudge points
        for refine_iter in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Compute facet circumradii
            try:
                hull = ConvexHull(points)
                faces = hull.simplices
                a = points[faces[:, 0]]
                b = points[faces[:, 1]]
                c = points[faces[:, 2]]
                
                max_r = 0.0
                worst_face = 0
                for i in range(len(faces)):
                    ai, bi, ci = a[i], b[i], c[i]
                    a_side = np.arccos(np.clip(np.dot(bi, ci), -1, 1))
                    b_side = np.arccos(np.clip(np.dot(ai, ci), -1, 1))
                    c_side = np.arccos(np.clip(np.dot(ai, bi), -1, 1))
                    sa = np.sin(a_side)
                    sb = np.sin(b_side)
                    sc = np.sin(c_side)
                    p = sa * ai + sb * bi + sc * ci
                    p_norm = np.linalg.norm(p)
                    if p_norm > 1e-10:
                        p = p / p_norm
                        r_a = np.arccos(np.clip(np.dot(p, ai), -1, 1))
                        r_b = np.arccos(np.clip(np.dot(p, bi), -1, 1))
                        r_c = np.arccos(np.clip(np.dot(p, ci), -1, 1))
                        r = max(r_a, r_b, r_c)
                        if r > max_r:
                            max_r = r
                            worst_face = i
                    else:
                        d_ab = np.arccos(np.clip(np.dot(ai, bi), -1, 1))
                        d_bc = np.arccos(np.clip(np.dot(bi, ci), -1, 1))
                        d_ca = np.arccos(np.clip(np.dot(ci, ai), -1, 1))
                        r = max(d_ab, d_bc, d_ca)
                        if r > max_r:
                            max_r = r
                            worst_face = i
                
                # Nudge the three points of the worst facet toward the circumcenter
                wf = faces[worst_face]
                ai, bi, ci = points[wf[0]], points[wf[1]], points[wf[2]]
                a_side = np.arccos(np.clip(np.dot(bi, ci), -1, 1))
                b_side = np.arccos(np.clip(np.dot(ai, ci), -1, 1))
                c_side = np.arccos(np.clip(np.dot(ai, bi), -1, 1))
                sa = np.sin(a_side)
                sb = np.sin(b_side)
                sc = np.sin(c_side)
                p = sa * ai + sb * bi + sc * ci
                p_norm = np.linalg.norm(p)
                if p_norm > 1e-10:
                    p = p / p_norm
                    # Move each vertex slightly toward p
                    step = 0.01
                    points[wf[0]] = points[wf[0]] + step * (p - points[wf[0]])
                    points[wf[1]] = points[wf[1]] + step * (p - points[wf[1]])
                    points[wf[2]] = points[wf[2]] + step * (p - points[wf[2]])
                    # Renormalize
                    for idx in wf:
                        points[idx] /= np.linalg.norm(points[idx])
            except Exception:
                break

    # Stage 5: Final gradient descent polish
    if time.time() < deadline - 5.0:
        points = gradient_descent_coverage(points, beta_min=80.0, beta_max=15.0, lr=0.01, decay=0.999, max_iter=200)

    # Final projection
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points = points / norms

    return points
# EVOLVE-BLOCK-END
