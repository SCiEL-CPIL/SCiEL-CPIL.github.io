"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """
        Compute the covering radius via convex hull facets.
        """
        try:
            pts = normalize(pts)
            if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
                return 180.0

            hull = ConvexHull(pts)

            max_r = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])

                try:
                    w = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    continue

                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    continue

                c /= c_norm

                cos_d = np.clip(c @ p0, -1.0, 1.0)
                d = np.arccos(cos_d)

                if d > max_r:
                    max_r = d

            return np.degrees(max_r)

        except Exception:
            return 180.0

    def get_facet_circumcenters(pts):
        """
        For each facet, compute the circumcenter (point on sphere equidistant from 3 vertices).
        Returns list of (face_indices, circumcenter).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return []

        results = []
        for face in hull.simplices:
            p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
            p_face = np.array([p0, p1, p2])
            G = p_face @ p_face.T
            rhs = np.array([1.0, 1.0, 1.0])
            try:
                w = np.linalg.solve(G, rhs)
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm > 1e-10:
                    c /= c_norm
                    results.append((face, c))
            except Exception:
                pass
        return results

    def facet_based_gradient(pts):
        """
        Compute gradient of the max facet circumradius.
        For the worst facet(s), push vertices to reduce circumradius.
        """
        pts = normalize(pts)
        facets = get_facet_circumcenters(pts)
        if not facets:
            return np.zeros_like(pts)

        # Compute circumradius for each facet
        radii = []
        for face, c in facets:
            cos_d = np.clip(c @ pts[face[0]], -1.0, 1.0)
            d = np.arccos(cos_d)
            radii.append(d)

        radii = np.array(radii)
        max_r = np.max(radii)

        if max_r < 1e-10:
            return np.zeros_like(pts)

        # Soft-max weighting over facets
        beta = 20.0
        exp_r = np.exp(beta * (radii - max_r))
        weights = exp_r / np.sum(exp_r)

        grad = np.zeros_like(pts)
        for idx, (face, c) in enumerate(facets):
            w = weights[idx]
            if w < 1e-8:
                continue

            # The circumcenter c is the "worst point" for this facet.
            # To reduce circumradius, move vertices away from c (towards the facet normal direction).
            # Actually, we want to move vertices so the circumradius decreases.
            # The circumcenter is equidistant from all 3 vertices. To reduce the radius,
            # we should move vertices towards the centroid of the facet (in spherical sense).
            # A simpler approach: move each vertex slightly towards the average of the other two.
            for v_idx in face:
                p_v = pts[v_idx]
                # Direction to move: towards the "center" of the opposite side
                # For vertex v_idx in face (i,j,k), move towards midpoint of the other two
                others = [f for f in face if f != v_idx]
                p_other = (pts[others[0]] + pts[others[1]])
                p_other = p_other / np.linalg.norm(p_other)

                # Spherical gradient: move p_v towards p_other
                # The gradient of distance to c is -c/sin(d) in tangent space
                # Simpler: just move towards p_other in tangent space
                tangent = p_other - np.dot(p_v, p_other) * p_v
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-10:
                    grad[v_idx] += w * tangent / tangent_norm

        return grad

    def lloyd_polish(pts, max_iters=100, step_size=0.15):
        """
        Lloyd's algorithm: move each point towards the centroid of circumcenters of adjacent facets.
        """
        pts = normalize(pts)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            facets = get_facet_circumcenters(pts)
            if not facets:
                break

            sum_centers = np.zeros_like(pts)
            counts = np.zeros(N)

            for face, c in facets:
                for v_idx in face:
                    sum_centers[v_idx] += c
                    counts[v_idx] += 1

            moved = False
            for i in range(N):
                if counts[i] > 0:
                    target = sum_centers[i] / counts[i]
                    target_norm = np.linalg.norm(target)
                    if target_norm > 1e-10:
                        target /= target_norm
                        new_pt = pts[i] + step_size * (target - pts[i])
                        new_pt_norm = np.linalg.norm(new_pt)
                        if new_pt_norm > 1e-10:
                            pts[i] = new_pt / new_pt_norm
                            moved = True

            if not moved:
                break

        return pts

    def local_search(pts, n_attempts=200):
        """
        Local search: randomly perturb a point and accept if it improves the covering radius.
        """
        pts = normalize(pts)
        best_cr = compute_covering_radius(pts)

        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break

            # Pick a random point to perturb
            i = np.random.randint(N)
            # Random perturbation on the sphere
            delta = np.random.randn(3) * 0.1
            new_pt = pts[i] + delta
            new_pt = new_pt / np.linalg.norm(new_pt)

            test_pts = pts.copy()
            test_pts[i] = new_pt

            cr = compute_covering_radius(test_pts)
            if cr < best_cr:
                best_cr = cr
                pts = test_pts

        return pts, best_cr

    def simulated_annealing(pts, n_iters=500, t0=0.5, t1=0.01):
        """
        Simulated annealing to escape local minima.
        """
        pts = normalize(pts)
        best_pts = pts.copy()
        best_cr = compute_covering_radius(pts)
        current_cr = best_cr

        for it in range(n_iters):
            if time.time() > deadline - 15:
                break

            temp = t0 * (t1 / t0) ** (it / n_iters)

            i = np.random.randint(N)
            delta = np.random.randn(3) * (0.05 + temp * 0.3)
            new_pt = pts[i] + delta
            new_pt = new_pt / np.linalg.norm(new_pt)

            test_pts = pts.copy()
            test_pts[i] = new_pt

            cr = compute_covering_radius(test_pts)
            delta_cr = cr - current_cr

            if delta_cr < 0 or np.random.rand() < np.exp(-delta_cr / (temp * 0.1)):
                pts = test_pts
                current_cr = cr
                if cr < best_cr:
                    best_cr = cr
                    best_pts = pts.copy()

        return best_pts, best_cr

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Main optimization
    np.random.seed(42)

    # Start with Fibonacci sphere
    pts = fibonacci_sphere(N)
    best_pts = pts.copy()
    best_cr = compute_covering_radius(pts)

    # Phase 1: Lloyd's algorithm
    if time.time() < deadline - 30:
        pts_lloyd = lloyd_polish(pts, max_iters=80, step_size=0.2)
        cr_lloyd = compute_covering_radius(pts_lloyd)
        if cr_lloyd < best_cr:
            best_cr = cr_lloyd
            best_pts = pts_lloyd

    # Phase 2: Facet-based gradient descent
    if time.time() < deadline - 25:
        pts_grad = normalize(best_pts)
        lr = 0.1
        for it in range(200):
            if time.time() > deadline - 20:
                break
            grad = facet_based_gradient(pts_grad)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            pts_grad = normalize(pts_grad - lr * grad / grad_norm)
            cr = compute_covering_radius(pts_grad)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts_grad.copy()
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                lr *= 0.7

    # Phase 3: Local search
    if time.time() < deadline - 15:
        pts_ls, cr_ls = local_search(best_pts, n_attempts=300)
        if cr_ls < best_cr:
            best_cr = cr_ls
            best_pts = pts_ls

    # Phase 4: Simulated annealing
    if time.time() < deadline - 10:
        pts_sa, cr_sa = simulated_annealing(best_pts, n_iters=400, t0=0.3, t1=0.02)
        if cr_sa < best_cr:
            best_cr = cr_sa
            best_pts = pts_sa

    # Phase 5: Final Lloyd polish
    if time.time() < deadline - 5:
        pts_final = lloyd_polish(best_pts, max_iters=50, step_size=0.05)
        cr_final = compute_covering_radius(pts_final)
        if cr_final < best_cr:
            best_cr = cr_final
            best_pts = pts_final

    # Phase 6: One more local search pass
    if time.time() < deadline - 3:
        pts_ls2, cr_ls2 = local_search(best_pts, n_attempts=100)
        if cr_ls2 < best_cr:
            best_cr = cr_ls2
            best_pts = pts_ls2

    return normalize(best_pts)
# EVOLVE-BLOCK-END
