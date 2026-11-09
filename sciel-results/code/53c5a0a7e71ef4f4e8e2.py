# EVOLVE-BLOCK-START
import numpy as np
import time


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon with equidistant vertices.
    Starts with a regular 20-gon and attempts local optimization to maximize
    the minimum vertex score while maintaining strict convexity.
    """
    n = 20
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    def compute_score(polygon):
        """Compute the minimum vertex score for a given polygon. Returns 0.0 if invalid."""
        pts = np.array(polygon, dtype=float)
        if pts.shape != (n, 2):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0

        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = pts.mean(axis=0)
        pts_norm = pts - centroid
        dists_to_centroid = np.linalg.norm(pts_norm, axis=1)
        mean_dist = dists_to_centroid.mean()
        if mean_dist <= 0.01:
            return 0.0
        pts_norm = pts_norm / mean_dist

        # Check for repeated points
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts_norm[i] - pts_norm[j]) < 0.05:
                    return 0.0

        # Check strict convexity: all cross products same sign and nonzero
        cross_signs = []
        for i in range(n):
            p0 = pts_norm[i]
            p1 = pts_norm[(i + 1) % n]
            p2 = pts_norm[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-12:
                return 0.0
            cross_signs.append(np.sign(cross))

        if len(set(cross_signs)) != 1:
            return 0.0

        # Total turning should be +-2*pi (implicit in convexity check above)

        # Compute vertex scores
        min_score = 1.0
        for i in range(n):
            v = pts_norm[i]
            dists = np.linalg.norm(pts_norm - v, axis=1)
            dists[i] = np.inf  # exclude self
            d = np.sort(dists)  # 19 distances

            if d[0] < 0.05:
                return 0.0

            gaps = d[1:] - d[:-1]  # 18 gaps
            k = np.argmin(gaps)  # first index minimizing gap

            D = (d[k] + d[k + 1]) / 2.0
            if D < 0.1:
                return 0.0

            if k == 0:
                d_sel = d[0:4]
            elif k == 17:
                d_sel = d[15:19]
            else:
                d_sel = d[k - 1:k + 3]

            ratios = np.maximum(d_sel / D, D / d_sel)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)

            if score < min_score:
                min_score = score
                if min_score == 0.0:
                    return 0.0

        return min_score

    # Initialize with regular 20-gon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)

    def get_polygon(radii):
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    best_polygon = get_polygon(radii)
    best_score = compute_score(best_polygon)

    # Local optimization: perturb radii to try to improve score
    rng = np.random.default_rng(42)
    step_size = 0.05
    max_iterations = 10000

    for iteration in range(max_iterations):
        if time.time() > deadline - 30.0:
            break

        # Pick 1-3 random vertices to perturb
        num_perturb = rng.integers(1, 4)
        idxs = rng.choice(n, num_perturb, replace=False)

        new_radii = radii.copy()
        for idx in idxs:
            delta = rng.uniform(-step_size, step_size)
            new_radii[idx] += delta
            if new_radii[idx] < 0.1:
                new_radii[idx] = 0.1 + rng.uniform(0, 0.05)

        new_polygon = get_polygon(new_radii)
        new_score = compute_score(new_polygon)

        if new_score > best_score:
            best_score = new_score
            best_polygon = new_polygon
            radii = new_radii
            # Reduce step size on improvement
            step_size *= 0.95
        else:
            # Occasionally try larger steps if stuck
            if iteration % 100 == 0:
                step_size = max(0.01, step_size * 1.1)

    # Final check
    if compute_score(best_polygon) < best_score:
        # Revert to regular polygon if something went wrong
        best_polygon = get_polygon(np.ones(n))
        best_score = compute_score(best_polygon)

    return best_polygon


# EVOLVE-BLOCK-END
