"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def calculate_radius(points):
        """Calculate the maximum radius for a given set of points."""
        if len(points) < 2:
            return 1.0
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        return min(min_pair_dist / 2.0, min_bound_dist)

    def bottleneck_info(points):
        """Return (radius, is_pair, i, j_or_idx) describing the active constraint."""
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)

        if min_pair_dist / 2.0 < min_bound_dist:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            return min_pair_dist / 2.0, True, idx[0], idx[1]
        else:
            idx = np.argmin(1.0 - r_i)
            return min_bound_dist, False, idx, -1

    def hex_lattice_init(N_points):
        """Generate a hexagonal lattice initialization."""
        s = np.sqrt(2.0 * np.pi / (3.0 * np.sqrt(3.0) * N_points))
        k = int(1.0 / s) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        points = np.array(rows)
        dists = np.linalg.norm(points, axis=1)
        points = points[dists <= 1.0 - s / 2]
        if len(points) < N_points:
            s *= 0.9
            k = int(1.0 / s) + 3
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = s * (i + 0.5 * (j % 2))
                    y = s * j * np.sqrt(3.0) / 2.0
                    rows.append([x, y])
            points = np.array(rows)
            dists = np.linalg.norm(points, axis=1)
            points = points[dists <= 1.0 - s / 2]
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        return points[sorted_idx[:N_points]].copy()

    def ring_init(N_points):
        """Ring-based initialization: concentric rings with hexagonal spacing."""
        points = [[0.0, 0.0]]
        # Estimate ring radii and counts
        s = np.sqrt(2.0 * np.pi / (3.0 * np.sqrt(3.0) * N_points))
        r = 0.0
        while len(points) < N_points:
            r += s * 0.9
            if r > 1.0 - s / 2:
                break
            n_on_ring = max(1, int(2 * np.pi * r / s))
            for k in range(n_on_ring):
                angle = 2 * np.pi * k / n_on_ring
                points.append([r * np.cos(angle), r * np.sin(angle)])
        points = np.array(points[:N_points])
        return points

    def random_init(N_points):
        """Random initialization with rejection sampling for non-overlap."""
        s = np.sqrt(2.0 * np.pi / (3.0 * np.sqrt(3.0) * N_points))
        min_dist = s * 0.7
        points = []
        attempts = 0
        max_attempts = 50000
        while len(points) < N_points and attempts < max_attempts:
            attempts += 1
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.sqrt(np.random.uniform(0, 1.0 - s / 2))
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            pt = np.array([x, y])
            if len(points) == 0:
                points.append(pt)
                continue
            arr = np.array(points)
            dists = np.linalg.norm(arr - pt, axis=1)
            if np.min(dists) >= min_dist:
                points.append(pt)
        # If we couldn't place enough, fill with hex lattice
        if len(points) < N_points:
            hex_pts = hex_lattice_init(N_points)
            # Replace the last few with hex points
            points = list(points)
            for i in range(len(points), N_points):
                points.append(hex_pts[i])
        return np.array(points[:N_points]).copy()

    def anneal_and_polish(points, max_iters, time_limit):
        """Anneal with bottleneck targeting, then coordinate-descent polish."""
        points = points.copy()
        current_radius = calculate_radius(points)
        best_points = points.copy()
        best_radius = current_radius

        # Phase 1: Annealed bottleneck adjustment
        T0 = 0.005
        T_end = 0.0001
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            T = T0 * (T_end / T0) ** (it / max_iters)

            radius, is_pair, i, j = bottleneck_info(points)

            if is_pair:
                direction = points[i] - points[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    continue
                direction /= norm
                # Try pushing apart
                for step in [0.002, 0.001, 0.0005]:
                    new_points = points.copy()
                    new_points[i] += direction * step
                    new_points[j] -= direction * step
                    # Keep inside circle
                    r_norms = np.linalg.norm(new_points, axis=1)
                    mask = r_norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= r_norms[mask, np.newaxis]
                    new_radius = calculate_radius(new_points)
                    if new_radius > current_radius:
                        points = new_points
                        current_radius = new_radius
                        if current_radius > best_radius:
                            best_radius = current_radius
                            best_points = points.copy()
                        break
                    elif np.random.random() < np.exp((new_radius - current_radius) / T):
                        points = new_points
                        current_radius = new_radius
                        break
            else:
                idx = i
                point = points[idx]
                norm = np.linalg.norm(point)
                if norm < 1e-12:
                    continue
                direction = -point / norm
                for step in [0.002, 0.001, 0.0005]:
                    new_points = points.copy()
                    new_points[idx] += direction * step
                    new_radius = calculate_radius(new_points)
                    if new_radius > current_radius:
                        points = new_points
                        current_radius = new_radius
                        if current_radius > best_radius:
                            best_radius = current_radius
                            best_points = points.copy()
                        break
                    elif np.random.random() < np.exp((new_radius - current_radius) / T):
                        points = new_points
                        current_radius = new_radius
                        break

        # Phase 2: Coordinate-descent polish (direct min-clearance line search)
        points = best_points.copy()
        current_radius = best_radius
        for _ in range(50):
            if time.time() > time_limit:
                break
            radius, is_pair, i, j = bottleneck_info(points)

            if is_pair:
                direction = points[i] - points[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    break
                direction /= norm
                # Line search: maximize min clearance along direction
                best_step = 0.0
                best_r = current_radius
                for step in np.linspace(0.0001, 0.01, 20):
                    new_points = points.copy()
                    new_points[i] += direction * step
                    new_points[j] -= direction * step
                    r_norms = np.linalg.norm(new_points, axis=1)
                    mask = r_norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= r_norms[mask, np.newaxis]
                    new_radius = calculate_radius(new_points)
                    if new_radius > best_r:
                        best_r = new_radius
                        best_step = step
                if best_step > 0:
                    points[i] += direction * best_step
                    points[j] -= direction * best_step
                    r_norms = np.linalg.norm(points, axis=1)
                    mask = r_norms > 1.0
                    if np.any(mask):
                        points[mask] /= r_norms[mask, np.newaxis]
                    current_radius = calculate_radius(points)
            else:
                idx = i
                point = points[idx]
                norm = np.linalg.norm(point)
                if norm < 1e-12:
                    break
                direction = -point / norm
                best_step = 0.0
                best_r = current_radius
                for step in np.linspace(0.0001, 0.01, 20):
                    new_points = points.copy()
                    new_points[idx] += direction * step
                    new_radius = calculate_radius(new_points)
                    if new_radius > best_r:
                        best_r = new_radius
                        best_step = step
                if best_step > 0:
                    points[idx] += direction * best_step
                    current_radius = calculate_radius(points)

            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()

        return best_points, best_radius

    # Multi-start
    inits = [
        hex_lattice_init(N),
        ring_init(N),
        random_init(N),
    ]

    best_overall = None
    best_overall_radius = 0.0

    # Budget allocation: give more time to better starts
    time_per_start = (deadline - start_time) / len(inits)
    min_time_per_start = 15.0

    for init_idx, init_points in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        alloc = min(time_per_start, remaining - 5.0)
        if alloc < min_time_per_start and init_idx > 0:
            break
        t_limit = time.time() + alloc

        # Number of anneal iterations proportional to time
        max_iters = max(100, int(alloc * 20))
        pts, rad = anneal_and_polish(init_points, max_iters, t_limit)

        if rad > best_overall_radius:
            best_overall_radius = rad
            best_overall = pts.copy()

    # Final polish with remaining time
    if best_overall is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 5.0:
            max_iters = max(50, int(remaining * 10))
            pts, rad = anneal_and_polish(best_overall, max_iters, time.time() + remaining)
            if rad > best_overall_radius:
                best_overall_radius = rad
                best_overall = pts

    # Ensure validity
    if best_overall is None:
        best_overall = hex_lattice_init(N)

    r_norms = np.linalg.norm(best_overall, axis=1)
    mask = r_norms > 1.0
    if np.any(mask):
        best_overall[mask] /= r_norms[mask, np.newaxis]

    return best_overall
# EVOLVE-BLOCK-END
