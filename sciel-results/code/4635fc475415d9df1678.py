"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)


def _soft_min_gradient(pos, t=100.0):
    """
    Compute gradient of the soft-min of all pairwise and boundary distances.
    We want to maximize the minimum distance, so we use gradient ascent.
    Returns (gradient, soft_min_value, actual_min).
    """
    n = len(pos)
    pos = pos.reshape(n, 2)

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    boundary_dists = 1.0 - np.linalg.norm(pos, axis=1)

    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]

    all_dists = np.concatenate([pair_dists, boundary_dists])

    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    grad = np.zeros_like(pos)

    # Pairwise gradient
    pair_weights = exp_terms[:len(pair_dists)] / S
    diff_pairs = diff[iu]
    norm_pairs = pair_dists[:, np.newaxis]
    norm_pairs[norm_pairs < 1e-12] = 1e-12
    unit_vecs = diff_pairs / norm_pairs

    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)

    # Boundary gradient
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms = np.linalg.norm(pos, axis=1)[:, np.newaxis]
    norms[norms < 1e-12] = 1e-12
    unit_pos = pos / norms
    grad += boundary_weights[:, np.newaxis] * unit_pos

    soft_min_val = (1.0 / t) * np.log(S) + d_min
    return grad, soft_min_val, d_min


def _local_optimize(points, max_iter=600, lr=0.004, deadline=None):
    """
    Local optimization using gradient ascent on the soft-min of distances.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)

    t = 100.0
    current_lr = lr
    patience = 0
    max_patience = 60

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        grad, soft_min_val, actual_min = _soft_min_gradient(best_points, t)

        new_points = best_points + current_lr * grad

        # Keep points inside unit circle
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis]

        score = _compute_score(new_points)

        if score > best_score:
            best_score = score
            best_points = new_points.copy()
            patience = 0
            current_lr = min(current_lr * 1.015, 0.015)
        else:
            patience += 1
            current_lr *= 0.98

        if patience >= max_patience:
            break

    return best_points


def _radial_ring_init():
    """
    Place circles in concentric rings with optimized radial distances.
    Each ring's radius is chosen so that the angular spacing between
    adjacent circles on that ring is at least the target spacing.
    """
    best_points = None
    best_score = -np.inf

    # Try different numbers of rings
    for n_rings in range(2, 15):
        # Distribute N circles among rings based on circumference
        # Ring i has radius r_i, circumference ~ 2*pi*r_i
        # We want to maximize the minimum angular spacing * r_i (arc length)

        # Start with uniform radial spacing, then optimize
        radii = np.linspace(0.05, 0.95, n_rings)

        # Distribute circles proportionally to circumference
        circumferences = 2 * np.pi * radii
        total_circ = np.sum(circumferences)
        counts = np.round(N * circumferences / total_circ).astype(int)

        # Ensure we have exactly N circles
        diff = N - np.sum(counts)
        # Adjust the largest ring
        counts[np.argmax(counts)] += diff
        counts = np.maximum(counts, 1)
        # Re-normalize if needed
        while np.sum(counts) > N:
            counts[np.argmax(counts)] -= 1
        while np.sum(counts) < N:
            counts[np.argmin(counts)] += 1

        # Now optimize radial positions
        # For each ring, the angular spacing is 2*pi/count_i
        # The chord distance between adjacent circles is 2*r_i*sin(pi/count_i)
        # We want this to be at least the target spacing
        # Also need to check inter-ring distances

        # Simple optimization: adjust radii to equalize spacing
        for opt_iter in range(50):
            # Compute current spacing
            spacings = []
            for i in range(n_rings):
                if counts[i] > 1:
                    chord = 2 * radii[i] * np.sin(np.pi / counts[i])
                    spacings.append(chord)
                else:
                    spacings.append(np.inf)

            # Inter-ring spacing: distance between ring i and ring i+1
            for i in range(n_rings - 1):
                dr = radii[i + 1] - radii[i]
                spacings.append(dr)

            if len(spacings) == 0:
                break

            min_spacing = np.min(spacings)

            # Adjust radii to increase the minimum spacing
            # Simple approach: move the ring with the smallest adjacent spacing
            # outward or inward depending on context
            for i in range(n_rings):
                if i == 0:
                    # Innermost ring: can only move outward
                    if counts[i] > 1:
                        chord = 2 * radii[i] * np.sin(np.pi / counts[i])
                        if chord < min_spacing * 1.1:
                            radii[i] += 0.005
                elif i == n_rings - 1:
                    # Outermost ring: check boundary
                    boundary_dist = 1.0 - radii[i]
                    if boundary_dist < min_spacing * 0.5:
                        radii[i] -= 0.005
                    # Also check inter-ring
                    if i > 0:
                        dr = radii[i] - radii[i-1]
                        if dr < min_spacing * 0.5:
                            radii[i] += 0.005
                else:
                    # Middle rings
                    dr_below = radii[i] - radii[i-1]
                    dr_above = radii[i+1] - radii[i]
                    if counts[i] > 1:
                        chord = 2 * radii[i] * np.sin(np.pi / counts[i])
                    else:
                        chord = np.inf

                    local_min = min(dr_below, dr_above, chord)
                    if local_min < min_spacing * 1.0:
                        # Move to increase the smallest
                        if dr_below < dr_above and dr_below < chord:
                            radii[i] -= 0.003
                        elif dr_above < dr_below and dr_above < chord:
                            radii[i] += 0.003
                        else:
                            # Chord is smallest, increase radius
                            radii[i] += 0.003

            # Keep radii within bounds
            radii[0] = max(radii[0], 0.01)
            radii[-1] = min(radii[-1], 0.98)
            # Ensure monotonic
            for i in range(1, n_rings):
                if radii[i] <= radii[i-1]:
                    radii[i] = radii[i-1] + 0.02

        # Generate points
        points = []
        for i in range(n_rings):
            count = counts[i]
            if count <= 0:
                continue
            r_i = radii[i]
            # Angular offset to stagger rings
            offset = (i * np.pi) / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append((r_i * np.cos(a), r_i * np.sin(a)))

        if len(points) != N:
            continue

        points = np.array(points)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]

    if len(points) < N:
        return None

    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s safety margin

    best_points = None
    best_score = -np.inf

    inits = []

    # 1. Radial ring initialization (primary strategy)
    ring_pts = _radial_ring_init()
    if ring_pts is not None:
        score = _compute_score(ring_pts)
        inits.append((score, ring_pts))

    # 2. Hexagonal lattice with various spacings and angles (backup)
    for angle_deg in [0, 30, 60]:
        for spacing_factor in [0.97, 0.99, 1.01, 1.03]:
            spacing = 2.0 * spacing_factor / (np.sqrt(N) + 1.0)
            pts = _hex_lattice_init(spacing, angle_deg)
            if pts is not None:
                score = _compute_score(pts)
                inits.append((score, pts))

    # Sort by score, keep top few
    inits.sort(key=lambda x: -x[0])
    top_inits = inits[:4] if len(inits) >= 4 else inits

    # Local optimization on the best initializations
    for score, pts in top_inits:
        if time.time() > deadline - 25:
            break

        optimized = _local_optimize(pts, max_iter=500, lr=0.004, deadline=deadline - 15)
        opt_score = _compute_score(optimized)

        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()

    # If we have time, do a final polish on the best
    if best_points is not None and time.time() < deadline - 12:
        optimized = _local_optimize(best_points, max_iter=300, lr=0.002, deadline=deadline - 5)
        opt_score = _compute_score(optimized)
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()

    # Final validation and fallback
    if best_points is None:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        pts = _hex_lattice_init(spacing, 0)
        if pts is not None:
            best_points = pts
        else:
            angles = np.linspace(0, 2 * np.pi * N, N)
            radii = np.linspace(0, 0.95, N)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.9999

    return best_points
# EVOLVE-BLOCK-END
