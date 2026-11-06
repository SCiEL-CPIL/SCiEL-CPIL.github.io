"""Optimized: Precomputed area indices + multi-target local search with incremental updates."""
import time
import numpy as np
import random
from itertools import combinations
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin before 60s limit

    # Precompute all triple indices once
    indices = list(combinations(range(30), 3))
    idx_arr = np.array(indices, dtype=np.int64)
    I, J, K = idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]
    n_triples = len(indices)

    def fast_evaluate(points):
        """Compute max multiplicity of any positive doubled area. Returns (score, target_area)."""
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        areas = np.abs(x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J]))
        positive = areas[areas > 0]
        if len(positive) == 0:
            return 0, 2
        if positive.max() > 50000:
            _, counts = np.unique(positive, return_counts=True)
            max_idx = np.argmax(counts)
            return int(counts[max_idx]), int(positive[max_idx])
        counts = np.bincount(positive.astype(np.int64))
        max_idx = int(np.argmax(counts))
        return int(counts[max_idx]), max_idx

    def evaluate_multi(points):
        """Return top-3 (score, area) pairs for multi-target search."""
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        areas = np.abs(x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J]))
        positive = areas[areas > 0]
        if len(positive) == 0:
            return [(0, 2), (0, 2), (0, 2)]
        if positive.max() > 50000:
            vals, counts = np.unique(positive, return_counts=True)
            order = np.argsort(-counts)
            top = [(int(counts[i]), int(vals[i])) for i in order[:3]]
        else:
            counts = np.bincount(positive.astype(np.int64))
            order = np.argsort(-counts)
            top = [(int(counts[i]), int(i)) for i in order[:3]]
        while len(top) < 3:
            top.append((0, 2))
        return top

    def count_area(points, target_A):
        """Count triples with |det| == target_A."""
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        areas = np.abs(x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J]))
        return int(np.sum(areas == target_A))

    def center_points(pts):
        """Center points to origin, return centered list if within bounds."""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        centered = [(p[0] - cx, p[1] - cy) for p in pts]
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in centered):
            return centered
        return None

    # Precompute which triples include each point
    triples_with = [[] for _ in range(30)]
    for t, (i, j, k) in enumerate(indices):
        triples_with[i].append(t)
        triples_with[j].append(t)
        triples_with[k].append(t)
    # Convert to numpy arrays for fast indexing
    triples_with_arr = [np.array(t, dtype=np.int64) for t in triples_with]

    # Precompute I, J, K subsets for each point index (for incremental updates)
    # For point p, the triples are those where p is I, J, or K
    # We need to recompute area for all triples containing point p
    # The formula: |x_i*(y_j-y_k) + x_j*(y_k-y_i) + x_k*(y_i-y_j)|
    # If we change point p, only triples containing p change

    # --- Phase 1: Generate diverse initial candidates ---
    best_points = None
    best_score = 0
    best_target_A = 2

    # Lattice bases: (u, v) vectors - expanded set with more diverse geometries
    bases = [
        ((1, 0), (0, 1)),
        ((1, 0), (0, 2)),
        ((2, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((1, 0), (2, 1)),
        ((1, 1), (1, -1)),
        ((1, 0), (1, 2)),
        ((2, 1), (1, 0)),
        ((1, 2), (1, -1)),
        ((2, 0), (0, 2)),
        ((1, 1), (2, -1)),
        ((3, 1), (1, 1)),
        ((1, 0), (3, 1)),
        ((2, 1), (1, 2)),
        ((1, 0), (0, 3)),
        ((3, 0), (0, 1)),
        ((1, 1), (0, 2)),
        ((0, 2), (1, 1)),
        ((2, 1), (1, 3)),
        ((1, 3), (2, 1)),
        # Additional diverse bases
        ((1, 1), (1, 2)),
        ((2, 1), (3, 1)),
        ((1, 2), (2, 3)),
        ((3, 1), (1, 2)),
        ((1, 1), (3, 2)),
        ((2, 2), (1, 0)),
        ((1, 0), (2, 2)),
        ((2, 1), (1, 1)),
        ((1, 3), (3, 1)),
        ((3, 2), (2, 1)),
    ]

    dims = [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2)]

    candidates = []
    for u, v in bases:
        if time.time() > deadline - 20:
            break
        for M, N in dims:
            if M * N != 30:
                continue
            pts = []
            for i in range(M):
                for j in range(N):
                    px = i * u[0] + j * v[0]
                    py = i * u[1] + j * v[1]
                    pts.append((px, py))
            if len(pts) != 30 or len(set(pts)) < 30:
                continue
            centered = center_points(pts)
            if centered is None:
                continue
            candidates.append(centered)

    # Also add some random initial points with different seeds
    random.seed(42)
    for _ in range(20):
        pts = set()
        while len(pts) < 30:
            pts.add((random.randint(-20, 20), random.randint(-20, 20)))
        candidates.append(list(pts))

    # Add structured random: points on circles/ellipses
    for radius in [5, 8, 10, 12]:
        pts = set()
        for angle in np.linspace(0, 2*np.pi, 30, endpoint=False):
            px = int(round(radius * np.cos(angle)))
            py = int(round(radius * np.sin(angle)))
            pts.add((px, py))
        if len(pts) == 30:
            candidates.append(list(pts))

    # Add points on two concentric circles
    for r1, r2 in [(4, 8), (5, 10), (6, 12)]:
        pts = set()
        for angle in np.linspace(0, 2*np.pi, 15, endpoint=False):
            pts.add((int(round(r1 * np.cos(angle))), int(round(r1 * np.sin(angle)))))
        for angle in np.linspace(0, 2*np.pi, 15, endpoint=False):
            pts.add((int(round(r2 * np.cos(angle))), int(round(r2 * np.sin(angle)))))
        if len(pts) == 30:
            candidates.append(list(pts))

    # Evaluate all candidates
    for pts in candidates:
        if time.time() > deadline - 12:
            break
        score, target_A = fast_evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
            best_target_A = target_A

    if best_points is None:
        best_points = [(i, j) for i in range(5) for j in range(6)]
        best_points = [(p[0] - 2, p[1] - 2) for p in best_points]
        best_score, best_target_A = fast_evaluate(best_points)

    # --- Phase 2: Multi-target local search with incremental area updates ---
    current_points = list(best_points)
    current_score = best_score
    current_target_A = best_target_A

    best_overall_points = list(current_points)
    best_overall_score = current_score

    # Build current areas array
    pts_arr = np.array(current_points, dtype=np.int64)
    x, y = pts_arr[:, 0], pts_arr[:, 1]
    current_areas = np.abs(x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J]))

    # Get top-3 target areas for multi-target search
    top_targets = evaluate_multi(current_points)

    small_nb = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]
    med_nb = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    large_nb = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
    xl_nb = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if (dx, dy) != (0, 0)]
    xxl_nb = [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if (dx, dy) != (0, 0)]

    max_restarts = 12
    restart = 0

    while time.time() < deadline - 2 and restart < max_restarts:
        no_improve = 0
        max_no_improve = 4000

        # Update top targets periodically
        if restart == 0 or no_improve == 0:
            top_targets = evaluate_multi(current_points)

        while time.time() < deadline - 2:
            if no_improve > max_no_improve:
                break

            # Adaptive neighborhood size
            if no_improve < 200:
                nb = small_nb
            elif no_improve < 600:
                nb = med_nb
            elif no_improve < 1200:
                nb = large_nb
            elif no_improve < 2000:
                nb = xl_nb
            else:
                nb = xxl_nb

            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            dx, dy = random.choice(nb)
            new_x, new_y = old_x + dx, old_y + dy

            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                no_improve += 1
                continue

            new_points = list(current_points)
            new_points[idx] = (new_x, new_y)
            if len(set(new_points)) < 30:
                no_improve += 1
                continue

            # Incremental update: only recompute triples involving point idx
            t_indices = triples_with_arr[idx]
            new_pts_arr = np.array(new_points, dtype=np.int64)
            nx, ny = new_pts_arr[:, 0], new_pts_arr[:, 1]
            new_areas = current_areas.copy()
            ti = t_indices
            new_areas[ti] = np.abs(
                nx[I[ti]] * (ny[J[ti]] - ny[K[ti]]) +
                nx[J[ti]] * (ny[K[ti]] - ny[I[ti]]) +
                nx[K[ti]] * (ny[I[ti]] - ny[J[ti]])
            )

            # Quick check: count each target area, use best
            best_tc = 0
            best_ta = current_target_A
            for score_t, area_t in top_targets:
                tc = int(np.sum(new_areas == area_t))
                if tc > best_tc:
                    best_tc = tc
                    best_ta = area_t

            if best_tc >= current_score - 2:
                # Full evaluation
                positive = new_areas[new_areas > 0]
                if len(positive) == 0:
                    score, target_A = 0, 2
                elif positive.max() > 50000:
                    _, counts = np.unique(positive, return_counts=True)
                    max_idx = np.argmax(counts)
                    score, target_A = int(counts[max_idx]), int(positive[max_idx])
                else:
                    counts = np.bincount(positive.astype(np.int64))
                    max_idx = int(np.argmax(counts))
                    score, target_A = int(counts[max_idx]), max_idx

                if score > current_score:
                    current_score = score
                    current_points = new_points
                    current_target_A = target_A
                    current_areas = new_areas
                    no_improve = 0
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_overall_points = list(current_points)
                    # Refresh top targets after improvement
                    top_targets = evaluate_multi(current_points)
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # Restart with perturbation
        if restart < max_restarts - 1:
            restart += 1
            current_points = list(best_overall_points)
            current_score = best_overall_score
            current_target_A = best_target_A
            # Rebuild areas array
            pts_arr = np.array(current_points, dtype=np.int64)
            x, y = pts_arr[:, 0], pts_arr[:, 1]
            current_areas = np.abs(x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J]))
            # Stronger perturbation: 15 random moves
            for _ in range(15):
                idx = random.randint(0, 29)
                old_x, old_y = current_points[idx]
                dx, dy = random.choice(xxl_nb)
                new_x, new_y = old_x + dx, old_y + dy
                if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                    new_pts = list(current_points)
                    new_pts[idx] = (new_x, new_y)
                    if len(set(new_pts)) == 30:
                        current_points = new_pts
                        # Update areas incrementally
                        t_indices = triples_with_arr[idx]
                        npa = np.array(current_points, dtype=np.int64)
                        nx, ny = npa[:, 0], npa[:, 1]
                        current_areas[t_indices] = np.abs(
                            nx[I[t_indices]] * (ny[J[t_indices]] - ny[K[t_indices]]) +
                            nx[J[t_indices]] * (ny[K[t_indices]] - ny[I[t_indices]]) +
                            nx[K[t_indices]] * (ny[I[t_indices]] - ny[J[t_indices]])
                        )
            # Re-evaluate after perturbation
            current_score, current_target_A = fast_evaluate(current_points)
            top_targets = evaluate_multi(current_points)

    # Final polish: try small moves on best solution
    if time.time() < deadline - 1:
        final_points = list(best_overall_points)
        final_score = best_overall_score
        pts_arr = np.array(final_points, dtype=np.int64)
        x, y = pts_arr[:, 0], pts_arr[:, 1]
        final_areas = np.abs(x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J]))

        for _ in range(500):
            if time.time() > deadline - 1:
                break
            idx = random.randint(0, 29)
            old_x, old_y = final_points[idx]
            dx, dy = random.choice(small_nb)
            new_x, new_y = old_x + dx, old_y + dy
            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                continue
            new_points = list(final_points)
            new_points[idx] = (new_x, new_y)
            if len(set(new_points)) < 30:
                continue
            t_indices = triples_with_arr[idx]
            new_pts_arr = np.array(new_points, dtype=np.int64)
            nx, ny = new_pts_arr[:, 0], new_pts_arr[:, 1]
            new_areas = final_areas.copy()
            ti = t_indices
            new_areas[ti] = np.abs(
                nx[I[ti]] * (ny[J[ti]] - ny[K[ti]]) +
                nx[J[ti]] * (ny[K[ti]] - ny[I[ti]]) +
                nx[K[ti]] * (ny[I[ti]] - ny[J[ti]])
            )
            positive = new_areas[new_areas > 0]
            if len(positive) == 0:
                score, target_A = 0, 2
            elif positive.max() > 50000:
                _, counts = np.unique(positive, return_counts=True)
                max_idx = np.argmax(counts)
                score, target_A = int(counts[max_idx]), int(positive[max_idx])
            else:
                counts = np.bincount(positive.astype(np.int64))
                max_idx = int(np.argmax(counts))
                score, target_A = int(counts[max_idx]), max_idx
            if score > final_score:
                final_score = score
                final_points = new_points
                final_areas = new_areas
                if final_score > best_overall_score:
                    best_overall_score = final_score
                    best_overall_points = list(final_points)

    return best_overall_points
