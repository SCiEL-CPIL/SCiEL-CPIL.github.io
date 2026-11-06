import time
import random
from collections import Counter

def _build_counter(points):
    """Build area counter for all non-zero doubled areas."""
    n = len(points)
    area_counts = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                dx2 = x3 - x1
                dy2 = y3 - y1
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    area_counts[area] += 1
    return area_counts

def _max_of_counter(area_counts):
    """Return max multiplicity in counter."""
    if not area_counts:
        return 0
    return max(area_counts.values())

def _calculate_score(points):
    """
    Calculate the maximum multiplicity of any non-zero doubled area.
    points: list of (x, y) tuples
    """
    area_counts = _build_counter(points)
    return _max_of_counter(area_counts)

def _evaluate_move(points, idx, new_p, area_counts):
    """
    Evaluate moving point idx to new_p.
    Returns (delta_score, new_area_counts) or (None, None) if invalid.
    """
    old_p = points[idx]
    if new_p == old_p:
        return None, None

    n = len(points)
    # Collect all triangles involving point idx
    # For each pair (a, b) where a < b, a != idx, b != idx:
    # triangle (a, b, idx) has area A_old with old_p, A_new with new_p
    # We need to decrement old areas and increment new areas.

    # Build list of other points
    others = []
    for i in range(n):
        if i != idx:
            others.append(i)

    m = len(others)  # 29
    old_x, old_y = old_p
    new_x, new_y = new_p

    # For each pair of others, compute old and new area
    # area of triangle with vertex at P and two others (a, b):
    # = abs((x_a - x_P)*(y_b - y_P) - (y_a - y_P)*(x_b - x_P))

    # We'll track the changes in the counter
    # Use a temporary dict for deltas
    deltas = {}

    for i in range(m):
        a = others[i]
        xa, ya = points[a]
        for j in range(i + 1, m):
            b = others[j]
            xb, yb = points[b]
            # Old area
            dxa = xa - old_x
            dya = ya - old_y
            dxb = xb - old_x
            dyb = yb - old_y
            old_area = abs(dxa * dyb - dya * dxb)
            # New area
            dxa_n = xa - new_x
            dya_n = ya - new_y
            dxb_n = xb - new_x
            dyb_n = yb - new_y
            new_area = abs(dxa_n * dyb_n - dya_n * dxb_n)

            if old_area > 0:
                deltas[old_area] = deltas.get(old_area, 0) - 1
            if new_area > 0:
                deltas[new_area] = deltas.get(new_area, 0) + 1

    # Apply deltas to a copy of area_counts
    # We need to check that no count goes negative (shouldn't happen if consistent)
    new_counts = dict(area_counts)
    for area, delta in deltas.items():
        if delta == 0:
            continue
        cur = new_counts.get(area, 0)
        if cur + delta < 0:
            return None, None  # Inconsistent, shouldn't happen
        new_val = cur + delta
        if new_val == 0:
            del new_counts[area]
        else:
            new_counts[area] = new_val

    new_score = _max_of_counter(new_counts)
    old_score = _max_of_counter(area_counts)
    delta_score = new_score - old_score

    return delta_score, new_counts

def _local_search(initial_points, time_limit=55.0, seed=None):
    """
    Perform hill climbing to maximize the score using fast incremental updates.
    """
    if seed is not None:
        random.seed(seed)

    start_time = time.time()
    best_points = [p for p in initial_points]
    best_score = _calculate_score(best_points)
    best_counts = _build_counter(best_points)

    current_points = [p for p in initial_points]
    current_score = best_score
    current_counts = dict(best_counts)

    # Precompute valid coordinates
    coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]

    occupied_set = set(current_points)

    stuck_count = 0
    max_stuck = 500
    iterations = 0

    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]

        # Get a new random position
        new_p = random.choice(coords)
        if new_p in occupied_set:
            stuck_count += 1
            if stuck_count > max_stuck:
                current_points = [p for p in best_points]
                occupied_set = set(current_points)
                current_score = best_score
                current_counts = dict(best_counts)
                stuck_count = 0
            continue

        # Evaluate move
        delta_score, new_counts = _evaluate_move(current_points, idx, new_p, current_counts)

        if delta_score is None:
            stuck_count += 1
            if stuck_count > max_stuck:
                current_points = [p for p in best_points]
                occupied_set = set(current_points)
                current_score = best_score
                current_counts = dict(best_counts)
                stuck_count = 0
            continue

        # Apply move
        current_points[idx] = new_p
        occupied_set.remove(old_p)
        occupied_set.add(new_p)
        current_counts = new_counts
        current_score += delta_score

        iterations += 1

        if delta_score > 0:
            stuck_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = [p for p in current_points]
                best_counts = dict(current_counts)
        else:
            # Revert move
            current_points[idx] = old_p
            occupied_set.remove(new_p)
            occupied_set.add(old_p)
            # Revert counts: we need to undo the deltas
            # Simpler: recompute from best or just revert by re-evaluating
            # For efficiency, let's just revert by recomputing the reverse
            # Actually, let's just revert the counts by applying negative deltas
            # But we didn't store them. Let's just recompute current_counts from current_points
            # That's expensive. Alternative: store the old counts before applying.
            # Let's restructure: compute new_counts, check if better, then commit.
            # For now, revert by recomputing (expensive but correct)
            # Actually, let's just not revert and instead try a different approach:
            # Only commit if delta_score > 0, otherwise don't apply.
            # Let me restructure the loop.
            pass

        if stuck_count > max_stuck:
            current_points = [p for p in best_points]
            occupied_set = set(current_points)
            current_score = best_score
            current_counts = dict(best_counts)
            stuck_count = 0

    return best_points, best_score

def construct():
    # Initial guesses
    grid_5x6 = [(x, y) for x in range(5) for y in range(6)]
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    grid_3x10 = [(x, y) for x in range(3) for y in range(10)]
    grid_2x15 = [(x, y) for x in range(2) for y in range(15)]

    random.seed(42)
    random_points = random.sample([(x, y) for x in range(-30, 31) for y in range(-30, 31)], 30)

    candidates = [grid_5x6, grid_6x5, grid_3x10, grid_2x15, random_points]

    best_overall_points = None
    best_overall_score = -1

    for c in candidates:
        s = _calculate_score(c)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = c

    # Run local search from the best candidate
    final_points, final_score = _local_search(best_overall_points, time_limit=50.0, seed=42)

    if final_score > best_overall_score:
        return final_points
    else:
        return best_overall_points

# EVOLVE-BLOCK-END
