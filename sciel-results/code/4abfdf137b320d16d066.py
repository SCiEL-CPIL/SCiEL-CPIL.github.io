import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    def count_edges(points):
        n = len(points)
        edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    # Directions on triangular lattice (distance 1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    # Build hexagonal rings from center
    visited = set()
    rings = {0: [(0, 0)]}
    visited.add((0, 0))
    current_layer = [(0, 0)]

    for r in range(1, 5):
        next_layer = []
        for p in current_layer:
            a, b = p
            for da, db in dirs:
                np = (a + da, b + db)
                if np not in visited:
                    visited.add(np)
                    next_layer.append(np)
        rings[r] = next_layer
        current_layer = next_layer

    # Base: rings 0, 1, 2 (1 + 6 + 12 = 19 points)
    base_points = []
    for r in range(3):
        base_points.extend(rings[r])
    base_set = set(base_points)

    # Ring 3 has 18 points. We need 32 - 19 = 13 more from ring 3.
    ring3 = rings[3]

    def angle(p):
        a, b = p
        x = a + 0.5 * b
        y = math.sqrt(3) / 2 * b
        return math.atan2(y, x)

    ring3_sorted = sorted(ring3, key=angle)
    n_ring = len(ring3_sorted)

    # Count neighbors in base for each ring3 point
    neighbor_counts = {}
    for p in ring3:
        a, b = p
        cnt = 0
        for da, db in dirs:
            if (a + da, b + db) in base_set:
                cnt += 1
        neighbor_counts[p] = cnt

    # Find best contiguous block of 13 in the cycle
    k = 13
    best_score = -1
    best_start = 0
    for i in range(n_ring):
        s = 0
        for j in range(k):
            idx = (i + j) % n_ring
            s += neighbor_counts[ring3_sorted[idx]]
        if s > best_score:
            best_score = s
            best_start = i

    selected_ring3 = []
    for j in range(k):
        idx = (best_start + j) % n_ring
        selected_ring3.append(ring3_sorted[idx])

    final_points = base_points + selected_ring3

    if len(final_points) != 32:
        # Fallback: simple dense cluster
        final_points = []
        for b in range(6):
            for a in range(6):
                final_points.append((a, b))
                if len(final_points) == 32:
                    break
            if len(final_points) == 32:
                break

    current = [list(p) for p in final_points]
    current_edges = count_edges(current)
    best_points = [list(p) for p in current]
    best_edges = current_edges

    # Local search refinement
    if time.time() - start_time < DEADLINE - 5:
        random.seed(42)
        # Use a set for fast occupancy checks
        occ = set(map(tuple, current))

        for it in range(2000):
            if time.time() - start_time > DEADLINE - 5:
                break

            # Randomly pick a point to try moving
            i = random.randint(0, 31)
            ox, oy = current[i][0], current[i][1]
            occ.discard((ox, oy))

            # Try all 6 neighbors plus stay
            candidates = [(ox, oy)]
            for da, db in dirs:
                candidates.append((ox + da, oy + db))
            # Also try a couple of distance-2 jumps for escaping local minima
            for da, db in [(2, 0), (0, 2), (-2, 0), (0, -2), (2, -1), (-1, 2)]:
                candidates.append((ox + da, oy + db))

            best_candidate = None
            best_candidate_edges = current_edges

            for nx, ny in candidates:
                if abs(nx) > 64 or abs(ny) > 64:
                    continue
                if (nx, ny) in occ:
                    continue
                # Check distance to all others
                valid = True
                for k in range(32):
                    if k == i:
                        continue
                    if dist_sq([nx, ny], current[k]) < 1:
                        valid = False
                        break
                if not valid:
                    continue

                # Compute new edge count incrementally
                # Remove edges incident to old position
                old_incident = 0
                for k in range(32):
                    if k == i:
                        continue
                    if dist_sq([ox, oy], current[k]) == 1:
                        old_incident += 1
                # Add edges incident to new position
                new_incident = 0
                for k in range(32):
                    if k == i:
                        continue
                    if dist_sq([nx, ny], current[k]) == 1:
                        new_incident += 1

                new_edges = current_edges - old_incident + new_incident
                if new_edges > best_candidate_edges:
                    best_candidate_edges = new_edges
                    best_candidate = (nx, ny)

            if best_candidate is not None:
                nx, ny = best_candidate
                current[i][0] = nx
                current[i][1] = ny
                occ.add((nx, ny))
                current_edges = best_candidate_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = [list(p) for p in current]
            else:
                occ.add((ox, oy))

    if best_points is None:
        return [[a, b] for a in range(16) for b in range(2)]

    return best_points
