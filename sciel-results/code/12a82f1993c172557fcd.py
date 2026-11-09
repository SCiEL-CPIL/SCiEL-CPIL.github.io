import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill

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

    # Local search refinement with multiple restarts and simulated annealing
    if time.time() - start_time < DEADLINE - 5:
        random.seed(42)
        # Use a set for fast occupancy checks
        occ = set(map(tuple, current))

        # Precompute all valid positions within bounds
        all_positions = set()
        for a in range(-64, 65):
            for b in range(-64, 65):
                all_positions.add((a, b))

        # Extended move set: 6 neighbors + 6 distance-2 + 6 distance-3
        extended_dirs = list(dirs)
        for da, db in dirs:
            extended_dirs.append((2*da, 2*db))
        for da, db in dirs:
            extended_dirs.append((3*da, 3*db))
        # Add some diagonal-ish moves
        extended_dirs.extend([(2, 1), (1, 2), (-2, -1), (-1, -2), (2, -1), (-2, 1),
                              (1, 2), (-1, -2), (1, -2), (-1, 2)])

        # Simulated annealing parameters
        T_start = 2.0
        T_end = 0.01
        cooling_rate = 0.995
        T = T_start
        no_improve = 0
        max_no_improve = 100

        for it in range(10000):
            if time.time() - start_time > DEADLINE - 5:
                break
            if no_improve > max_no_improve:
                # Restart with current best
                current = [list(p) for p in best_points]
                current_edges = count_edges(current)
                occ = set(map(tuple, current))
                T = T_start
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.5)
                continue

            # Randomly pick a point to try moving
            i = random.randint(0, 31)
            ox, oy = current[i][0], current[i][1]
            occ.discard((ox, oy))

            # Try extended neighborhood
            candidates = [(ox, oy)]
            for da, db in extended_dirs:
                candidates.append((ox + da, oy + db))
            # Add a few random far jumps
            for _ in range(3):
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                candidates.append((ox + dx, oy + dy))

            best_candidate = None
            best_candidate_edges = current_edges

            for nx, ny in candidates:
                if abs(nx) > 64 or abs(ny) > 64:
                    continue
                if (nx, ny) in occ:
                    continue
                # Check distance to all others
                valid = True
                for kk in range(32):
                    if kk == i:
                        continue
                    if dist_sq([nx, ny], current[kk]) < 1:
                        valid = False
                        break
                if not valid:
                    continue

                # Compute new edge count incrementally
                old_incident = 0
                for kk in range(32):
                    if kk == i:
                        continue
                    if dist_sq([ox, oy], current[kk]) == 1:
                        old_incident += 1
                new_incident = 0
                for kk in range(32):
                    if kk == i:
                        continue
                    if dist_sq([nx, ny], current[kk]) == 1:
                        new_incident += 1

                new_edges = current_edges - old_incident + new_incident
                delta_e = new_edges - current_edges

                if delta_e > 0:
                    if new_edges > best_candidate_edges:
                        best_candidate_edges = new_edges
                        best_candidate = (nx, ny)
                else:
                    # Accept worse moves with probability exp(delta_e / T)
                    if random.random() < math.exp(delta_e / max(T, 1e-10)):
                        if new_edges > best_candidate_edges:
                            best_candidate_edges = new_edges
                            best_candidate = (nx, ny)

            if best_candidate is not None:
                nx, ny = best_candidate
                current[i][0] = nx
                current[i][1] = ny
                occ.add((nx, ny))
                if best_candidate_edges > current_edges:
                    no_improve = 0
                else:
                    no_improve += 1
                current_edges = best_candidate_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = [list(p) for p in current]
                    no_improve = 0
            else:
                occ.add((ox, oy))
                no_improve += 1

            T *= cooling_rate

    if best_points is None:
        return [[a, b] for a in range(16) for b in range(2)]

    return best_points
