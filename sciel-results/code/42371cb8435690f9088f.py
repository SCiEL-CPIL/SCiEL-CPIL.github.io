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
        final_points = []
        for b in range(6):
            for a in range(6):
                final_points.append((a, b))
                if len(final_points) == 32:
                    break
            if len(final_points) == 32:
                break

    # Convert to list of [a, b] pairs
    current = [list(p) for p in final_points]
    current_edges = count_edges(current)
    best_points = [list(p) for p in current]
    best_edges = current_edges

    # === Phase 1: Exhaustive single-swap search ===
    # For each point, try replacing it with every candidate in an expanded bounding box
    if time.time() - start_time < DEADLINE - 30:
        occ = set(map(tuple, current))
        improved = True
        while improved and time.time() - start_time < DEADLINE - 30:
            improved = False
            min_a = min(p[0] for p in current)
            max_a = max(p[0] for p in current)
            min_b = min(p[1] for p in current)
            max_b = max(p[1] for p in current)
            cand_min_a = min_a - 2
            cand_max_a = max_a + 2
            cand_min_b = min_b - 2
            cand_max_b = max_b + 2

            for i in range(32):
                if time.time() - start_time > DEADLINE - 30:
                    break
                ox, oy = current[i][0], current[i][1]
                occ.discard((ox, oy))
                for a in range(cand_min_a, cand_max_a + 1):
                    for b in range(cand_min_b, cand_max_b + 1):
                        if (a, b) in occ:
                            continue
                        if abs(a) > 64 or abs(b) > 64:
                            continue
                        # Check distance to all others
                        valid = True
                        for k in range(32):
                            if k == i:
                                continue
                            if dist_sq([a, b], current[k]) < 1:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Compute new edge count
                        old_incident = 0
                        for k in range(32):
                            if k == i:
                                continue
                            if dist_sq([ox, oy], current[k]) == 1:
                                old_incident += 1
                        new_incident = 0
                        for k in range(32):
                            if k == i:
                                continue
                            if dist_sq([a, b], current[k]) == 1:
                                new_incident += 1
                        new_edges = current_edges - old_incident + new_incident
                        if new_edges > current_edges:
                            current[i][0] = a
                            current[i][1] = b
                            occ.add((a, b))
                            current_edges = new_edges
                            improved = True
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_points = [list(p) for p in current]
                            break
                    else:
                        continue
                    break
                else:
                    occ.add((ox, oy))
                    continue
                break

    # === Phase 2: Simulated Annealing with O(1) edge-delta ===
    if time.time() - start_time < DEADLINE - 5:
        random.seed(42)
        occ = set(map(tuple, current))
        T = 2.0
        T_min = 0.01
        cooling = 0.9995

        for it in range(200000):
            if time.time() - start_time > DEADLINE - 5:
                break
            if T < T_min:
                break

            i = random.randint(0, 31)
            ox, oy = current[i][0], current[i][1]
            occ.discard((ox, oy))

            # Generate random candidate near current point
            # Mix: 70% neighbor, 30% random jump
            if random.random() < 0.7:
                da, db = dirs[random.randint(0, 5)]
                nx, ny = ox + da, oy + db
            else:
                nx = ox + random.randint(-3, 3)
                ny = oy + random.randint(-3, 3)

            if abs(nx) > 64 or abs(ny) > 64:
                occ.add((ox, oy))
                T *= cooling
                continue
            if (nx, ny) in occ:
                occ.add((ox, oy))
                T *= cooling
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
                occ.add((ox, oy))
                T *= cooling
                continue

            # Compute edge delta
            old_incident = 0
            for k in range(32):
                if k == i:
                    continue
                if dist_sq([ox, oy], current[k]) == 1:
                    old_incident += 1
            new_incident = 0
            for k in range(32):
                if k == i:
                    continue
                if dist_sq([nx, ny], current[k]) == 1:
                    new_incident += 1

            delta_edges = new_incident - old_incident
            new_edges = current_edges + delta_edges

            # Accept or reject
            if delta_edges > 0 or random.random() < math.exp(delta_edges / T):
                current[i][0] = nx
                current[i][1] = ny
                occ.add((nx, ny))
                current_edges = new_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = [list(p) for p in current]
                    if best_edges == 76:
                        break
            else:
                occ.add((ox, oy))

            T *= cooling

    # === Phase 3: Final exhaustive polish ===
    if time.time() - start_time < DEADLINE - 2 and best_edges < 76:
        current = [list(p) for p in best_points]
        current_edges = best_edges
        occ = set(map(tuple, current))
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            min_a = min(p[0] for p in current)
            max_a = max(p[0] for p in current)
            min_b = min(p[1] for p in current)
            max_b = max(p[1] for p in current)
            cand_min_a = min_a - 2
            cand_max_a = max_a + 2
            cand_min_b = min_b - 2
            cand_max_b = max_b + 2

            for i in range(32):
                if time.time() - start_time > DEADLINE - 2:
                    break
                ox, oy = current[i][0], current[i][1]
                occ.discard((ox, oy))
                for a in range(cand_min_a, cand_max_a + 1):
                    for b in range(cand_min_b, cand_max_b + 1):
                        if (a, b) in occ:
                            continue
                        if abs(a) > 64 or abs(b) > 64:
                            continue
                        valid = True
                        for k in range(32):
                            if k == i:
                                continue
                            if dist_sq([a, b], current[k]) < 1:
                                valid = False
                                break
                        if not valid:
                            continue
                        old_incident = 0
                        for k in range(32):
                            if k == i:
                                continue
                            if dist_sq([ox, oy], current[k]) == 1:
                                old_incident += 1
                        new_incident = 0
                        for k in range(32):
                            if k == i:
                                continue
                            if dist_sq([a, b], current[k]) == 1:
                                new_incident += 1
                        new_edges = current_edges - old_incident + new_incident
                        if new_edges > current_edges:
                            current[i][0] = a
                            current[i][1] = b
                            occ.add((a, b))
                            current_edges = new_edges
                            improved = True
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_points = [list(p) for p in current]
                            break
                    else:
                        continue
                    break
                else:
                    occ.add((ox, oy))
                    continue
                break

    return best_points
