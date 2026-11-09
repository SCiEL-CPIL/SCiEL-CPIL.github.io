import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

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

    def calc_compactness(points):
        # Sum of squared distances to centroid
        n = len(points)
        if n == 0:
            return 0
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        # Convert to Cartesian for true geometric compactness
        # x = a + 0.5*b, y = sqrt(3)/2 * b
        total = 0.0
        for p in points:
            x = p[0] + 0.5 * p[1]
            y = math.sqrt(3) / 2 * p[1]
            cx_c = cx + 0.5 * cy
            cy_c = math.sqrt(3) / 2 * cy
            total += (x - cx_c)**2 + (y - cy_c)**2
        return total

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

    # Base: rings 0, 1, 2 (19 points)
    base_points = []
    for r in range(3):
        base_points.extend(rings[r])
    base_set = set(base_points)

    # Ring 3 has 18 points. We need 13 more.
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
    # We want to maximize edges, so we want high neighbor counts.
    # Also, contiguity helps compactness.
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
    current = [list(p) for p in final_points]
    current_edges = count_edges(current)
    
    # Initialize best
    best_points = [list(p) for p in current]
    best_edges = current_edges
    best_compactness = calc_compactness(current)

    # Local Search Phase 1: Maximize Edges
    # We expect to start at 76. If not, this fixes it.
    random.seed(42)
    occ = set(map(tuple, current))
    
    # Fast local search
    for it in range(5000):
        if time.time() - start_time > DEADLINE - 10:
            break

        i = random.randint(0, 31)
        ox, oy = current[i][0], current[i][1]
        occ.discard((ox, oy))

        candidates = [(ox, oy)]
        for da, db in dirs:
            candidates.append((ox + da, oy + db))
        for da, db in [(2, 0), (0, 2), (-2, 0), (0, -2), (2, -1), (-1, 2),
                       (-2, 1), (1, -2), (2, 1), (-1, -2), (-2, -1), (1, 2)]:
            candidates.append((ox + da, oy + db))

        best_candidate = None
        best_candidate_edges = current_edges
        best_candidate_comp = best_compactness

        for nx, ny in candidates:
            if abs(nx) > 64 or abs(ny) > 64:
                continue
            if (nx, ny) in occ:
                continue
            
            valid = True
            for k in range(32):
                if k == i:
                    continue
                if dist_sq([nx, ny], current[k]) < 1:
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
                if dist_sq([nx, ny], current[k]) == 1:
                    new_incident += 1

            new_edges = current_edges - old_incident + new_incident
            
            if new_edges > best_candidate_edges:
                best_candidate_edges = new_edges
                best_candidate = (nx, ny)
                # Update compactness estimate (approximate)
                best_candidate_comp = best_compactness # Placeholder, will recalc if accepted
            elif new_edges == best_candidate_edges:
                # If edges are same, prefer more compact?
                # Only if we are already at max edges (76)
                if current_edges == 76:
                    # Quick compactness check: distance to centroid
                    # This is expensive, so we only do it if we have spare time
                    # For now, we just accept the first one to save time
                    pass

        if best_candidate is not None:
            nx, ny = best_candidate
            current[i][0] = nx
            current[i][1] = ny
            occ.add((nx, ny))
            current_edges = best_candidate_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best_points = [list(p) for p in current]
                best_compactness = calc_compactness(current)
            elif current_edges == best_edges:
                # If we hit max edges, try to improve compactness
                if current_edges == 76:
                    c = calc_compactness(current)
                    if c < best_compactness:
                        best_compactness = c
                        best_points = [list(p) for p in current]
        else:
            occ.add((ox, oy))

    # Local Search Phase 2: Polish Compactness (if we have time and are at 76 edges)
    if best_edges == 76 and time.time() - start_time < DEADLINE - 5:
        current = [list(p) for p in best_points]
        current_comp = best_compactness
        occ = set(map(tuple, current))
        random.seed(123)
        
        for it in range(3000):
            if time.time() - start_time > DEADLINE - 5:
                break

            i = random.randint(0, 31)
            ox, oy = current[i][0], current[i][1]
            occ.discard((ox, oy))

            candidates = [(ox, oy)]
            for da, db in dirs:
                candidates.append((ox + da, oy + db))
            for da, db in [(2, 0), (0, 2), (-2, 0), (0, -2), (2, -1), (-1, 2),
                           (-2, 1), (1, -2), (2, 1), (-1, -2), (-2, -1), (1, 2)]:
                candidates.append((ox + da, oy + db))

            best_candidate = None
            best_candidate_comp = current_comp
            best_candidate_edges = -1

            for nx, ny in candidates:
                if abs(nx) > 64 or abs(ny) > 64:
                    continue
                if (nx, ny) in occ:
                    continue
                
                valid = True
                for k in range(32):
                    if k == i:
                        continue
                    if dist_sq([nx, ny], current[k]) < 1:
                        valid = False
                        break
                if not valid:
                    continue

                # Check if edges remain 76
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

                new_edges = 76 - old_incident + new_incident
                if new_edges < 76:
                    continue

                # Calculate new compactness
                # Approximate: just move the point and see if it's closer to centroid
                # Full recalc is expensive. Let's do it only for the best candidate.
                # For now, just check if the move is "inward"
                # Centroid is roughly (0,0) for symmetric shapes
                # If dist_sq((nx,ny), (0,0)) < dist_sq((ox,oy), (0,0)), it's better
                old_d = ox*ox + ox*oy + oy*oy
                new_d = nx*nx + nx*ny + ny*ny
                if new_d < old_d:
                    if best_candidate is None or new_d < best_candidate_comp:
                        best_candidate_comp = new_d
                        best_candidate = (nx, ny)
                        best_candidate_edges = new_edges

            if best_candidate is not None:
                nx, ny = best_candidate
                current[i][0] = nx
                current[i][1] = ny
                occ.add((nx, ny))
                # Update best if it's truly more compact
                # We accept the move if it reduces the "radius"
                if best_candidate_edges == 76:
                    # Recalc full compactness to be sure?
                    # It's O(N). 3000 iters * 32 = 100k ops. Fast.
                    c = calc_compactness(current)
                    if c < best_compactness:
                        best_compactness = c
                        best_points = [list(p) for p in current]
            else:
                occ.add((ox, oy))

    return best_points
