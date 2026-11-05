import time
import random
import math
from collections import Counter
import numpy as np

def construct():
    deadline = time.time() + 75.0
    random.seed(42)
    N = 30

    # Precompute all triplets
    triplets = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                triplets.append((i, j, k))

    # Precompute for each index which triplets it participates in
    idx_triplets = [[] for _ in range(N)]
    for t_idx, (i, j, k) in enumerate(triplets):
        idx_triplets[i].append(t_idx)
        idx_triplets[j].append(t_idx)
        idx_triplets[k].append(t_idx)

    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    def full_count(points):
        areas = []
        for i, j, k in triplets:
            a = doubled_area(points[i], points[j], points[k])
            if a > 0:
                areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count

    def count_target(points, target):
        count = 0
        for i, j, k in triplets:
            if doubled_area(points[i], points[j], points[k]) == target:
                count += 1
        return count

    def generate_hex_config():
        """Generate points from a hexagonal lattice pattern."""
        points = []
        seen = set()
        # Hexagonal lattice: points at (i + j/2, j*sqrt(3)/2) approximated on integer grid
        # Use a simpler approach: points on a triangular lattice
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i + j // 2
                y = j * 3 // 2
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pt = (x, y)
                    if pt not in seen:
                        seen.add(pt)
                        points.append(pt)
        random.shuffle(points)
        return points[:30]

    def generate_triangular_config():
        """Generate points from a triangular lattice."""
        points = []
        seen = set()
        # Triangular lattice with basis vectors (1,0) and (1,1)
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i + j
                y = j
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pt = (x, y)
                    if pt not in seen:
                        seen.add(pt)
                        points.append(pt)
        random.shuffle(points)
        return points[:30]

    def generate_double_lattice_config():
        """Generate 30 points as union of two congruent 15-point triangular lattices shifted by a vector."""
        # First lattice: triangular lattice with basis (1,0), (0,1)
        lattice1 = []
        seen1 = set()
        for i in range(-2, 3):
            for j in range(-2, 3):
                pt = (i, j)
                if pt not in seen1:
                    seen1.add(pt)
                    lattice1.append(pt)
        lattice1 = lattice1[:15]
        
        # Second lattice: same shape shifted by vector (3, 2)
        shift = (3, 2)
        lattice2 = [(x + shift[0], y + shift[1]) for x, y in lattice1]
        
        points = lattice1 + lattice2
        # Ensure distinctness
        seen = set()
        filtered = []
        for p in points:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        return filtered[:30]

    def generate_lattice_direction_config():
        """Generate points from primitive lattice directions with varying magnitudes."""
        directions = [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2), (2,-1), (-1,2)]
        points = []
        seen = set()
        
        for dx, dy in directions:
            for mag in range(1, 8):
                for sign in [1, -1]:
                    x = sign * dx * mag
                    y = sign * dy * mag
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pt = (x, y)
                        if pt not in seen and pt != (0,0):
                            seen.add(pt)
                            points.append(pt)
        
        if (0,0) not in seen:
            seen.add((0,0))
            points.append((0,0))
        
        for step in [(1,0), (0,1), (1,1)]:
            for start in range(-10, 11, 2):
                for t in range(5):
                    x = start + step[0]*t
                    y = 0 + step[1]*t
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pt = (x, y)
                        if pt not in seen:
                            seen.add(pt)
                            points.append(pt)
        
        random.shuffle(points)
        return points[:30]

    def generate_arithmetic_progression_config():
        """Generate points along arithmetic progressions."""
        points = []
        seen = set()
        
        for slope in [0, 1, -1, 2, -2, 1/2, -1/2]:
            for t in range(-6, 7):
                if slope == 0:
                    x, y = t, 0
                elif slope == 1:
                    x, y = t, t
                elif slope == -1:
                    x, y = t, -t
                elif slope == 2:
                    x, y = t, 2*t
                elif slope == -2:
                    x, y = t, -2*t
                elif slope == 0.5:
                    x, y = 2*t, t
                else:
                    x, y = -2*t, t
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pt = (x, y)
                    if pt not in seen:
                        seen.add(pt)
                        points.append(pt)
        
        random.shuffle(points)
        return points[:30]

    def generate_grid_config():
        """Generate a 6x5 grid."""
        return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]

    def generate_dense_config():
        """Generate a dense 5x6 grid centered at origin."""
        return [(x, y) for x in range(-2, 3) for y in range(-3, 3)]

    # Try multiple initial configurations
    initial_configs = [
        generate_double_lattice_config(),
        generate_hex_config(),
        generate_triangular_config(),
        generate_grid_config(),
        generate_lattice_direction_config(),
        generate_arithmetic_progression_config(),
        generate_dense_config(),
        [(x, y) for x in range(-3, 3) for y in range(-3, 3) if abs(x) + abs(y) <= 4][:30],
    ]

    best_initial = None
    best_initial_count = 0

    for config in initial_configs:
        seen = set()
        filtered = []
        for p in config:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        filtered = filtered[:30]

        counter, ba, bc = full_count(filtered)
        if bc > best_initial_count:
            best_initial_count = bc
            best_initial = filtered

    coords = list(best_initial)
    counter, target, current_count = full_count(coords)
    best_overall = current_count
    best_points = list(coords)
    best_target = target

    # Simulated Annealing with target-focused optimization
    T = 15.0
    cooling_rate = 0.985

    no_improve = 0
    max_no_improve = 200
    restarts = 0
    max_restarts = 15

    while time.time() < deadline:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break

            # Re-evaluate current state
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba

            # Switch to the most common area in current configuration
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 50
            T = 10.0

            # Random restart with perturbation
            for _ in range(5):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    current_count = count_target(coords, target)
                    break

            if current_count < best_overall:
                coords = list(best_points)
                target = best_target
                current_count = best_overall
            continue

        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]

        # Generate candidates
        candidates = []
        # Local moves: small perturbations
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    candidates.append(new_pt)
        # Structured moves: snap to lattice directions
        for dir_vec in [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2)]:
            for mag in range(1, 6):
                for sign in [1, -1]:
                    new_pt = (old_pt[0] + sign*dir_vec[0]*mag, old_pt[1] + sign*dir_vec[1]*mag)
                    if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                        candidates.append(new_pt)
        # Random global moves
        for _ in range(8):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt != old_pt and new_pt not in coords:
                candidates.append(new_pt)

        if not candidates:
            no_improve += 1
            T *= cooling_rate
            continue

        best_delta = -999
        best_new_pt = None

        # Precompute old matches for this idx
        old_matches = 0
        for t_idx in idx_triplets[idx]:
            i, j, k = triplets[t_idx]
            if doubled_area(coords[i], coords[j], coords[k]) == target:
                old_matches += 1

        for new_pt in candidates:
            new_matches = 0
            for t_idx in idx_triplets[idx]:
                i, j, k = triplets[t_idx]
                if i == idx:
                    p1, p2, p3 = new_pt, coords[j], coords[k]
                elif j == idx:
                    p1, p2, p3 = coords[i], new_pt, coords[k]
                else:
                    p1, p2, p3 = coords[i], coords[j], new_pt
                if doubled_area(p1, p2, p3) == target:
                    new_matches += 1

            delta = new_matches - old_matches
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt

        if best_new_pt is not None:
            if best_delta > 0:
                accept = True
            else:
                if T > 0.1:
                    accept_prob = math.exp(best_delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False

            if accept:
                coords[idx] = best_new_pt
                current_count += best_delta
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
            else:
                no_improve += 1

        T *= cooling_rate

        # Periodically re-evaluate target to adapt to current configuration
        if no_improve > 50 and time.time() < deadline - 5:
            counter, ba, bc = full_count(coords)
            if bc > current_count:
                target = ba
                current_count = bc
                no_improve = 0

    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]

    # Final polish: try to improve by testing all single-point moves
    if time.time() < deadline - 2:
        for _ in range(5):  # Multiple polish passes
            improved = False
            for idx in range(N):
                old_pt = coords[idx]
                best_delta = 0
                best_new_pt = None
                for dx in range(-6, 7):
                    for dy in range(-6, 7):
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                            old_matches = 0
                            for t_idx in idx_triplets[idx]:
                                i, j, k = triplets[t_idx]
                                if doubled_area(coords[i], coords[j], coords[k]) == target:
                                    old_matches += 1
                            new_matches = 0
                            for t_idx in idx_triplets[idx]:
                                i, j, k = triplets[t_idx]
                                if i == idx:
                                    p1, p2, p3 = new_pt, coords[j], coords[k]
                                elif j == idx:
                                    p1, p2, p3 = coords[i], new_pt, coords[k]
                                else:
                                    p1, p2, p3 = coords[i], coords[j], new_pt
                                if doubled_area(p1, p2, p3) == target:
                                    new_matches += 1
                            delta = new_matches - old_matches
                            if delta > best_delta:
                                best_delta = delta
                                best_new_pt = new_pt
                if best_new_pt is not None and best_delta > 0:
                    coords[idx] = best_new_pt
                    current_count += best_delta
                    improved = True
                    if current_count > best_overall:
                        best_overall = current_count
                        best_points = list(coords)
            if not improved:
                break

    return best_points
