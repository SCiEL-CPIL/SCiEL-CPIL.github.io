import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def grid_configuration():
    points = [[x, y] for x in range(4) for y in range(4)]
    return points, fast_count(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Strategy: Greedy set-packing of 4-point lines
    # 1. Generate candidate 4-point lines from a pool of points
    # 2. Greedily select non-conflicting lines (no 5 collinear, max 1 shared point between selected lines)
    # 3. Collect all points from selected lines
    # 4. If we have exactly 16 points, evaluate. If not, add/remove points to reach 16.
    # 5. Local search refinement on the final point set.

    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = fast_count(best_points)

    # Point pool: use a larger grid to generate diverse candidates
    pool_size = 20
    pool = []
    for x in range(-pool_size, pool_size + 1, 4):
        for y in range(-pool_size, pool_size + 1, 4):
            pool.append([x, y])

    # Generate candidate 4-point lines
    # A candidate line is defined by 4 points that are collinear
    # We'll generate all lines that contain at least 4 points from the pool

    def generate_candidate_lines(pts):
        """Generate all lines that contain exactly 4 points from the given list."""
        n = len(pts)
        lines = {}  # line_key -> set of point indices
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)

        candidates = []
        for key, indices in lines.items():
            if len(indices) == 4:
                candidates.append((key, indices))
            elif len(indices) > 4:
                # For lines with >4 points, we can't use them directly (would violate no-5 constraint)
                # But we can pick subsets of 4
                # For simplicity, skip for now; the greedy will handle it
                pass

        return candidates

    def greedy_line_packing(pool_pts, max_lines=15):
        """
        Greedily select 4-point lines from the pool such that:
        - No two selected lines share more than 1 point
        - No line has 5+ points in the final configuration
        - Maximize number of selected lines
        Returns: list of selected lines (each is a set of 4 point indices), and the set of all points used
        """
        candidates = generate_candidate_lines(pool_pts)

        # Sort candidates: prefer lines that are "central" or have high potential
        # For now, just use random order with some heuristic
        random.shuffle(candidates)

        selected_lines = []
        point_usage = {}  # point_index -> count of selected lines using it
        all_points_used = set()

        for key, indices in candidates:
            # Check if this line can be added
            # Constraint 1: No point used in more than 3 selected lines (to avoid 5 collinear)
            # Actually, the constraint is: no 5 points collinear in the FINAL set.
            # If we select lines, the final set is the union of all points in selected lines.
            # A line in the final set has 5+ points if 5 of the final points are collinear.
            # This is hard to check incrementally. Let's use a simpler constraint:
            # Each point can be in at most 3 selected lines (heuristic)

            valid = True
            for idx in indices:
                if point_usage.get(idx, 0) >= 3:
                    valid = False
                    break
            if not valid:
                continue

            # Check conflict with already selected lines: share at most 1 point
            for sel_indices in selected_lines:
                overlap = len(indices & sel_indices)
                if overlap > 1:
                    valid = False
                    break
            if not valid:
                continue

            # Add this line
            selected_lines.append(indices)
            for idx in indices:
                point_usage[idx] = point_usage.get(idx, 0) + 1
                all_points_used.add(idx)

            if len(selected_lines) >= max_lines:
                break

        return selected_lines, all_points_used

    # Try multiple random subsets of the pool to find good configurations
    for trial in range(200):
        if time.time() > deadline - 10:
            break

        # Randomly sample 24-32 points from the pool
        sample_size = random.randint(24, 32)
        if len(pool) < sample_size:
            sample = pool[:]
        else:
            sample = random.sample(pool, sample_size)

        selected_lines, all_points_used = greedy_line_packing(sample)

        if len(all_points_used) < 4:
            continue

        # We need exactly 16 points. If we have more, randomly drop some.
        # If we have fewer, add random points from the pool.
        current_points = [sample[i] for i in sorted(all_points_used)]

        if len(current_points) > 16:
            # Randomly select 16
            current_points = random.sample(current_points, 16)
        elif len(current_points) < 16:
            # Add random points from pool not already in current_points
            used_set = set(map(tuple, current_points))
            for p in pool:
                if len(current_points) >= 16:
                    break
                if tuple(p) not in used_set:
                    current_points.append(p)
                    used_set.add(tuple(p))

        if len(current_points) != 16:
            continue

        # Check for duplicates
        if len(set(map(tuple, current_points))) != 16:
            continue

        # Check bounds
        if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in current_points):
            continue

        score = fast_count(current_points)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in current_points]

        # Local search refinement
        if score > 0 and time.time() < deadline - 20:
            # Simple local search: try moving each point slightly
            pts = [p[:] for p in current_points]
            cur_score = score
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                for i in range(16):
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if dx == 0 and dy == 0:
                                continue
                            new_x = pts[i][0] + dx
                            new_y = pts[i][1] + dy
                            if abs(new_x) > 1000 or abs(new_y) > 1000:
                                continue
                            # Check collision
                            collision = False
                            for j in range(16):
                                if j != i and pts[j][0] == new_x and pts[j][1] == new_y:
                                    collision = True
                                    break
                            if collision:
                                continue

                            old_pt = pts[i][:]
                            pts[i] = [new_x, new_y]
                            new_score = fast_count(pts)
                            if new_score > cur_score:
                                cur_score = new_score
                                improved = True
                                if cur_score > best_score:
                                    best_score = cur_score
                                    best_points = [p[:] for p in pts]
                            else:
                                pts[i] = old_pt

    # Final local search on the best configuration found
    if best_score > 0 and time.time() < deadline - 10:
        pts = [p[:] for p in best_points]
        cur_score = best_score
        # More thorough local search
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(16):
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = pts[i][0] + dx
                        new_y = pts[i][1] + dy
                        if abs(new_x) > 1000 or abs(new_y) > 1000:
                            continue
                        collision = False
                        for j in range(16):
                            if j != i and pts[j][0] == new_x and pts[j][1] == new_y:
                                collision = True
                                break
                        if collision:
                            continue

                        old_pt = pts[i][:]
                        pts[i] = [new_x, new_y]
                        new_score = fast_count(pts)
                        if new_score > cur_score:
                            cur_score = new_score
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_points = [p[:] for p in pts]
                        else:
                            pts[i] = old_pt
            if not improved:
                break

    final_score = fast_count(best_points)
    if final_score < 0:
        grid_pts, _ = grid_configuration()
        return grid_pts

    return best_points
