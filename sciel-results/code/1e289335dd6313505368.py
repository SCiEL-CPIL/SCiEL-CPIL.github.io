import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit

    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def full_score(points):
        """Count lines with exactly 4 points and lines with 5+ points."""
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_counts[key] += 1
        num_4 = 0
        num_5_plus = 0
        for count in line_counts.values():
            if count == 6:
                num_4 += 1
            elif count >= 10:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate_score(n4, n5):
        return n4 - 10000 * n5

    def initial_grid():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        return pts

    def random_config():
        """Generate a random configuration with no 5 collinear."""
        pts = []
        used = set()
        for _ in range(16):
            for _attempt in range(100):
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in used:
                    # Check no 5 collinear with existing points
                    if not causes_5_collinear(pts, (x, y)):
                        pts.append([x, y])
                        used.add((x, y))
                        break
            else:
                # Fallback: just add a unique point
                while (x, y) in used:
                    x = random.randint(-50, 50)
                    y = random.randint(-50, 50)
                pts.append([x, y])
                used.add((x, y))
        return pts

    def causes_5_collinear(pts, new_pt):
        """Check if adding new_pt to pts would create 5 collinear."""
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                for k in range(j + 1, len(pts)):
                    # Check if new_pt is collinear with pts[i], pts[j], pts[k]
                    # and also collinear with one more existing point
                    key1 = canonical_line(pts[i], pts[j])
                    key2 = canonical_line(pts[i], pts[k])
                    if key1 == key2:
                        # Check if new_pt is on this line
                        a, b, c = key1
                        if a * new_pt[0] + b * new_pt[1] == c:
                            # Count how many existing points are on this line
                            count = 0
                        for p in pts:
                            if a * p[0] + b * p[1] == c:
                                count += 1
                            if count >= 2:
                                return True
        return False

    # Simulated Annealing with incremental evaluation
    def sa_search(initial_pts, max_iters, t_start, t_end):
        """Simulated annealing with move and swap operations."""
        points = [p[:] for p in initial_pts]
        n4, n5 = full_score(points)
        current_score = evaluate_score(n4, n5)
        best_score = current_score
        best_points = [p[:] for p in points]

        # Build line count structure for incremental updates
        # We'll use a dictionary: line_key -> set of point indices
        def build_line_structure(pts):
            line_to_pts = defaultdict(set)
            n = len(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    key = canonical_line(pts[i], pts[j])
                    line_to_pts[key].add(i)
                    line_to_pts[key].add(j)
            return line_to_pts

        line_to_pts = build_line_structure(points)

        def get_line_count_info(line_to_pts):
            """Return (num_4, num_5plus) from line structure."""
            num_4 = 0
            num_5 = 0
            for pts_set in line_to_pts.values():
                cnt = len(pts_set)
                if cnt == 4:
                    num_4 += 1
                elif cnt >= 5:
                    num_5 += 1
            return num_4, num_5

        def remove_point_from_lines(pts, idx, line_to_pts):
            """Remove point idx from all lines it's on. Returns list of (line_key, old_size)."""
            removed = []
            for key in list(line_to_pts.keys()):
                if idx in line_to_pts[key]:
                    old_size = len(line_to_pts[key])
                    line_to_pts[key].discard(idx)
                    if len(line_to_pts[key]) <= 1:
                        del line_to_pts[key]
                    else:
                        removed.append((key, old_size))
            return removed

        def add_point_to_lines(pts, idx, line_to_pts):
            """Add point idx to all lines formed with other points."""
            n = len(pts)
            for j in range(n):
                if j == idx:
                    continue
                key = canonical_line(pts[idx], pts[j])
                if key not in line_to_pts:
                    line_to_pts[key] = set()
                line_to_pts[key].add(idx)
                line_to_pts[key].add(j)

        def try_move(idx, new_pos):
            """Try moving point idx to new_pos. Returns (accepted, new_n4, new_n5)."""
            if abs(new_pos[0]) > 1000 or abs(new_pos[1]) > 1000:
                return False, n4, n5
            # Check distinctness
            for k in range(16):
                if k != idx and points[k][0] == new_pos[0] and points[k][1] == new_pos[1]:
                    return False, n4, n5

            old_pos = points[idx][:]
            # Remove point from lines
            remove_point_from_lines(points, idx, line_to_pts)
            # Set new position
            points[idx] = list(new_pos)
            # Add point to lines
            add_point_to_lines(points, idx, line_to_pts)

            new_n4, new_n5 = get_line_count_info(line_to_pts)
            new_score = evaluate_score(new_n4, new_n5)
            old_score = evaluate_score(n4, n5)
            delta = new_score - old_score

            if delta > 0:
                return True, new_n4, new_n5
            else:
                # Decide to accept based on temperature
                if t > 0.01:
                    import math
                    prob = math.exp(delta / t)
                    if random.random() < prob:
                        return True, new_n4, new_n5
                # Reject: revert
                points[idx] = old_pos
                remove_point_from_lines(points, idx, line_to_pts)
                add_point_to_lines(points, idx, line_to_pts)
                return False, n4, n5

        def try_swap(idx1, idx2):
            """Try swapping positions of points idx1 and idx2."""
            p1 = points[idx1][:]
            p2 = points[idx2][:]
            # Check if swap creates duplicate
            # After swap, idx1 gets p2's pos, idx2 gets p1's pos
            # Need to check no other point has these positions
            for k in range(16):
                if k != idx1 and k != idx2:
                    if points[k][0] == p2[0] and points[k][1] == p2[1]:
                        return False, n4, n5
                    if points[k][0] == p1[0] and points[k][1] == p1[1]:
                        return False, n4, n5

            # Remove both points
            remove_point_from_lines(points, idx1, line_to_pts)
            remove_point_from_lines(points, idx2, line_to_pts)
            # Swap
            points[idx1] = list(p2)
            points[idx2] = list(p1)
            # Add both points
            add_point_to_lines(points, idx1, line_to_pts)
            add_point_to_lines(points, idx2, line_to_pts)

            new_n4, new_n5 = get_line_count_info(line_to_pts)
            new_score = evaluate_score(new_n4, new_n5)
            old_score = evaluate_score(n4, n5)
            delta = new_score - old_score

            if delta > 0:
                return True, new_n4, new_n5
            else:
                if t > 0.01:
                    import math
                    prob = math.exp(delta / t)
                    if random.random() < prob:
                        return True, new_n4, new_n5
                # Reject: revert
                points[idx1] = list(p1)
                points[idx2] = list(p2)
                remove_point_from_lines(points, idx1, line_to_pts)
                remove_point_from_lines(points, idx2, line_to_pts)
                add_point_to_lines(points, idx1, line_to_pts)
                add_point_to_lines(points, idx2, line_to_pts)
                return False, n4, n5

        iters = 0
        while iters < max_iters and time.time() < deadline:
            t = t_start * (t_end / t_start) ** (iters / max_iters)
            iters += 1

            # Choose move type: 70% single move, 30% swap
            if random.random() < 0.7:
                idx = random.randint(0, 15)
                ox, oy = points[idx]
                # Random move: small step with some probability of larger step
                if random.random() < 0.8:
                    step = random.randint(1, 3)
                else:
                    step = random.randint(4, 15)
                nx = ox + random.randint(-step, step)
                ny = oy + random.randint(-step, step)
                accepted, n4, n5 = try_move(idx, (nx, ny))
            else:
                idx1 = random.randint(0, 15)
                idx2 = random.randint(0, 15)
                if idx1 == idx2:
                    continue
                accepted, n4, n5 = try_swap(idx1, idx2)

            if accepted:
                cur_sc = evaluate_score(n4, n5)
                if cur_sc > best_score:
                    best_score = cur_sc
                    best_points = [p[:] for p in points]

        return best_points, best_score

    # Start with the 4x4 grid
    init_pts = initial_grid()
    n4_init, n5_init = full_score(init_pts)
    best_score = evaluate_score(n4_init, n5_init)
    best_points = init_pts

    # Run SA from the grid
    max_iters = 500000
    t_start = 10.0
    t_end = 0.001
    bp, bs = sa_search(init_pts, max_iters, t_start, t_end)
    if bs > best_score:
        best_score = bs
        best_points = bp

    # If time remains, try SA from random configs
    while time.time() < deadline - 5.0:
        rand_pts = random_config()
        rn4, rn5 = full_score(rand_pts)
        rsc = evaluate_score(rn4, rn5)
        if rsc >= best_score - 5:  # Only continue if promising
            iters = min(100000, max(10000, int((deadline - time.time()) * 2000)))
            t_s = 5.0
            t_e = 0.001
            bp, bs = sa_search(rand_pts, iters, t_s, t_e)
            if bs > best_score:
                best_score = bs
                best_points = bp

    # Final validation
    fn4, fn5 = full_score(best_points)
    if fn5 > 0:
        # Fallback to grid if somehow invalid
        best_points = initial_grid()

    return best_points

# Test briefly
if __name__ == "__main__":
    import time
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")
    print(pts)
    print(len(pts))
    # Verify no 5 collinear
    from collections import defaultdict
    from math import gcd
    def _cl(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0: return (0,0,0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)
    
    line_counts = defaultdict(int)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            line_counts[_cl(pts[i], pts[j])] += 1
            
    max_collinear = 0
    for k, v in line_counts.items():
        if v == 10: max_collinear = max(max_collinear, 5)
        elif v >= 15: max_collinear = max(max_collinear, 6)
        elif v == 6: max_collinear = max(max_collinear, 4)
        elif v == 3: max_collinear = max(max_collinear, 3)
        elif v == 1: max_collinear = max(max_collinear, 2)
        
    print("Max collinear:", max_collinear)
    count_4 = 0
    for k, v in line_counts.items():
        if v == 6:
            count_4 += 1
    print("4-lines:", count_4)
