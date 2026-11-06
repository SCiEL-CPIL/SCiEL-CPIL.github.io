import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        for s in line_map.values():
            if len(s) >= 4:
                return True
        return False

    def count_triples(pts):
        """Count total collinear triples in the set."""
        n = len(pts)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        total = 0
        for s in line_map.values():
            k = len(s)
            if k >= 3:
                total += k * (k - 1) * (k - 2) // 6
        return total

    def exact_max_no3(pts):
        """Exact branch-and-bound for largest subset with no 3 collinear."""
        n = len(pts)
        if n < 3:
            return n

        # Build conflict masks: for each point i, list of (j,k) pairs collinear with i
        conflicts = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        conflicts[i].append((1 << j) | (1 << k))
                        conflicts[j].append((1 << i) | (1 << k))
                        conflicts[k].append((1 << i) | (1 << j))

        # Order by descending conflict count for better pruning
        order = sorted(range(n), key=lambda i: len(conflicts[i]), reverse=True)
        order_inv = [0] * n
        for idx, orig in enumerate(order):
            order_inv[orig] = idx

        best = [0]

        def bb(step, mask, size):
            if size > best[0]:
                best[0] = size
            if step == n:
                return
            if size + (n - step) <= best[0]:
                return
            if elapsed() > deadline - 3:
                return

            idx = order[step]

            # Exclude
            bb(step + 1, mask, size)

            # Include if valid
            valid = True
            for cmask in conflicts[idx]:
                if (mask & cmask) == cmask:
                    valid = False
                    break
            if valid:
                bb(step + 1, mask | (1 << idx), size + 1)

        bb(0, 0, 0)
        return best[0]

    def fast_score_estimate(pts):
        """
        Fast upper bound on max no-3-collinear subset.
        If we have T triples, by pigeonhole, any subset of size m with no 3 collinear
        must have at most C(m,3) - T triples removed... not quite.
        
        Better: use the fact that each triple "blocks" at least 1 point from being in a no-3 set.
        Actually, a simple bound: if there are T triples, the max no-3 subset is at most n - T/3 roughly? No.
        
        Let's use: the max independent set in the 3-uniform hypergraph of triples.
        A simple upper bound: n - floor(T / 3) is not valid.
        
        Actually, let's just use a greedy lower bound and a simple upper bound.
        For the search, we'll use exact when feasible.
        
        Upper bound: n (trivial)
        Better: if we partition points into lines with >= 3, each such line of size k contributes at most k-2 to a no-3 set.
        """
        n = len(pts)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        
        # Find lines with >= 3 points
        large_lines = [s for s in line_map.values() if len(s) >= 3]
        
        if not large_lines:
            return n  # No triples, all points can be in a no-3 set
        
        # Upper bound: for each line with k points, at most k-2 can be in the no-3 set.
        # But points may be on multiple lines. Use a simple bound:
        # The max no-3 set is at most n - sum over lines of (k - 2) / (number of lines containing that point)
        # This is getting complex. Let's just return n as a loose upper bound and rely on exact search.
        return n

    def greedy_construct():
        """
        Construct 20 points maximizing collinear triples while keeping no 4 collinear.
        Strategy: place points to form many 3-point lines.
        """
        pts = []
        used_lines = set()  # lines that already have 3 points (can't add more)

        # Start with a structured pattern: use a few "anchor" lines
        # Place points at intersections of predefined lines to create many triples
        
        # Strategy: create a grid of potential triple-lines
        # Use lines with small integer coefficients for easy intersection
        
        # Generate candidate lines (A, B, C) with small coefficients
        candidate_lines = []
        for A in range(-3, 4):
            for B in range(-3, 4):
                if A == 0 and B == 0:
                    continue
                for C in range(-20, 21):
                    # Check if this line passes through integer points in range
                    candidate_lines.append((A, B, C))
        
        # For each candidate line, find integer points on it in [-100, 100]^2
        # Then greedily pick lines that share points to create dense triple networks
        
        # Simpler approach: random with triple-maximizing selection
        random.seed()
        
        # Phase 1: Place first 3 points
        for _ in range(3):
            p = (random.randint(-50, 50), random.randint(-50, 50))
            while p in pts:
                p = (random.randint(-50, 50), random.randint(-50, 50))
            pts.append(p)

        while len(pts) < 20:
            best_point = None
            best_triples = -1
            best_line_count = -1

            # Generate candidates: intersections of lines through existing point pairs
            candidates = set()
            
            # Lines through pairs of existing points
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    key = get_line_key(pts[i], pts[j])
                    if key is None:
                        continue
                    A, B, C = key
                    # Find integer points on this line in range
                    # Ax + By + C = 0
                    if A != 0:
                        # x = (-By - C) / A
                        for y in range(-100, 101):
                            num = -B * y - C
                            if num % A == 0:
                                x = num // A
                                if -100 <= x <= 100:
                                    candidates.add((x, y))
                    elif B != 0:
                        # y = (-Ax - C) / B
                        for x in range(-100, 101):
                            num = -A * x - C
                            if num % B == 0:
                                y = num // B
                                if -100 <= y <= 100:
                                    candidates.add((x, y))
            
            # Also add some random candidates
            for _ in range(50):
                candidates.add((random.randint(-100, 100), random.randint(-100, 100)))
            
            # Also try points near existing points
            for p in pts:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        candidates.add((p[0] + dx, p[1] + dy))

            for cand in candidates:
                if cand in pts:
                    continue
                if not (-100 <= cand[0] <= 100 and -100 <= cand[1] <= 100):
                    continue

                # Check no 4 collinear
                valid = True
                line_counts = {}
                for p1 in pts:
                    key = get_line_key(cand, p1)
                    if key:
                        line_counts[key] = line_counts.get(key, 0) + 1
                for count in line_counts.values():
                    if count >= 3:
                        valid = False
                        break
                if not valid:
                    continue

                # Count new triples created
                new_triples = 0
                for key, count in line_counts.items():
                    if count >= 2:
                        new_triples += count * (count - 1) // 2

                # Also count how many existing lines this point lies on (with >= 2 existing points)
                line_count = sum(1 for c in line_counts.values() if c >= 2)

                if new_triples > best_triples or (new_triples == best_triples and line_count > best_line_count):
                    best_triples = new_triples
                    best_line_count = line_count
                    best_point = cand

            if best_point is None:
                # Fallback: random valid point
                found = False
                for _ in range(1000):
                    cand = (random.randint(-100, 100), random.randint(-100, 100))
                    if cand in pts:
                        continue
                    valid = True
                    line_counts = {}
                    for p1 in pts:
                        key = get_line_key(cand, p1)
                        if key:
                            line_counts[key] = line_counts.get(key, 0) + 1
                    for count in line_counts.values():
                        if count >= 3:
                            valid = False
                            break
                    if valid:
                        best_point = cand
                        found = True
                        break
                if not found:
                    # Very unlikely, but handle it
                    best_point = (random.randint(-100, 100), random.randint(-100, 100))

            pts.append(best_point)

        return pts

    def local_search(pts, max_iters):
        current_pts = [tuple(p) for p in pts]
        if has_4_collinear(current_pts):
            return None, float('inf')

        current_triples = count_triples(current_pts)
        best_pts = current_pts[:]
        best_triples = current_triples
        best_score = None  # Lazy compute

        for it in range(max_iters):
            if elapsed() > deadline - 15:
                break

            idx = random.randint(0, 19)
            old_p = current_pts[idx]

            # Generate candidate replacements
            new_p = None
            if random.random() < 0.6:
                # Local move
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
            else:
                # Try to place on a line with 2 existing points to create a triple
                # Pick two random existing points and try to find a third on their line
                i, j = random.sample(range(20), 2)
                key = get_line_key(current_pts[i], current_pts[j])
                if key:
                    A, B, C = key
                    # Find integer points on this line
                    if A != 0:
                        for y in range(-100, 101):
                            num = -B * y - C
                            if num % A == 0:
                                x = num // A
                                if -100 <= x <= 100 and (x, y) not in current_pts:
                                    new_p = (x, y)
                                    break
                    elif B != 0:
                        for x in range(-100, 101):
                            num = -A * x - C
                            if num % B == 0:
                                y = num // B
                                if -100 <= y <= 100 and (x, y) not in current_pts:
                                    new_p = (x, y)
                                    break
                if new_p is None:
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))

            if new_p is None or new_p in current_pts:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue

            new_pts = current_pts[:]
            new_pts[idx] = new_p

            if has_4_collinear(new_pts):
                continue

            new_triples = count_triples(new_pts)

            # Accept if more triples (better for minimizing max no-3 subset)
            if new_triples > current_triples:
                current_pts = new_pts
                current_triples = new_triples
                if new_triples > best_triples:
                    best_triples = new_triples
                    best_pts = new_pts
            elif new_triples == current_triples and random.random() < 0.15:
                current_pts = new_pts

        # Compute exact score for best candidate
        if elapsed() < deadline - 10:
            best_score = exact_max_no3(best_pts)
        else:
            best_score = 20  # Fallback

        return best_pts, best_score

    best_points = None
    best_score_val = float('inf')

    # Multiple restarts with greedy construction + local search
    num_restarts = 8
    for r in range(num_restarts):
        if elapsed() > deadline - 30:
            break

        cand = greedy_construct()
        if has_4_collinear(cand):
            continue

        # Local search with more iterations for early restarts
        iters = max(500, int((deadline - 30 - elapsed()) * 30))
        pts, score = local_search(cand, iters)

        if pts is not None and score < best_score_val:
            best_score_val = score
            best_points = pts

    # Final polish: try to improve the best found
    if best_points is not None and elapsed() < deadline - 20:
        # Do a few more targeted local search steps
        for _ in range(3):
            if elapsed() > deadline - 20:
                break
            iters = int((deadline - 20 - elapsed()) * 20)
            if iters < 100:
                break
            pts, score = local_search(best_points, iters)
            if pts is not None and score < best_score_val:
                best_score_val = score
                best_points = pts

    if best_points is None:
        # Fallback: random valid set
        while elapsed() < deadline - 5:
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break

    if best_points is None:
        # Absolute fallback
        best_points = [(i * 9, (i * 13) % 100) for i in range(20)]
        while has_4_collinear(best_points) and elapsed() < deadline - 1:
            i = random.randint(0, 19)
            best_points[i] = (best_points[i][0] + random.randint(-2, 2), 
                             best_points[i][1] + random.randint(-2, 2))
            if best_points[i][0] > 100:
                best_points[i] = (best_points[i][0] - 200, best_points[i][1])
            if best_points[i][0] < -100:
                best_points[i] = (best_points[i][0] + 200, best_points[i][1])
            if best_points[i][1] > 100:
                best_points[i] = (best_points[i][0], best_points[i][1] - 200)
            if best_points[i][1] < -100:
                best_points[i] = (best_points[i][0], best_points[i][1] + 200)

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
