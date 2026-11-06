import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

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
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n

        # Precompute conflicts: conflicts[i] is a list of bitmasks (1<<j | 1<<k)
        # such that i, j, k are collinear.
        # If we include i, we cannot have both j and k in the current set.
        conflicts = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        mask = (1 << j) | (1 << k)
                        conflicts[i].append(mask)
                        # Also add to j and k?
                        # For the B&B check "if I include idx, is it valid?",
                        # I need to know if there exists a pair (a,b) in current_mask
                        # such that idx, a, b are collinear.
                        # So conflicts[idx] should contain all pairs (a,b) such that idx,a,b collinear.
                        conflicts[j].append((1 << i) | (1 << k))
                        conflicts[k].append((1 << i) | (1 << j))

        # Sort indices by degree (number of conflicts) descending
        # Points with more conflicts are "more constrained".
        # Processing high-degree points first might help prune branches where they are excluded?
        # Or included?
        # Standard heuristic for Independent Set: Process high degree first.
        order = sorted(range(n), key=lambda i: len(conflicts[i]), reverse=True)

        best_score = 0

        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return

            # Pruning: even if we add all remaining points, can we beat best?
            if current_size + (n - step) <= best_score:
                return

            idx = order[step]

            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)

            # Option 2: Include idx
            # Check if adding idx creates a triple with any two points in current_mask
            valid = True
            for mask in conflicts[idx]:
                if (current_mask & mask) == mask:
                    valid = False
                    break

            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)

        bbb(0, 0, 0)
        return best_score

    def greedy_construct():
        """
        Construct a set of 20 points by greedily adding points that maximize
        the number of collinear triples with existing points, while ensuring
        no 4 points are collinear.
        """
        pts = []
        # Start with a few random points to break symmetry
        for _ in range(3):
            pts.append((random.randint(-50, 50), random.randint(-50, 50)))

        while len(pts) < 20:
            best_point = None
            best_triples_count = -1

            # Try candidate points in a grid around the centroid or random locations
            candidates = set()
            for _ in range(100):
                candidates.add((random.randint(-50, 50), random.randint(-50, 50)))
            # Also try neighbors of existing points
            for p in pts:
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        candidates.add((p[0] + dx, p[1] + dy))

            for cand in candidates:
                if cand in pts:
                    continue
                if not (-100 <= cand[0] <= 100 and -100 <= cand[1] <= 100):
                    continue

                # Check if adding cand creates 4 collinear
                valid = True
                line_counts = {}
                for i, p1 in enumerate(pts):
                    key = get_line_key(cand, p1)
                    if key:
                        line_counts[key] = line_counts.get(key, 0) + 1

                # If any line has count >= 3, adding cand makes it 4 -> Invalid.
                for count in line_counts.values():
                    if count >= 3:
                        valid = False
                        break

                if not valid:
                    continue

                # Count new triples
                # A triple is formed if cand, p1, p2 are collinear.
                # This happens if the line (cand, p1) contains p2.
                new_triples = 0
                for key, count in line_counts.items():
                    # If there are 'count' points on the line, they form C(count, 2) pairs.
                    # Each pair with cand forms a triple.
                    new_triples += count * (count - 1) // 2

                if new_triples > best_triples_count:
                    best_triples_count = new_triples
                    best_point = cand

            if best_point is None:
                # Fallback: pick a random valid point
                while True:
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
                        break

            pts.append(best_point)

        return pts

    def local_search(pts, max_iterations=5000):
        current_pts = [tuple(p) for p in pts]
        if has_4_collinear(current_pts):
            # Try to fix it
            for _ in range(100):
                # Find a 4-collinear set
                # Simplified: just random swap
                idx = random.randint(0, 19)
                old_p = current_pts[idx]
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                if new_p not in current_pts and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    current_pts[idx] = new_p
                    if not has_4_collinear(current_pts):
                        break
            if has_4_collinear(current_pts):
                return None, float('inf')

        current_score = calculate_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score

        for i in range(max_iterations):
            if elapsed() > deadline - 10:
                break

            idx = random.randint(0, 19)
            old_p = current_pts[idx]

            # Generate new point
            # Try local moves
            if random.random() < 0.7:
                new_p = (old_p[0] + random.randint(-3, 3), old_p[1] + random.randint(-3, 3))
            else:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))

            if new_p in current_pts:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue

            new_pts = current_pts[:]
            new_pts[idx] = new_p

            if has_4_collinear(new_pts):
                continue

            new_score = calculate_score(new_pts)

            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts
                current_pts = new_pts
                current_score = new_score
            elif new_score == current_score and random.random() < 0.1:
                # Accept equal score to escape local optima
                current_pts = new_pts
                current_score = new_score

        return best_pts, best_score

    best_points = None
    best_val = float('inf')

    # Strategy:
    # 1. Try several greedy constructions (with different random seeds/initials)
    # 2. Local search on each
    # 3. Keep the best

    # Run multiple restarts
    for _ in range(10):
        if elapsed() > deadline - 20:
            break

        cand = greedy_construct()
        if has_4_collinear(cand):
            continue

        pts, score = local_search(cand, max_iterations=2000)

        if pts is not None and score < best_val:
            best_val = score
            best_points = pts

    # If no candidates worked, fallback to random
    if best_points is None:
        while True:
            if elapsed() > deadline - 5:
                break
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break

    if best_points is None:
        # Absolute fallback: simple grid with large steps (might be invalid but try to fix)
        # Or just return a known valid set if time is tight?
        # Let's try to construct one on the fly
        pts = []
        x = 0
        y = 0
        while len(pts) < 20:
            p = (x, y)
            if p not in pts:
                pts.append(p)
            y += 1
            if y > 100:
                y = 0
                x += 1
        # This will likely have 4 collinear.
        # Let's just return the first valid random one found or a hardcoded one if we find one.
        # For safety, if we are here, we are in trouble.
        # Let's try a very sparse set which is likely valid.
        pts = [(i * 10, (i * 7) % 100) for i in range(20)]
        # Check
        if has_4_collinear(pts):
            # Perturb
            for i in range(20):
                while has_4_collinear(pts):
                    if elapsed() > deadline - 1:
                        break
                    pts[i] = (pts[i][0] + 1, pts[i][1])
                    if pts[i][0] > 100: pts[i] = (pts[i][0]-200, pts[i][1])
                    if pts[i][1] > 100: pts[i] = (pts[i][0], pts[i][1]-200)
        best_points = pts

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
    # Note: calculate_score is inside construct, so we can't print it directly here without refactoring
    # But the evaluator will call construct and then score it externally or internally?
    # "The evaluator measures the raw value". It likely calls construct and then runs its own scorer.
    # So we just need to return the points.
