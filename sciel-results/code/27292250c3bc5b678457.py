import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def elapsed():
        return time.time() - start_time

    def line_key(p1, p2):
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
            A, B, C = -A, -B, -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                c = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        c += 1
                        if c >= 4:
                            return True
        return False

    def build_triple_conflicts(pts):
        """Build conflict structure: for each point, which pairs of other points are collinear with it."""
        n = len(pts)
        # triple_conflict[i] = set of bitmasks (1<<j)|(1<<k) where j,k > i and i,j,k collinear
        # Actually, let's build: for each pair (i,j), list of k such that i,j,k collinear
        pair_to_thirds = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        if (i, j) not in pair_to_thirds:
                            pair_to_thirds[(i, j)] = []
                        pair_to_thirds[(i, j)].append(k)
        return pair_to_thirds

    def count_perfect_triples(pts):
        """Count number of lines with exactly 3 points."""
        n = len(pts)
        lines = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        count = 0
        for key, pts_on_line in lines.items():
            if len(pts_on_line) == 3:
                count += 1
        return count

    def fast_greedy_independent(pts):
        """Fast greedy estimate of max subset with no 3 collinear."""
        n = len(pts)
        # For each point, count how many other points share a line with it (that would form triples)
        # Build conflict: point i conflicts with pair (j,k) if i,j,k collinear
        point_conflict_count = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        point_conflict_count[i] += 1
                        point_conflict_count[j] += 1
                        point_conflict_count[k] += 1

        # Greedy: pick points with fewest conflicts first
        order = sorted(range(n), key=lambda i: point_conflict_count[i])
        selected = set()
        for idx in order:
            # Check if adding idx creates any triple
            valid = True
            for j in selected:
                for k in selected:
                    if j == k:
                        continue
                    x1, y1 = pts[idx]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.add(idx)
        return len(selected)

    def exact_max_no3(pts):
        """Exact B&B for largest subset with no 3 collinear."""
        n = len(pts)
        if n < 3:
            return n

        # Precompute collinearity
        collinear = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        collinear[i][j] = collinear[j][i] = True
                        collinear[i][k] = collinear[k][i] = True
                        collinear[j][k] = collinear[k][j] = True

        # Conflict: can't have all three of i,j,k if any pair is collinear... 
        # Actually: i,j,k form a bad triple if all three are on the same line
        # Let's use: bad_triples[i] = set of (j,k) pairs that with i form a collinear triple
        bad_with = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not collinear[i][j]:
                    continue
                for k in range(j + 1, n):
                    if collinear[i][k] and collinear[j][k]:
                        bad_with[i].append((j, k))
                        bad_with[j].append((i, k))
                        bad_with[k].append((i, j))

        # Order by degree
        order = sorted(range(n), key=lambda i: len(bad_with[i]), reverse=True)
        order_idx = {v: i for i, v in enumerate(order)}

        best = [0]

        def greedy_ub(step, mask):
            count = 0
            for s in range(step, n):
                idx = order[s]
                if mask & (1 << idx):
                    continue
                # Check if idx can be added
                valid = True
                for (j, k) in bad_with[idx]:
                    if (mask & (1 << j)) and (mask & (1 << k)):
                        valid = False
                        break
                if valid:
                    count += 1
                    mask |= (1 << idx)
            return count

        def bb(step, mask, size):
            if elapsed() > DEADLINE - 5:
                return
            if size > best[0]:
                best[0] = size
            if step == n:
                return
            if size + (n - step) <= best[0]:
                return
            ub = greedy_ub(step, mask)
            if size + ub <= best[0]:
                return
            idx = order[step]
            # Try including idx
            valid = True
            for (j, k) in bad_with[idx]:
                if (mask & (1 << j)) and (mask & (1 << k)):
                    valid = False
                    break
            if valid:
                bb(step + 1, mask | (1 << idx), size + 1)
                if elapsed() > DEADLINE - 5:
                    return
            # Try excluding idx
            bb(step + 1, mask, size)

        # Initial lower bound from greedy
        best[0] = fast_greedy_independent(pts)
        bb(0, 0, 0)
        return best[0]

    def generate_structured_candidates():
        """Generate candidates from a structured grid to ensure good properties."""
        candidates = set()
        # Use a grid with spacing to avoid accidental 4-collinear
        for x in range(-90, 91, 5):
            for y in range(-90, 91, 5):
                candidates.add((x, y))
        # Add some random points
        for _ in range(500):
            candidates.add((random.randint(-100, 100), random.randint(-100, 100)))
        return list(candidates)

    def greedy_construct(seed):
        """Greedy construction maximizing triple density while avoiding 4-collinear."""
        rng = random.Random(seed)
        all_cands = generate_structured_candidates()
        pts = []

        # Start with a few random points
        for _ in range(3):
            p = rng.choice(all_cands)
            if p not in pts:
                pts.append(p)

        while len(pts) < 20:
            best_cand = None
            best_score = -1
            best_triples = -1

            # Sample a subset of candidates for efficiency
            sample_size = min(300, len(all_cands))
            candidates = rng.sample(all_cands, sample_size)

            for cand in candidates:
                if cand in pts:
                    continue
                if not (-100 <= cand[0] <= 100 and -100 <= cand[1] <= 100):
                    continue

                # Check no 4 collinear
                trial = pts + [cand]
                if has_4_collinear(trial):
                    continue

                # Count new triples formed
                new_triples = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        x3, y3 = cand
                        if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                            new_triples += 1

                # Estimate score (lower is better)
                est = fast_greedy_independent(trial)

                if est < best_score or (est == best_score and new_triples > best_triples):
                    best_score = est
                    best_triples = new_triples
                    best_cand = cand

            if best_cand is None:
                # Fallback: find any valid point
                for _ in range(200):
                    cand = (rng.randint(-100, 100), rng.randint(-100, 100))
                    if cand in pts:
                        continue
                    trial = pts + [cand]
                    if not has_4_collinear(trial):
                        best_cand = cand
                        break
                if best_cand is None:
                    break

            pts.append(best_cand)

        return pts

    def simulated_annealing(pts, seed, max_iters=5000):
        """Simulated annealing to minimize the score."""
        rng = random.Random(seed)
        current = [tuple(p) for p in pts]
        current_score = exact_max_no3(current)
        best = current[:]
        best_score = current_score
        T = 2.0
        cooling = 0.9995

        for it in range(max_iters):
            if elapsed() > DEADLINE - 10:
                break

            # Generate a move
            move_type = rng.random()
            if move_type < 0.6:
                # Single point move
                idx = rng.randint(0, 19)
                old = current[idx]
                if rng.random() < 0.5:
                    new = (old[0] + rng.randint(-6, 6), old[1] + rng.randint(-6, 6))
                else:
                    new = (rng.randint(-100, 100), rng.randint(-100, 100))
                if new in current or not (-100 <= new[0] <= 100 and -100 <= new[1] <= 100):
                    continue
                trial = current[:]
                trial[idx] = new
            elif move_type < 0.85:
                # Two-point swap
                i1, i2 = rng.sample(range(20), 2)
                p1, p2 = current[i1], current[i2]
                new1 = (p1[0] + rng.randint(-4, 4), p1[1] + rng.randint(-4, 4))
                new2 = (p2[0] + rng.randint(-4, 4), p2[1] + rng.randint(-4, 4))
                if new1 in current or new2 in current or new1 == new2:
                    continue
                if not (-100 <= new1[0] <= 100 and -100 <= new1[1] <= 100):
                    continue
                if not (-100 <= new2[0] <= 100 and -100 <= new2[1] <= 100):
                    continue
                trial = current[:]
                trial[i1] = new1
                trial[i2] = new2
            else:
                # Random restart of 3 points
                idxs = rng.sample(range(20), 3)
                trial = current[:]
                for idx in idxs:
                    trial[idx] = (rng.randint(-100, 100), rng.randint(-100, 100))
                if len(set(trial)) != 20:
                    continue

            if has_4_collinear(trial):
                continue

            trial_score = exact_max_no3(trial)
            delta = trial_score - current_score

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current = trial
                current_score = trial_score
                if current_score < best_score:
                    best_score = current_score
                    best = current[:]

            T *= cooling

        return best, best_score

    best_points = None
    best_val = float('inf')

    # Multiple restarts
    for attempt in range(25):
        if elapsed() > DEADLINE - 30:
            break

        cand = greedy_construct(seed=100 + attempt)
        if len(cand) < 20 or has_4_collinear(cand):
            continue

        # Simulated annealing
        pts, score = simulated_annealing(cand, seed=500 + attempt, max_iters=3000)

        if pts is not None and score < best_val:
            best_val = score
            best_points = pts

        # Additional polish if time permits
        if elapsed() < DEADLINE - 40 and best_points is not None:
            pts2, score2 = simulated_annealing(best_points, seed=900 + attempt, max_iters=4000)
            if pts2 is not None and score2 < best_val:
                best_val = score2
                best_points = pts2

    # Fallback
    if best_points is None:
        pts = []
        while len(pts) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p in pts:
                continue
            trial = pts + [p]
            if not has_4_collinear(trial):
                pts.append(p)
        best_points = pts
        best_val = exact_max_no3(best_points)

    return [[x, y] for x, y in best_points]


if __name__ == "__main__":
    pts = construct()
    print(pts)
