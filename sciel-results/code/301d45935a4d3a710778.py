import time
import random
import math
from collections import defaultdict
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def _line_key(p1, p2):
        """Compute canonical line key from two points."""
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            return ('V', x1)
        if y1 == y2:
            return ('H', y1)
        m = Fraction(y2 - y1, x2 - x1)
        b = Fraction(y1 * x2 - x1 * y2, x2 - x1)
        return ('S', m, b)

    def _evaluate(points):
        """
        Returns (score, is_valid).
        score: number of lines with exactly 4 points
        is_valid: False if any line has >= 5 points or points not distinct
        """
        n = len(points)
        if len(set(tuple(p) for p in points)) != n:
            return 0, False

        line_pts = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(points[i], points[j])
                line_pts[key].add(i)
                line_pts[key].add(j)

        score = 0
        for key, pts in line_pts.items():
            c = len(pts)
            if c >= 5:
                return 0, False
            if c == 4:
                score += 1
        return score, True

    def _incremental_eval(points, changed_idx):
        """
        Fast evaluation: only recompute lines involving changed point.
        Returns (score, is_valid) or (None, None) if invalid.
        """
        n = len(points)
        # Check distinctness quickly
        seen = set()
        for i, p in enumerate(points):
            tp = tuple(p)
            if tp in seen:
                return 0, False
            seen.add(tp)

        # Build line map for all pairs except those involving changed_idx
        line_pts = defaultdict(set)
        ci = changed_idx
        for i in range(n):
            for j in range(i + 1, n):
                if i == ci or j == ci:
                    continue
                key = _line_key(points[i], points[j])
                line_pts[key].add(i)
                line_pts[key].add(j)

        # Add lines involving changed point
        for j in range(n):
            if j == ci:
                continue
            key = _line_key(points[ci], points[j])
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(ci)
            line_pts[key].add(j)

        score = 0
        for key, pts in line_pts.items():
            c = len(pts)
            if c >= 5:
                return 0, False
            if c == 4:
                score += 1
        return score, True

    def _initial_4x4():
        pts = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                pts.append([x, y])
        return pts

    def _initial_perturbed():
        """4x4 grid with some random perturbations."""
        pts = _initial_4x4()
        for _ in range(random.randint(2, 5)):
            idx = random.randint(0, 15)
            pts[idx] = [random.randint(-4, 4), random.randint(-4, 4)]
        # Ensure distinct
        seen = set()
        for i, p in enumerate(pts):
            tp = tuple(p)
            if tp in seen:
                pts[i] = [random.randint(-6, 6), random.randint(-6, 6)]
            seen.add(tuple(pts[i]))
        return pts

    def _initial_affine_like():
        """
        Try a configuration inspired by affine plane AG(2,4) mapped to integers.
        AG(2,4) has 16 points and 20 lines of 4 points each.
        We map the 4x4 grid of GF(4)^2 to integer coordinates.
        """
        # GF(4) = {0, 1, a, a^2} where a^2 = a + 1
        # Represent elements as 0,1,2,3
        # We'll use a 4x4 grid but with non-standard coordinates
        # to break some 4-point lines while creating others
        pts = []
        # Standard 4x4 grid
        for i in range(4):
            for j in range(4):
                pts.append([i - 1.5, j - 1.5])
        # Convert to integers by scaling
        pts = [[int(round(x * 2)), int(round(y * 2))] for x, y in pts]
        # This gives a 4x4 grid from -3 to 3 step 2
        # Let's perturb some points
        random.shuffle(pts)
        for k in range(random.randint(1, 3)):
            idx = random.randint(0, 15)
            pts[idx] = [pts[idx][0] + random.randint(-1, 1),
                       pts[idx][1] + random.randint(-1, 1)]
        # Ensure within bounds and distinct
        seen = set()
        for i, p in enumerate(pts):
            p[0] = max(-1000, min(1000, p[0]))
            p[1] = max(-1000, min(1000, p[1]))
            tp = tuple(p)
            if tp in seen:
                pts[i] = [random.randint(-5, 5), random.randint(-5, 5)]
                tp = tuple(pts[i])
            seen.add(tp)
        return pts

    # Generate multiple initial candidates
    candidates = []
    candidates.append(_initial_4x4())
    for _ in range(5):
        candidates.append(_initial_perturbed())
    for _ in range(3):
        candidates.append(_initial_affine_like())

    best_points = None
    best_score = -1
    for cand in candidates:
        s, valid = _evaluate(cand)
        if valid and s > best_score:
            best_score = s
            best_points = [p[:] for p in cand]

    if best_points is None:
        best_points = _initial_4x4()
        best_score, _ = _evaluate(best_points)

    # Simulated Annealing with multiple restarts
    current_points = [p[:] for p in best_points]
    current_score = best_score

    T = 8.0
    cooling = 0.9995
    min_T = 0.05
    GRID = 12  # Search grid half-size

    step = 0
    no_improve_steps = 0
    max_no_improve = 5000

    while time.time() < deadline:
        step += 1
        move_type = random.random()

        new_points = [p[:] for p in current_points]
        changed_idx = -1

        if move_type < 0.6:
            # Shift a point by small amount
            idx = random.randint(0, 15)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            new_points[idx][0] += dx
            new_points[idx][1] += dy
            if abs(new_points[idx][0]) > 1000 or abs(new_points[idx][1]) > 1000:
                continue
            # Keep within grid to maintain density
            if abs(new_points[idx][0]) > GRID or abs(new_points[idx][1]) > GRID:
                if random.random() > 0.05:
                    continue
            changed_idx = idx

        elif move_type < 0.8:
            # Move one point to a random nearby location
            idx = random.randint(0, 15)
            new_points[idx] = [
                current_points[idx][0] + random.randint(-3, 3),
                current_points[idx][1] + random.randint(-3, 3)
            ]
            if abs(new_points[idx][0]) > 1000 or abs(new_points[idx][1]) > 1000:
                continue
            if abs(new_points[idx][0]) > GRID or abs(new_points[idx][1]) > GRID:
                if random.random() > 0.05:
                    continue
            changed_idx = idx

        elif move_type < 0.9:
            # Swap two points
            i, j = random.sample(range(16), 2)
            new_points[i], new_points[j] = new_points[j][:], new_points[i][:]
            changed_idx = i  # Approximate; full eval needed

        else:
            # Replace a point with random point in grid
            idx = random.randint(0, 15)
            new_points[idx] = [random.randint(-GRID, GRID), random.randint(-GRID, GRID)]
            changed_idx = idx

        # Evaluate
        if changed_idx >= 0 and move_type < 0.8:
            new_score, is_valid = _incremental_eval(new_points, changed_idx)
        else:
            new_score, is_valid = _evaluate(new_points)

        if not is_valid:
            no_improve_steps += 1
        else:
            delta = new_score - current_score
            if delta >= 0 or (T > min_T and random.random() < math.exp(delta / T)):
                current_points = new_points
                current_score = new_score

                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1

        T *= cooling
        if T < min_T:
            T = min_T

        # Restart if stuck
        if no_improve_steps > max_no_improve:
            no_improve_steps = 0
            # Restart from best with perturbation
            current_points = [p[:] for p in best_points]
            current_score = best_score
            # Perturb
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 15)
                current_points[idx] = [
                    current_points[idx][0] + random.randint(-2, 2),
                    current_points[idx][1] + random.randint(-2, 2)
                ]
                if abs(current_points[idx][0]) > 1000 or abs(current_points[idx][1]) > 1000:
                    current_points[idx] = [random.randint(-GRID, GRID), random.randint(-GRID, GRID)]
            s, v = _evaluate(current_points)
            if v:
                current_score = s
            T = 5.0

    # Final polish: try to improve best with targeted moves
    if time.time() < deadline - 5:
        polish_pts = [p[:] for p in best_points]
        polish_score = best_score
        for _ in range(20000):
            if time.time() > deadline - 3:
                break
            idx = random.randint(0, 15)
            orig = [polish_pts[idx][0], polish_pts[idx][1]]
            # Try small shifts
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    polish_pts[idx][0] = orig[0] + dx
                    polish_pts[idx][1] = orig[1] + dy
                    if abs(polish_pts[idx][0]) > 1000 or abs(polish_pts[idx][1]) > 1000:
                        continue
                    s, v = _incremental_eval(polish_pts, idx)
                    if v and s > polish_score:
                        polish_score = s
                        best_points = [p[:] for p in polish_pts]
                        best_score = polish_score
                        break
                    elif v and s == polish_score and random.random() < 0.1:
                        polish_pts[idx][0] = orig[0] + dx
                        polish_pts[idx][1] = orig[1] + dy
                        break
                    else:
                        polish_pts[idx] = orig[:]
            else:
                continue
            break

    return best_points
