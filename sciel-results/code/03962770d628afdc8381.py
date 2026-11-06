import time
import random
import math
from collections import defaultdict

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points.

    Fast scorer: build line -> set of point indices, then count len(set).
    """
    n = 16
    line_pts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n = -dx_n, -dy_n
            a, b = dy_n, -dx_n
            c = a * x1 + b * y1
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 1:
                a, b, c = a // g2, b // g2, c // g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            s = line_pts.get(key)
            if s is None:
                line_pts[key] = {i, j}
            else:
                s.add(i)
                s.add(j)

    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5


def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * x1 + b * y1
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under 150s wall-clock

    rng = random.Random(12345)

    base = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score_points(base)
    best_points = [p[:] for p in base]

    def in_bounds(pt):
        return -1000 <= pt[0] <= 1000 and -1000 <= pt[1] <= 1000

    def distinct_of(current, idx, pt):
        for k, p in enumerate(current):
            if k != idx and p == pt:
                return False
        return True

    def completion_move(current, idx, rng):
        """Try to place current[idx] at the 4th lattice position of a
        3-collinear set among the other 15 points. Returns a candidate
        point or None."""
        others = [k for k in range(16) if k != idx]
        # Collect lines with exactly 3 of the 'others' points.
        lines = defaultdict(set)
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                key = _line_key(current[others[i]], current[others[j]])
                if key is None:
                    continue
                lines[key].add(others[i])
                lines[key].add(others[j])
        cands = []
        for key, s in lines.items():
            if len(s) != 3:
                continue
            a, b, c = key
            # 3 collinear lattice points: find direction and the 4th lattice slot.
            pts = [current[t] for t in s]
            # direction from the line
            dx, dy = None, None
            for p, q in ((pts[0], pts[1]), (pts[0], pts[2]), (pts[1], pts[2])):
                ddx = q[0] - p[0]
                ddy = q[1] - p[1]
                gg = math.gcd(abs(ddx), abs(ddy))
                if gg == 0:
                    continue
                ddx //= gg
                ddy //= gg
                if ddx < 0 or (ddx == 0 and ddy < 0):
                    ddx, ddy = -ddx, -ddy
                dx, dy = ddx, ddy
                break
            if dx is None:
                continue
            # The three points are on an arithmetic lattice along (dx,dy).
            # Project each onto the axis parameter t = (x*dx + y*dy) (integer).
            ts = sorted((p[0] * dx + p[1] * dy) for p in pts)
            # The 4th position must extend the run by one step.
            # Candidate t values: min-1 and max+1.
            for t in (ts[0] - 1, ts[2] + 1):
                # Reconstruct a lattice point with this t that lies on the line.
                # Find base point: pick pts[0], its t0.
                t0 = pts[0][0] * dx + pts[0][1] * dy
                k = t - t0
                # point = p0 + k*(dx,dy) only if p0 has t0 and step is unit.
                # But (dx,dy) is the unit direction; moving k steps changes t by k.
                nx = pts[0][0] + k * dx
                ny = pts[0][1] + k * dy
                # Verify on line and integer (it is).
                if a * nx + b * ny == c and in_bounds([nx, ny]):
                    cands.append([nx, ny])
        if not cands:
            return None
        return rng.choice(cands)

    def sa_search(initial, deadline, use_completion):
        current = [p[:] for p in initial]
        s, h5 = _score_points(current)
        if h5:
            return current, s
        cur_score = s
        T = 0.5
        no_improve = 0
        max_no_improve = 700
        while time.time() < deadline and no_improve < max_no_improve:
            idx = rng.randint(0, 15)
            old = current[idx][:]
            new_pt = None
            if use_completion and rng.random() < 0.30:
                new_pt = completion_move(current, idx, rng)
            if new_pt is None:
                if rng.random() < 0.15:
                    new_pt = [rng.randint(-60, 60), rng.randint(-60, 60)]
                else:
                    step = rng.randint(1, 5)
                    new_pt = [old[0] + rng.randint(-step, step),
                              old[1] + rng.randint(-step, step)]
            if not in_bounds(new_pt) or not distinct_of(current, idx, new_pt):
                continue
            current[idx] = new_pt
            s_new, h5_new = _score_points(current)
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue
            if s_new > cur_score:
                cur_score = s_new
                no_improve = 0
                if s_new > best_score:
                    best_score = s_new
                    best_points = [p[:] for p in current]
            elif s_new == cur_score:
                # accept sideways with some probability to wander
                if T > 0.05 and rng.random() < 0.5:
                    no_improve += 1
                else:
                    current[idx] = old
                    no_improve += 1
            else:
                if T > 0.05 and rng.random() < math.exp((s_new - cur_score) / T):
                    cur_score = s_new
                    no_improve += 1
                else:
                    current[idx] = old
                    no_improve += 1
            T = max(0.05, T * 0.9995)
        return current, cur_score

    def hill_climb(initial, deadline):
        current = [p[:] for p in initial]
        s, h5 = _score_points(current)
        if h5:
            return current, s
        cur_score = s
        while time.time() < deadline:
            improved = False
            for idx in range(16):
                if time.time() > deadline:
                    break
                old = current[idx][:]
                # try a small neighborhood + completion
                tried = set()
                for _ in range(40):
                    if rng.random() < 0.4:
                        np = completion_move(current, idx, rng)
                        if np is None:
                            continue
                    else:
                        step = rng.randint(1, 4)
                        np = [old[0] + rng.randint(-step, step),
                              old[1] + rng.randint(-step, step)]
                    key = tuple(np)
                    if key in tried:
                        continue
                    tried.add(key)
                    if not in_bounds(np) or not distinct_of(current, idx, np):
                        continue
                    current[idx] = np
                    s_new, h5_new = _score_points(current)
                    if h5_new:
                        current[idx] = old
                        continue
                    if s_new > cur_score:
                        cur_score = s_new
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_points = [p[:] for p in current]
                    else:
                        current[idx] = old
                if time.time() > deadline:
                    break
            if not improved:
                break
        return current, cur_score

    # Candidate generators
    def gen_grid():
        return [[x, y] for x in range(4) for y in range(4)]

    def gen_perturbed():
        pts = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(rng.randint(2, 8)):
            idx = rng.randint(0, 15)
            np = [pts[idx][0] + rng.randint(-4, 4),
                  pts[idx][1] + rng.randint(-4, 4)]
            if any(np == p for p in pts):
                continue
            pts[idx] = np
        return pts

    def gen_random_structured():
        bx = rng.randint(-10, 10)
        by = rng.randint(-10, 10)
        pts = []
        while len(pts) < 16:
            if rng.random() < 0.4:
                x = bx + rng.randint(-6, 6)
                y = by + rng.randint(-6, 6)
            else:
                x = rng.randint(-50, 50)
                y = rng.randint(-50, 50)
            if [x, y] not in pts:
                pts.append([x, y])
        return pts

    generators = [gen_grid, gen_perturbed, gen_random_structured]

    restarts = 0
    while time.time() < deadline - 20:
        # Choose starting point: mix of best, grid, and fresh randoms
        r = rng.random()
        if r < 0.45:
            cand = [p[:] for p in best_points]
        elif r < 0.70:
            cand = gen_grid()
        elif r < 0.85:
            cand = gen_perturbed()
        else:
            cand = gen_random_structured()

        # Validate candidate
        if len(set(map(tuple, cand))) != 16:
            cand = gen_grid()
        if any(not in_bounds(p) for p in cand):
            cand = gen_grid()

        if time.time() > deadline - 18:
            break

        # SA with line-completion
        res_pts, res_score = sa_search(cand, deadline - 15, use_completion=True)
        if res_score > best_score:
            best_score = res_score
            best_points = [p[:] for p in res_pts]

        restarts += 1
        if best_score >= 15:
            break

    # Final intensive hill-climb on the best to lock in gains
    if time.time() < deadline - 3:
        res_pts, res_score = hill_climb(best_points, deadline - 1)
        if res_score > best_score:
            best_score = res_score
            best_points = [p[:] for p in res_pts]

    return best_points
