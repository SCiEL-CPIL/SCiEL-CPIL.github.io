"""GA + repair + local-search hybrid for the 16-point 4-collinear maximisation task."""
import time
import math
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _canonical_line_key(x1, y1, x2, y2):
    """Canonical (a,b,c) for line through two points: a*x + b*y = c, gcd=1, a>0 or (a==0,b>0)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    a = dy // g
    b = -dx // g
    c = a * x1 + b * y1
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _score(points):
    """Count lines with exactly 4 points. Returns -1 if any line has >=5."""
    n = 16
    pts = np.array(points, dtype=np.int64)
    xs = pts[:, 0]
    ys = pts[:, 1]

    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(xs[i], ys[i], xs[j], ys[j])
            if k is not None:
                line_set.add(k)

    if not line_set:
        return 0

    keys = list(line_set)
    a_arr = np.array([k[0] for k in keys], dtype=np.int64)
    b_arr = np.array([k[1] for k in keys], dtype=np.int64)
    c_arr = np.array([k[2] for k in keys], dtype=np.int64)

    lhs = a_arr[:, None] * xs[None, :] + b_arr[:, None] * ys[None, :]
    counts = np.sum(lhs == c_arr[:, None], axis=1)

    if np.any(counts >= 5):
        return -1
    return int(np.sum(counts == 4))


def _is_valid(points):
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(int(x)) > 1000 or abs(int(y)) > 1000:
            return False
        t = (int(x), int(y))
        if t in seen:
            return False
        seen.add(t)
    return _score(points) >= 0


def _score_and_lines(points):
    """Return (score, dict of line_key -> set of point indices on that line).
    Returns (-1, None) if invalid (5+ collinear)."""
    n = 16
    pts = points
    xs = np.array([p[0] for p in pts], dtype=np.int64)
    ys = np.array([p[1] for p in pts], dtype=np.int64)

    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(xs[i], ys[i], xs[j], ys[j])
            if k is not None:
                line_map[k].add(i)
                line_map[k].add(j)

    score = 0
    line_points = {}
    for k, idx_set in line_map.items():
        cnt = len(idx_set)
        if cnt >= 5:
            return -1, None
        line_points[k] = idx_set
        if cnt == 4:
            score += 1
    return score, line_points


def _find_completion_candidates(points, line_points, rng, max_candidates=40):
    """Find candidate (nx, ny, line_key) that would complete a 3-point line to 4.
    Only consider lines with exactly 3 of our points; find integer points on that line
    that are not already in our set and within bounds."""
    candidates = []
    three_lines = [(k, idx_set) for k, idx_set in line_points.items() if len(idx_set) == 3]
    if not three_lines:
        return candidates

    rng.shuffle(three_lines)
    existing_set = set((p[0], p[1]) for p in points)

    for k, idx_set in three_lines:
        if len(candidates) >= max_candidates:
            break
        a, b, c = k
        pts_on_line = [points[i] for i in idx_set]
        tried = set()
        for px, py in pts_on_line:
            for dy in range(-6, 7):
                ny = py + dy
                if abs(ny) > 1000:
                    continue
                if a != 0:
                    num = c - b * ny
                    if num % a != 0:
                        continue
                    nx = num // a
                else:
                    if b * ny != c:
                        continue
                    for dx in range(-6, 7):
                        nx2 = px + dx
                        if abs(nx2) > 1000:
                            continue
                        if (nx2, ny) in existing_set or (nx2, ny) in tried:
                            continue
                        tried.add((nx2, ny))
                        candidates.append((nx2, ny, k))
                    continue
                if abs(nx) > 1000:
                    continue
                if (nx, ny) in existing_set or (nx, ny) in tried:
                    continue
                tried.add((nx, ny))
                candidates.append((nx, ny, k))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _local_search_heuristic(points, deadline, rng, max_iter=50000, pert_radius=4,
                            max_no_improve=300):
    """Heuristic local search: 70% line-completion moves, 30% random perturbation."""
    best = [list(p) for p in points]
    best_score, line_points = _score_and_lines(best)
    if best_score < 0:
        best_score = 0
        line_points = None

    existing_set = set((p[0], p[1]) for p in best)
    no_improve = 0

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break

        moved = False
        if line_points is not None and rng.random() < 0.70:
            cands = _find_completion_candidates(best, line_points, rng, max_candidates=15)
            rng.shuffle(cands)
            for nx, ny, line_k in cands:
                if (nx, ny) in existing_set:
                    continue
                for idx in range(16):
                    if best[idx][0] == nx and best[idx][1] == ny:
                        continue
                    old_x, old_y = best[idx]
                    new_pts = [list(p) for p in best]
                    new_pts[idx] = [nx, ny]
                    s, lp = _score_and_lines(new_pts)
                    if s > best_score:
                        existing_set.discard((old_x, old_y))
                        existing_set.add((nx, ny))
                        best = new_pts
                        best_score = s
                        line_points = lp
                        no_improve = 0
                        moved = True
                        break
                if moved:
                    break
        else:
            idx = rng.randint(0, 15)
            x, y = best[idx]
            for _try in range(8):
                dx = rng.randint(-pert_radius, pert_radius)
                dy = rng.randint(-pert_radius, pert_radius)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in existing_set:
                    continue
                new_pts = [list(p) for p in best]
                new_pts[idx] = [nx, ny]
                s, lp = _score_and_lines(new_pts)
                if s > best_score:
                    existing_set.discard((x, y))
                    existing_set.add((nx, ny))
                    best = new_pts
                    best_score = s
                    line_points = lp
                    no_improve = 0
                    moved = True
                    break
        if not moved:
            no_improve += 1

    return best, best_score


def _repair(points, rng, deadline):
    """Repair: repeatedly break any 5+-collinear set by nudging one of its points.
    Prefer nudges that land on a 3-point line (creating a 4-line) and that do not
    break existing 4-lines. Returns (points, score, line_points) or None if it
    cannot repair before the deadline."""
    cur = [list(p) for p in points]
    score, lp = _score_and_lines(cur)
    if score >= 0:
        return cur, score, lp

    existing_set = set((p[0], p[1]) for p in cur)
    for _ in range(300):
        if time.time() > deadline:
            return None
        # Find a bad line (>=5)
        bad = None
        for k, idx_set in lp.items():
            if len(idx_set) >= 5:
                bad = (k, idx_set)
                break
        if bad is None:
            return cur, score, lp
        k, idx_set = bad
        a, b, c = k
        # Build set of lines that currently contain exactly 4 (to try to preserve)
        four_lines = {kk for kk, ss in lp.items() if len(ss) == 4}
        # Candidate nudges: for each point on the bad line, try small integer offsets
        candidates = []
        for idx in idx_set:
            x, y = cur[idx]
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    if (nx, ny) in existing_set:
                        continue
                    # Skip if the new point stays on the bad line
                    if a * nx + b * ny == c:
                        continue
                    candidates.append((idx, nx, ny))
        rng.shuffle(candidates)
        improved = False
        for idx, nx, ny in candidates:
            old_x, old_y = cur[idx]
            new_pts = [list(p) for p in cur]
            new_pts[idx] = [nx, ny]
            s, lpn = _score_and_lines(new_pts)
            if s is None or s < 0:
                continue
            # Accept if it reduces the number of bad lines (i.e. is valid)
            # Prefer moves that also don't reduce score too much
            # Count bad lines in new config
            bad_new = sum(1 for ss in lpn.values() if len(ss) >= 5)
            if bad_new < 1:
                # fully valid: accept
                existing_set.discard((old_x, old_y))
                existing_set.add((nx, ny))
                cur = new_pts
                score = s
                lp = lpn
                improved = True
                break
            # otherwise still has a bad line; only accept if it at least keeps score >= current valid-ish
            # We'll keep searching
        if not improved:
            # Fallback: move a random bad-line point to a random nearby valid spot
            idx = rng.choice(list(idx_set))
            x, y = cur[idx]
            for _t in range(60):
                dx = rng.randint(-8, 8)
                dy = rng.randint(-8, 8)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in existing_set:
                    continue
                if a * nx + b * ny == c:
                    continue
                new_pts = [list(p) for p in cur]
                new_pts[idx] = [nx, ny]
                s, lpn = _score_and_lines(new_pts)
                if s is None or s < 0:
                    continue
                existing_set.discard((x, y))
                existing_set.add((nx, ny))
                cur = new_pts
                score = s
                lp = lpn
                improved = True
                break
        if not improved:
            return None
    return cur, score, lp


def _crossover(p1, p2, rng):
    """Subset crossover: random subset of indices from p1, complement from p2."""
    child = [None] * 16
    subset = set(rng.sample(range(16), rng.randint(6, 10)))
    for i in range(16):
        if i in subset:
            child[i] = [int(p1[i][0]), int(p1[i][1])]
        else:
            child[i] = [int(p2[i][0]), int(p2[i][1])]
    return child


def _perturb(points, rng, radius=3, n_moves=3):
    pts = [list(p) for p in points]
    seen = set((p[0], p[1]) for p in pts)
    for _ in range(n_moves):
        idx = rng.randint(0, 15)
        x, y = pts[idx]
        for _t in range(20):
            dx = rng.randint(-radius, radius)
            dy = rng.randint(-radius, radius)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            if (nx, ny) in seen:
                continue
            seen.discard((x, y))
            pts[idx] = [nx, ny]
            seen.add((nx, ny))
            break
    return pts


def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _make_perturbed_grid(rng, n_perturb, radius):
    grid = _make_grid_4x4()
    pts = [list(p) for p in grid]
    seen = set((p[0], p[1]) for p in pts)
    for _ in range(n_perturb):
        idx = rng.randint(0, 15)
        x, y = pts[idx]
        for _t in range(20):
            dx = rng.randint(-radius, radius)
            dy = rng.randint(-radius, radius)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            if (nx, ny) in seen:
                continue
            seen.discard((x, y))
            pts[idx] = [nx, ny]
            seen.add((nx, ny))
            break
    return pts


def _make_random_points(rng, scale=6):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _ga(pop, best_points, best_score, rng, deadline, pop_size=10):
    """Population-based GA: tournament select, subset crossover, repair, local improve."""
    # pop: list of (points, score)
    pop = [([list(p) for p in pts], s) for pts, s in pop]
    # ensure all entries are valid (score >= 0); repair invalid ones
    clean_pop = []
    for pts, s in pop:
        if s < 0:
            rep = _repair(pts, rng, deadline)
            if rep is None:
                continue
            pts, s, _ = rep
        clean_pop.append((pts, s))
    # top up with fresh randoms
    while len(clean_pop) < pop_size:
        pts = _make_random_points(rng, scale=8)
        s = _score(pts)
        if s < 0:
            rep = _repair(pts, rng, deadline)
            if rep is None:
                continue
            pts, s, _ = rep
        clean_pop.append((pts, s))
    pop = clean_pop[:pop_size]

    # update global best
    for pts, s in pop:
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in pts]

    while time.time() < deadline:
        # selection: two random parents (with bias toward good)
        p1 = rng.choice(pop)[0]
        # tournament of 3 for second parent
        t = rng.sample(pop, min(3, len(pop)))
        p2 = max(t, key=lambda e: e[1])[0]
        child = _crossover(p1, p2, rng)
        s = _score(child)
        if s < 0:
            rep = _repair(child, rng, deadline)
            if rep is None:
                continue
            child, s, _ = rep
        # local improve a bit
        child, s = _local_search_heuristic(child, min(time.time() + 1.5, deadline),
                                           rng, max_iter=2000, pert_radius=3,
                                           max_no_improve=60)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in child]
        # insert into population, drop worst
        pop.append((child, s))
        pop.sort(key=lambda e: e[1])
        pop = pop[-pop_size:]
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    rng = random.Random(42)

    # Generate diverse initial candidates
    candidates = []
    candidates.append(_make_grid_4x4())
    for k in range(1, 5):
        candidates.append(_make_perturbed_grid(rng, k, 2))
    for _ in range(6):
        candidates.append(_make_random_points(rng, scale=8))
    for _ in range(3):
        candidates.append(_make_random_points(rng, scale=4))

    best_points = None
    best_score = -1
    scored = []
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        if s < 0:
            rep = _repair(cand, rng, deadline)
            if rep is None:
                continue
            cand, s, _ = rep
        scored.append((cand, s))
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in cand]

    if best_points is None:
        return _make_grid_4x4()

    # Multi-restart local search with heuristic
    remaining = deadline - time.time()
    n_restarts = max(2, int(remaining / 25))
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        restart_deadline = time.time() + min(18, deadline - time.time() - 3)

        if r == 0:
            start_pts = best_points
        else:
            start_pts = _make_perturbed_grid(rng, rng.randint(1, 4), 3)
            if rng.random() < 0.5:
                start_pts = [list(p) for p in best_points]
                for _ in range(rng.randint(2, 5)):
                    idx = rng.randint(0, 15)
                    dx = rng.randint(-3, 3)
                    dy = rng.randint(-3, 3)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = start_pts[idx][0] + dx, start_pts[idx][1] + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        if (nx, ny) not in [(p[0], p[1]) for i, p in enumerate(start_pts) if i != idx]:
                            start_pts[idx] = [nx, ny]

        pts, s = _local_search_heuristic(start_pts, restart_deadline, rng,
                                         max_iter=40000, pert_radius=4)
        if s > best_score:
            best_score = s
            best_points = pts

    # Genetic algorithm phase with population + repair
    remaining = deadline - time.time()
    if remaining > 15:
        ga_deadline = time.time() + remaining * 0.55
        # build initial population from scored candidates + best
        init_pop = []
        for cand, s in scored[:6]:
            init_pop.append((cand, s))
        init_pop.append((best_points, best_score))
        # add a couple of randoms
        for _ in range(2):
            pts = _make_random_points(rng, scale=8)
            s = _score(pts)
            if s < 0:
                rep = _repair(pts, rng, ga_deadline)
                if rep is None:
                    continue
                pts, s, _ = rep
            init_pop.append((pts, s))
        best_points, best_score = _ga(init_pop, best_points, best_score, rng,
                                      ga_deadline, pop_size=10)

    # Final high-precision polish pass
    remaining = deadline - time.time()
    if remaining > 6:
        polish_deadline = time.time() + min(remaining * 0.8, deadline - 2)
        pts, s = _local_search_heuristic(best_points, polish_deadline, rng,
                                         max_iter=60000, pert_radius=2,
                                         max_no_improve=500)
        if s > best_score:
            best_score = s
            best_points = pts

    # Final validation
    if not _is_valid(best_points):
        return _make_grid_4x4()

    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
