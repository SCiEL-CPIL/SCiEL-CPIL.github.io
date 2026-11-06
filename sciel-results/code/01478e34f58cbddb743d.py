"""Combined: GA population search + line-completion polish + multi-restart."""
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
    """Find candidate (point_index, new_x, new_y) that would complete a 3-point line to 4."""
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


def _find_five_collinear(points):
    """Find indices of any 5+ collinear points. Returns None if valid."""
    n = 16
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if k is not None:
                line_map[k].add(i)
                line_map[k].add(j)
    for k, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return list(idx_set)
    return None


def _repair_five_collinear(points, rng, max_iter=200):
    """Shift points to break 5+ collinear sets while trying to preserve 4-collinear lines."""
    pts = [list(p) for p in points]
    for _ in range(max_iter):
        five_idx = _find_five_collinear(pts)
        if five_idx is None:
            break
        # Pick a random point from the 5-collinear set and move it
        idx = rng.choice(five_idx)
        x, y = pts[idx]
        moved = False
        for _try in range(20):
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            if any(pts[j][0] == nx and pts[j][1] == ny for j in range(16) if j != idx):
                continue
            pts[idx] = [nx, ny]
            # Check if this breaks the 5-collinear
            if _find_five_collinear(pts) is None:
                moved = True
                break
            # If it creates another 5-collinear, revert
            if _find_five_collinear(pts) is not None:
                pts[idx] = [x, y]
                continue
            moved = True
            break
        if not moved:
            # Larger move
            dx = rng.randint(-8, 8)
            dy = rng.randint(-8, 8)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                if not any(pts[j][0] == nx and pts[j][1] == ny for j in range(16) if j != idx):
                    pts[idx] = [nx, ny]
    return pts


def _local_search(points, deadline, rng, max_iter=50000):
    """Local search: 70% line-completion moves, 30% random perturbation."""
    best = [list(p) for p in points]
    best_score, line_points = _score_and_lines(best)
    if best_score < 0:
        best_score = 0
        line_points = None

    existing_set = set((p[0], p[1]) for p in best)
    no_improve = 0
    max_no_improve = 400

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break

        moved = False
        if line_points is not None and rng.random() < 0.70:
            cands = _find_completion_candidates(best, line_points, rng, max_candidates=20)
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
            for _try in range(10):
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
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


def _crossover(parent1, parent2, rng):
    """Crossover: select random subset of indices from parent1, rest from parent2."""
    child = []
    for i in range(16):
        if rng.random() < 0.5:
            child.append(list(parent1[i]))
        else:
            child.append(list(parent2[i]))
    # Ensure uniqueness
    seen = set()
    for i in range(16):
        key = (child[i][0], child[i][1])
        if key in seen:
            # Find a replacement from the other parent
            for j in range(16):
                if i == j:
                    continue
                alt = (parent1[i][0], parent1[i][1]) if (parent2[i][0], parent2[i][1]) == key else (parent2[i][0], parent2[i][1])
                if alt not in seen:
                    child[i] = [alt[0], alt[1]]
                    break
            else:
                # Nudge to make unique
                child[i][0] += 1
                if abs(child[i][0]) > 1000:
                    child[i][0] = -1000
                while (child[i][0], child[i][1]) in seen:
                    child[i][1] += 1
                    if abs(child[i][1]) > 1000:
                        child[i][1] = -1000
        seen.add((child[i][0], child[i][1]))
    return child


def _genetic_algorithm(population, deadline, rng, pop_size=8, max_generations=50):
    """Genetic algorithm with population of 16-point configurations."""
    # Initialize population with scores
    scored_pop = []
    for ind in population:
        pts = [list(p) for p in ind]
        s, lp = _score_and_lines(pts)
        if s < 0:
            s = 0
            lp = None
        scored_pop.append((s, pts, lp))

    best_score = max(s for s, _, _ in scored_pop)
    best_pts = [list(p) for s, p, _ in scored_pop if s == best_score][0]

    gen = 0
    while gen < max_generations and time.time() < deadline:
        gen += 1
        # Sort by score descending
        scored_pop.sort(key=lambda x: -x[0])

        # Selection: tournament
        new_pop = []
        # Elitism: keep top 2
        for i in range(min(2, len(scored_pop))):
            new_pop.append((scored_pop[i][0], [list(p) for p in scored_pop[i][1]], scored_pop[i][2]))

        # Fill rest with crossover + mutation + repair
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
            # Tournament selection of 2 parents
            p1_idx = _tournament_select(scored_pop, rng, k=3)
            p2_idx = _tournament_select(scored_pop, rng, k=3)
            p1 = scored_pop[p1_idx][1]
            p2 = scored_pop[p2_idx][1]

            child = _crossover(p1, p2, rng)

            # Mutation: perturb 1-3 random points
            n_mut = rng.randint(1, 3)
            for _ in range(n_mut):
                idx = rng.randint(0, 15)
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = child[idx][0] + dx, child[idx][1] + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    if not any(child[j][0] == nx and child[j][1] == ny for j in range(16) if j != idx):
                        child[idx] = [nx, ny]

            # Repair 5-collinear
            child = _repair_five_collinear(child, rng, max_iter=100)

            # Score
            s, lp = _score_and_lines(child)
            if s < 0:
                s = 0
                lp = None
            new_pop.append((s, child, lp))

        # Update best
        for s, pts, lp in new_pop:
            if s > best_score:
                best_score = s
                best_pts = [list(p) for p in pts]

        scored_pop = new_pop

    return best_pts, best_score


def _tournament_select(scored_pop, rng, k=3):
    """Tournament selection: pick k random, return index of best."""
    indices = rng.sample(range(len(scored_pop)), min(k, len(scored_pop)))
    best_idx = indices[0]
    best_s = scored_pop[best_idx][0]
    for idx in indices[1:]:
        if scored_pop[idx][0] > best_s:
            best_s = scored_pop[idx][0]
            best_idx = idx
    return best_idx


def _anneal(points, deadline, rng):
    """Simulated annealing with line-completion bias."""
    cur = [list(p) for p in points]
    cur_score, cur_lp = _score_and_lines(cur)
    if cur_score < 0:
        cur_score = 0
        cur_lp = None
    best = [list(p) for p in cur]
    best_score = cur_score
    existing_set = set((p[0], p[1]) for p in cur)
    T = 4.0
    step = 0

    while time.time() < deadline:
        step += 1
        if step % 300 == 0:
            T *= 0.95
            if T < 0.01:
                break

        if cur_lp is not None and rng.random() < 0.5:
            cands = _find_completion_candidates(cur, cur_lp, rng, max_candidates=8)
            if not cands:
                idx = rng.randint(0, 15)
                x, y = cur[idx]
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
            else:
                nx, ny, _ = rng.choice(cands)
                idx = rng.randint(0, 15)
        else:
            idx = rng.randint(0, 15)
            x, y = cur[idx]
            dx = rng.randint(-4, 4)
            dy = rng.randint(-4, 4)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy

        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) in existing_set:
            continue

        new_pts = [list(p) for p in cur]
        new_pts[idx] = [nx, ny]
        s, lp = _score_and_lines(new_pts)
        if s < 0:
            continue
        if s >= cur_score or rng.random() < math.exp((s - cur_score) / max(T, 1e-6)):
            existing_set.discard((cur[idx][0], cur[idx][1]))
            existing_set.add((nx, ny))
            cur = new_pts
            cur_score = s
            cur_lp = lp
            if s > best_score:
                best = [list(p) for p in cur]
                best_score = s

    return best, best_score


def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _make_affine_plane_4x4():
    """
    Attempt to construct a configuration inspired by the affine plane of order 4.
    """
    pts = []
    for x in range(4):
        for y in range(4):
            nx = x + y
            ny = y
            pts.append([nx, ny])
    return pts


def _make_modular_pattern():
    """
    Use a modular arithmetic pattern to create many 4-point lines.
    """
    pts = []
    for i in range(4):
        for j in range(4):
            x = i * 3 + j
            y = i + j * 3
            pts.append([x, y])
    return pts


def _make_perturbed_grid(rng, n_perturb, radius):
    grid = _make_grid_4x4()
    pts = [list(p) for p in grid]
    for _ in range(n_perturb):
        idx = rng.randint(0, 15)
        dx = rng.randint(-radius, radius)
        dy = rng.randint(-radius, radius)
        if dx == 0 and dy == 0:
            continue
        nx, ny = pts[idx][0] + dx, pts[idx][1] + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) in [(p[0], p[1]) for i, p in enumerate(pts) if i != idx]:
            continue
        pts[idx] = [nx, ny]
    return pts


def _make_random_points(rng, scale=8):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _make_sheared_grid(rng, shear):
    """Create a sheared 4x4 grid."""
    pts = []
    for x in range(4):
        for y in range(4):
            nx = x + shear * y
            ny = y
            pts.append([nx, ny])
    return pts


def _make_diagonal_ensemble(rng):
    """Create points on multiple diagonal lines."""
    pts = []
    seen = set()
    slopes = [0, 1, -1, 2, -2, 3, -3]
    for slope in slopes:
        for intercept in range(-2, 3):
            if len(pts) >= 16:
                break
            for x in range(-3, 4):
                y = slope * x + intercept
                if abs(x) <= 1000 and abs(y) <= 1000:
                    if (x, y) not in seen:
                        seen.add((x, y))
                        pts.append([x, y])
                        break
            if len(pts) >= 16:
                break
    while len(pts) < 16:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _polish_exhaustive(points, deadline, rng):
    """High-precision polish: exhaustive single-point moves with wider radius."""
    best = [list(p) for p in points]
    best_score, line_points = _score_and_lines(best)
    if best_score < 0:
        best_score = 0
        line_points = None
    existing_set = set((p[0], p[1]) for p in best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() > deadline:
                break
            x, y = best[idx]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
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
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s kill

    rng = random.Random(42)

    # Generate diverse initial candidates
    candidates = []
    candidates.append(_make_grid_4x4())
    candidates.append(_make_affine_plane_4x4())
    candidates.append(_make_modular_pattern())
    candidates.append(_make_sheared_grid(rng, 1))
    candidates.append(_make_sheared_grid(rng, 2))
    candidates.append(_make_diagonal_ensemble(rng))
    
    for k in range(1, 5):
        candidates.append(_make_perturbed_grid(rng, k, 2))
    for _ in range(6):
        candidates.append(_make_random_points(rng, scale=10))
    for _ in range(3):
        candidates.append(_make_random_points(rng, scale=5))

    best_points = None
    best_score = -1

    # Score all initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in cand]

    if best_score < 0:
        return _make_grid_4x4()

    # Phase 1: Local search on best initial candidate
    ls_deadline = time.time() + min(25, deadline - time.time() - 60)
    pts, s = _local_search(best_points, ls_deadline, rng)
    if s > best_score:
        best_score = s
        best_points = pts

    # Phase 2: Genetic algorithm with population
    # Build population from diverse sources
    ga_pop = []
    ga_pop.append([list(p) for p in best_points])
    ga_pop.append(_make_grid_4x4())
    ga_pop.append(_make_affine_plane_4x4())
    ga_pop.append(_make_modular_pattern())
    ga_pop.append(_make_sheared_grid(rng, 1))
    ga_pop.append(_make_sheared_grid(rng, 2))
    ga_pop.append(_make_perturbed_grid(rng, 3, 3))
    ga_pop.append(_make_diagonal_ensemble(rng))

    # Add perturbed versions of best
    for _ in range(4):
        pert = [list(p) for p in best_points]
        for _ in range(rng.randint(2, 5)):
            idx = rng.randint(0, 15)
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            nx, ny = pert[idx][0] + dx, pert[idx][1] + dy
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                if not any(pert[j][0] == nx and pert[j][1] == ny for j in range(16) if j != idx):
                    pert[idx] = [nx, ny]
        ga_pop.append(pert)

    ga_deadline = time.time() + min(40, deadline - time.time() - 30)
    ga_pts, ga_score = _genetic_algorithm(ga_pop, ga_deadline, rng, pop_size=10, max_generations=80)
    if ga_score > best_score:
        best_score = ga_score
        best_points = ga_pts

    # Phase 3: Local search from GA result
    if time.time() < deadline - 15:
        ls2_deadline = time.time() + min(15, deadline - time.time() - 10)
        pts, s = _local_search(best_points, ls2_deadline, rng)
        if s > best_score:
            best_score = s
            best_points = pts

    # Phase 4: Final annealing
    remaining = deadline - time.time()
    if remaining > 10:
        anneal_deadline = time.time() + remaining * 0.5
        pts, s = _anneal(best_points, anneal_deadline, rng)
        if s > best_score:
            best_score = s
            best_points = pts

    # Phase 5: Final polish
    if time.time() < deadline - 2:
        pts, s = _polish_exhaustive(best_points, deadline - 1, rng)
        if s > best_score:
            best_score = s
            best_points = pts

    # Final validation
    if not _is_valid(best_points):
        return _make_grid_4x4()

    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
