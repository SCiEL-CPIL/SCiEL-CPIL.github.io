"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    a = dy
    b = -dx
    c = -a * x1 - b * y1
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _build_line_members(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary = sum(1 for members in lines.values() if len(members) == 2)
    return lines, ordinary

def _delta_count_fast(points, line_members, ordinary_count, idx, new_pos):
    if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
        return None, None

    for j in range(12):
        if j != idx and points[j][0] == new_pos[0] and points[j][1] == new_pos[1]:
            return None, None

    new_pts = [p[:] for p in points]
    new_pts[idx] = list(new_pos)
    if not _valid(new_pts):
        return None, None

    old_lines_with_idx = {}
    for key, members in line_members.items():
        if idx in members:
            old_lines_with_idx[key] = set(members)

    delta = 0
    for key, members in old_lines_with_idx.items():
        k = len(members)
        if k == 2:
            delta -= 1
        elif k == 3:
            delta += 1

    new_line_keys = {}
    for j in range(12):
        if j == idx:
            continue
        key = _line_key(new_pos, points[j])
        if key not in new_line_keys:
            new_line_keys[key] = set()
        new_line_keys[key].add(j)

    old_ordinary_through_idx = 0
    for key, members in old_lines_with_idx.items():
        if len(members) == 2:
            old_ordinary_through_idx += 1

    new_lines_through_idx = {}
    for j in range(12):
        if j == idx:
            continue
        key = _line_key(new_pos, points[j])
        if key not in new_lines_through_idx:
            new_lines_through_idx[key] = set()
        new_lines_through_idx[key].add(j)

    new_ordinary_through_idx = 0
    for key, js in new_lines_through_idx.items():
        a, b, c = key
        total = 1 + len(js)
        for m in range(12):
            if m == idx or m in js:
                continue
            xm, ym = points[m]
            if a * xm + b * ym + c == 0:
                total += 1
        if total == 2:
            new_ordinary_through_idx += 1

    new_ordinary = ordinary_count - old_ordinary_through_idx + new_ordinary_through_idx

    new_line_members = {}
    for key, members in line_members.items():
        if idx not in members:
            new_line_members[key] = set(members)
    for key, js in new_lines_through_idx.items():
        a, b, c = key
        members = {idx} | js
        for m in range(12):
            if m == idx or m in js:
                continue
            xm, ym = points[m]
            if a * xm + b * ym + c == 0:
                members.add(m)
        new_line_members[key] = members

    return new_ordinary, new_line_members

def _count_ordinary(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    ordinary = 0
    for key, cnt in lines.items():
        if cnt == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if cross_i != 0:
                return True
        return False
    return True

def _delta_count(points, line_members, ordinary_count, idx, new_pos):
    new_ordinary, _ = _delta_count_fast(points, line_members, ordinary_count, idx, new_pos)
    return new_ordinary

def _delta_swap(points, line_members, ordinary_count, idx1, idx2):
    if points[idx1][0] == points[idx2][0] and points[idx1][1] == points[idx2][1]:
        return None
    new_points = [p[:] for p in points]
    new_points[idx1], new_points[idx2] = points[idx2][:], points[idx1][:]
    if not _valid(new_points):
        return None
    old_pos1 = points[idx1][:]
    old_pos2 = points[idx2][:]
    new_ordinary, new_line_members = _delta_count_fast(points, line_members, ordinary_count, idx1, old_pos2)
    if new_ordinary is None:
        return None
    new_ordinary, new_line_members = _delta_count_fast(new_points, new_line_members, new_ordinary, idx2, old_pos1)
    if new_ordinary is None:
        return None
    return new_ordinary

def _generate_strategic_candidates(points, idx, limit=300):
    x, y = points[idx]
    candidates = set()
    others = [j for j in range(12) if j != idx]
    for j in others:
        for k in others:
            if j >= k:
                continue
            xj, yj = points[j]
            xk, yk = points[k]
            dx = xk - xj
            dy = yk - yj
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n != 0:
                t_approx = (x - xj) / dx_n
            elif dy_n != 0:
                t_approx = (y - yj) / dy_n
            else:
                continue
            t_set = set()
            for t in range(int(t_approx) - 5, int(t_approx) + 6):
                t_set.add(t)
            for m in others:
                if m == j or m == k:
                    continue
                xm, ym = points[m]
                if dx_n != 0:
                    t_m = (xm - xj) / dx_n
                    for t in range(int(t_m) - 2, int(t_m) + 3):
                        t_set.add(t)
                elif dy_n != 0:
                    t_m = (ym - yj) / dy_n
                    for t in range(int(t_m) - 2, int(t_m) + 3):
                        t_set.add(t)
            for t in t_set:
                nx = xj + t * dx_n
                ny = yj + t * dy_n
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    is_same = False
                    for m in range(12):
                        if m != idx and points[m][0] == nx and points[m][1] == ny:
                            is_same = True
                            break
                    if not is_same:
                        candidates.add((nx, ny))
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                is_same = False
                for m in range(12):
                    if m != idx and points[m][0] == nx and points[m][1] == ny:
                        is_same = True
                        break
                if not is_same:
                    candidates.add((nx, ny))
    for _ in range(20):
        nx = x + random.randint(-20, 20)
        ny = y + random.randint(-20, 20)
        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
            is_same = False
            for m in range(12):
                if m != idx and points[m][0] == nx and points[m][1] == ny:
                    is_same = True
                    break
            if not is_same:
                candidates.add((nx, ny))
    return list(candidates)[:limit]

def _parabola_initial():
    pts = []
    for i in range(11):
        x = i
        y = x * x
        pts.append([x, y])
    pts.append([100, 1])
    return pts

def _affine_parabola_initial(scale=3, offset=0):
    pts = []
    for i in range(11):
        x = i
        y = x * x
        pts.append([scale * x + offset, scale * y + offset])
    pts.append([100 * scale + offset, 1 * scale + offset])
    return pts

def _sa_search(points, line_members, count, deadline, best_points, best_count):
    current = [p[:] for p in points]
    cur_lm, cur_cnt = _build_line_members(current)
    if cur_cnt < best_count:
        best_count = cur_cnt
        best_points = [p[:] for p in current]
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    while time.time() < deadline and T > T_min:
        i = random.randint(0, 11)
        x, y = current[i]
        cand_type = random.random()
        if cand_type < 0.5:
            others = [j for j in range(12) if j != i]
            j = random.choice(others)
            k = random.choice([m for m in others if m != j])
            xj, yj = current[j]
            xk, yk = current[k]
            dx = xk - xj
            dy = yk - yj
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
            else:
                dx_n = dx // g
                dy_n = dy // g
                if dx_n != 0:
                    t_approx = (x - xj) / dx_n
                elif dy_n != 0:
                    t_approx = (y - yj) / dy_n
                else:
                    nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
                    t_approx = None
                if t_approx is not None:
                    t = int(round(t_approx)) + random.randint(-3, 3)
                    nx = xj + t * dx_n
                    ny = yj + t * dy_n
                else:
                    nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
        elif cand_type < 0.75:
            others = [j for j in range(12) if j != i]
            if len(others) >= 4:
                j1, k1 = random.sample(others, 2)
                j2, k2 = random.sample([m for m in others if m not in (j1, k1)], 2)
                xj1, yj1 = current[j1]
                xk1, yk1 = current[k1]
                xj2, yj2 = current[j2]
                xk2, yk2 = current[k2]
                dx1 = xk1 - xj1
                dy1 = yk1 - yj1
                a1, b1 = dy1, -dx1
                c1 = -a1 * xj1 - b1 * yj1
                dx2 = xk2 - xj2
                dy2 = yk2 - yj2
                a2, b2 = dy2, -dx2
                c2 = -a2 * xj2 - b2 * yj2
                det = a1 * b2 - a2 * b1
                if det != 0:
                    nx = (b1 * c2 - b2 * c1) / det
                    ny = (a2 * c1 - a1 * c2) / det
                    nx = int(round(nx))
                    ny = int(round(ny))
                else:
                    nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
            else:
                nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
        else:
            nx = x + random.randint(-10, 10)
            ny = y + random.randint(-10, 10)
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            T *= cooling
            continue
        valid = True
        for m in range(12):
            if m != i and current[m][0] == nx and current[m][1] == ny:
                valid = False
                break
        if not valid:
            T *= cooling
            continue
        new_count = _delta_count(current, cur_lm, cur_cnt, i, (nx, ny))
        if new_count is None:
            T *= cooling
            continue
        delta = new_count - cur_cnt
        if delta < 0 or random.random() < math.exp(-delta / T):
            current[i] = [nx, ny]
            cur_cnt = new_count
            cur_lm, cur_cnt = _build_line_members(current)
            if cur_cnt < best_count:
                best_count = cur_cnt
                best_points = [p[:] for p in current]
                if best_count <= 6:
                    return best_points, best_count, cur_lm
        T *= cooling
    return best_points, best_count, cur_lm

def _two_point_move(points, line_members, count, i, j, deadline):
    best_result = None
    best_cnt = count
    cand_i = _generate_strategic_candidates(points, i, limit=50)
    cand_j = _generate_strategic_candidates(points, j, limit=50)
    for ni in cand_i[:20]:
        if time.time() > deadline:
            break
        for nj in cand_j[:20]:
            if time.time() > deadline:
                break
            if ni[0] == nj[0] and ni[1] == nj[1]:
                continue
            if any(points[m][0] == ni[0] and points[m][1] == ni[1] for m in range(12) if m != i and m != j):
                continue
            if any(points[m][0] == nj[0] and points[m][1] == nj[1] for m in range(12) if m != i and m != j):
                continue
            new_pts = [p[:] for p in points]
            new_pts[i] = list(ni)
            new_pts[j] = list(nj)
            if not _valid(new_pts):
                continue
            new_lm, new_c = _build_line_members(new_pts)
            if new_c < best_cnt:
                best_cnt = new_c
                best_result = (new_pts, new_c, new_lm)
    return best_result

def construct():
    deadline = time.time() + 115
    random.seed(42)
    candidates_list = [
        [[i, 0] for i in range(11)] + [[0, 1]],
        [[x, y] for x in range(4) for y in range(3)],
        [[x, y] for x in range(6) for y in range(2)],
        _parabola_initial(),
        _affine_parabola_initial(scale=2, offset=0),
        _affine_parabola_initial(scale=3, offset=1),
        _affine_parabola_initial(scale=5, offset=-2),
    ]
    best_points = None
    best_count = float('inf')
    best_line_members = None
    for pts in candidates_list:
        if not _valid(pts):
            continue
        line_members, c = _build_line_members(pts)
        if c < best_count:
            best_count = c
            best_points = [p[:] for p in pts]
            best_line_members = line_members
    current_points = best_points
    current_count = best_count
    current_line_members = best_line_members
    stuck_count = 0
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            cand_positions = _generate_strategic_candidates(current_points, i, limit=200)
            best_move = None
            best_move_count = current_count
            for nx, ny in cand_positions:
                if time.time() > deadline:
                    break
                new_count = _delta_count(current_points, current_line_members, current_count, i, (nx, ny))
                if new_count is not None and new_count < best_move_count:
                    best_move_count = new_count
                    best_move = (i, nx, ny, new_count)
            if best_move:
                idx, nx, ny, new_count = best_move
                current_points[idx] = [nx, ny]
                current_count = new_count
                current_line_members, current_count = _build_line_members(current_points)
                improved = True
                stuck_count = 0
                if current_count <= 6:
                    return current_points
        for i in range(12):
            if time.time() > deadline:
                break
            for j in range(i + 1, 12):
                if time.time() > deadline:
                    break
                new_count = _delta_swap(current_points, current_line_members, current_count, i, j)
                if new_count is not None and new_count < current_count:
                    current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
                    current_count = new_count
                    current_line_members, current_count = _build_line_members(current_points)
                    improved = True
                    stuck_count = 0
                    if current_count <= 6:
                        return current_points
        if not improved:
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline:
                        break
                    result = _two_point_move(current_points, current_line_members, current_count, i, j, deadline)
                    if result is not None:
                        current_points, current_count, current_line_members = result
                        improved = True
                        stuck_count = 0
                        if current_count <= 6:
                            return current_points
        if not improved:
            stuck_count += 1
            for i in range(12):
                if time.time() > deadline:
                    break
                x, y = current_points[i]
                for _ in range(30):
                    nx = x + random.randint(-50, 50)
                    ny = y + random.randint(-50, 50)
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_count = _delta_count(current_points, current_line_members, current_count, i, (nx, ny))
                    if new_count is not None and new_count < current_count:
                        current_points[i] = [nx, ny]
                        current_count = new_count
                        current_line_members, current_count = _build_line_members(current_points)
                        improved = True
                        stuck_count = 0
                        break
                if improved:
                    break
            if not improved and stuck_count >= 2:
                sa_deadline = min(deadline, time.time() + 20)
                best_points, best_count, best_line_members = _sa_search(
                    current_points, current_line_members, current_count,
                    sa_deadline, current_points, current_count
                )
                if best_count < current_count:
                    current_points = [p[:] for p in best_points]
                    current_count = best_count
                    current_line_members, current_count = _build_line_members(current_points)
                    stuck_count = 0
                    if current_count <= 6:
                        return current_points
            if not improved and stuck_count >= 5:
                new_pts = []
                for _ in range(12):
                    ri = random.randint(-15, 15)
                    rj = random.randint(-15, 15)
                    new_pts.append([2*ri + rj, rj])
                if _valid(new_pts):
                    line_m, cnt = _build_line_members(new_pts)
                    if cnt < current_count:
                        current_points = new_pts
                        current_count = cnt
                        current_line_members = line_m
                        stuck_count = 0
                else:
                    break
    return current_points
# EVOLVE-BLOCK-END
