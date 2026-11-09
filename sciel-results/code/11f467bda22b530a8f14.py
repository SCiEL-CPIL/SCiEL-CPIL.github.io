"""Model-and-optimize: fix target doubled area A, constrain 29 points to have
cross-product == A with a pivot, then polish via incremental local search."""
import time
import random
from collections import Counter

T_START = time.time()

def _elapsed():
    return time.time() - T_START

def _deadline_ok(margin=8.0):
    return _elapsed() < 90.0 - margin

def _doubled_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        ax, ay = points[i]
        for j in range(i+1, n):
            bx, by = points[j]
            dx1, dy1 = bx-ax, by-ay
            for k in range(j+1, n):
                cx, cy = points[k]
                dx2, dy2 = cx-ax, cy-ay
                det = abs(dx1*dy2 - dy1*dx2)
                if det > 0:
                    areas.append(det)
    return areas

def _valid(pts):
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    for x, y in pts:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _score(pts):
    if not _valid(pts):
        return 0
    return max(Counter(_doubled_areas(pts)).values(), default=0)

def _full_counter(pts):
    return Counter(_doubled_areas(pts))

def _fast_move(points, idx, old_pt, new_pt, counter):
    """In-place: update counter after moving point idx from old_pt to new_pt.
    Returns (new_max, undo_list).  Caller must apply undo_list if rejecting."""
    n = len(points)
    ox, oy = old_pt
    nx, ny = new_pt
    others = [i for i in range(n) if i != idx]
    undo = []
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]
            bx, by = points[j]
            det = abs((ax-ox)*(by-oy) - (ay-oy)*(bx-ox))
            if det > 0:
                old_v = counter.get(det, 0)
                counter[det] = old_v - 1
                if old_v - 1 <= 0:
                    del counter[det]
                undo.append((det, -1, old_v))
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]
            bx, by = points[j]
            det = abs((ax-nx)*(by-ny) - (ay-ny)*(bx-nx))
            if det > 0:
                old_v = counter.get(det, 0)
                counter[det] = old_v + 1
                undo.append((det, 1, old_v))
    mx = max(counter.values(), default=0)
    return mx, undo

def _fast_swap(points, i, j, counter):
    """In-place: update counter after swapping points i and j.
    Returns (new_max, undo_list).  Caller must apply undo_list if rejecting."""
    n = len(points)
    xi, yi = points[i]
    xj, yj = points[j]
    others = [k for k in range(n) if k != i and k != j]
    undo = []
    for a in others:
        for b in others:
            if a >= b:
                continue
            ax, ay = points[a]
            bx, by = points[b]
            det_i = abs((ax-xi)*(by-yi) - (ay-yi)*(bx-xi))
            if det_i > 0:
                old_v = counter.get(det_i, 0)
                counter[det_i] = old_v - 1
                if old_v - 1 <= 0:
                    del counter[det_i]
                undo.append((det_i, -1, old_v))
            det_j = abs((ax-xj)*(by-yj) - (ay-yj)*(bx-xj))
            if det_j > 0:
                old_v = counter.get(det_j, 0)
                counter[det_j] = old_v - 1
                if old_v - 1 <= 0:
                    del counter[det_j]
                undo.append((det_j, -1, old_v))
    for k in others:
        xk, yk = points[k]
        det_ij = abs((xi-xj)*(yk-yj) - (yi-yj)*(xk-xj))
        if det_ij > 0:
            old_v = counter.get(det_ij, 0)
            counter[det_ij] = old_v - 1
            if old_v - 1 <= 0:
                del counter[det_ij]
            undo.append((det_ij, -1, old_v))
    for a in others:
        for b in others:
            if a >= b:
                continue
            ax, ay = points[a]
            bx, by = points[b]
            det_i = abs((ax-xj)*(by-yj) - (ay-yj)*(bx-xj))
            if det_i > 0:
                old_v = counter.get(det_i, 0)
                counter[det_i] = old_v + 1
                undo.append((det_i, 1, old_v))
            det_j = abs((ax-xi)*(by-yi) - (ay-yi)*(bx-xi))
            if det_j > 0:
                old_v = counter.get(det_j, 0)
                counter[det_j] = old_v + 1
                undo.append((det_j, 1, old_v))
    mx = max(counter.values(), default=0)
    return mx, undo

def _construct_for_A(A, rng):
    """Construct 30 points with many triples of doubled area A."""
    directions = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (3, 1),
        (1, 3), (3, 2), (2, 3), (-1, 1), (1, 2), (2, 1), (4, 1), (1, 4),
        (4, 3), (3, 4), (5, 2), (2, 5)
    ]
    rng.shuffle(directions)
    
    best_pts = None
    best_score = 0
    
    pivot_cands = []
    for x in range(-15, 16, 5):
        for y in range(-15, 16, 5):
            pivot_cands.append((x, y))
    pivot_cands.append((0, 0))
    rng.shuffle(pivot_cands)
    
    for pivot in pivot_cands[:10]:
        px, py = pivot
        candidates = []
        for vx, vy in directions:
            if vx == 0:
                if A % vy == 0:
                    y = py - A // vy
                    for x in range(-30, 31):
                        if (x, y) != (px, py) and -30 <= y <= 30:
                            candidates.append((x, y, vx, vy))
            elif vy == 0:
                if A % vx == 0:
                    x = px + A // vx
                    for y in range(-30, 31):
                        if (x, y) != (px, py) and -30 <= x <= 30:
                            candidates.append((x, y, vx, vy))
            else:
                for y in range(-30, 31):
                    num = A + (y - py) * vx
                    if num % vy == 0:
                        x = px + num // vy
                        if -30 <= x <= 30 and (x, y) != (px, py):
                            candidates.append((x, y, vx, vy))
        
        if len(candidates) < 29:
            continue
        
        rng.shuffle(candidates)
        candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
        
        chosen = [(px, py)]
        used = set(chosen)
        for c in candidates:
            pt = (c[0], c[1])
            if pt not in used:
                chosen.append(pt)
                used.add(pt)
            if len(chosen) == 30:
                break
        
        if len(chosen) < 30:
            continue
        
        while len(chosen) < 30:
            rx = rng.randint(-30, 30)
            ry = rng.randint(-30, 30)
            if (rx, ry) not in used:
                chosen.append((rx, ry))
                used.add((rx, ry))
        
        s = _score(chosen)
        if s > best_score:
            best_score = s
            best_pts = list(chosen)
    
    return best_pts, best_score

def _local_search(pts, counter, max_iter=200, margin=10.0):
    """Hill-climb with in-place counter updates.  Returns (pts, score, counter)."""
    n = len(pts)
    best_pts = list(pts)
    best_counter = Counter(counter)
    best_score = max(best_counter.values(), default=0)
    occupied = set(best_pts)

    nbs = [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)]
    rng = random.Random(hash(tuple(pts)) % (2**32))

    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        for idx in order:
            if not _deadline_ok(margin):
                break
            ox, oy = best_pts[idx]
            for dx, dy in nbs:
                nx, ny = ox+dx, oy+dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                np = (nx, ny)
                if np in occupied:
                    continue
                new_s, undo = _fast_move(best_pts, idx, (ox, oy), np, best_counter)
                if new_s > best_score:
                    best_pts[idx] = np
                    best_score = new_s
                    occupied.discard((ox, oy))
                    occupied.add(np)
                    improved = True
                    break
                else:
                    for det, sign, old_v in undo:
                        if old_v > 0:
                            best_counter[det] = old_v
                        else:
                            best_counter.pop(det, None)
            if improved:
                break

        if not improved and _deadline_ok(margin):
            i, j = rng.sample(range(n), 2)
            if best_pts[i] != best_pts[j]:
                new_s, undo = _fast_swap(best_pts, i, j, best_counter)
                if new_s > best_score:
                    best_pts[i], best_pts[j] = best_pts[j], best_pts[i]
                    best_score = new_s
                    improved = True
                else:
                    for det, sign, old_v in undo:
                        if old_v > 0:
                            best_counter[det] = old_v
                        else:
                            best_counter.pop(det, None)

        if not improved:
            break
    return best_pts, best_score, best_counter

def _simulated_anneal(pts, counter, max_iter=300, margin=8.0):
    """Simulated annealing with in-place counter updates."""
    n = len(pts)
    cur_pts = list(pts)
    cur_counter = Counter(counter)
    cur_score = max(cur_counter.values(), default=0)
    best_pts = list(cur_pts)
    best_counter = Counter(cur_counter)
    best_score = cur_score
    occupied = set(cur_pts)

    nbs = [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)]
    rng = random.Random(hash(tuple(pts)) % (2**32))

    T_start = 15.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iter))

    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        T = T_start * (cooling ** it)

        if rng.random() < 0.7:
            idx = rng.randint(0, n-1)
            ox, oy = cur_pts[idx]
            for _ in range(20):
                dx, dy = rng.choice(nbs)
                nx, ny = ox+dx, oy+dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                np = (nx, ny)
                if np in occupied:
                    continue
                new_s, undo = _fast_move(cur_pts, idx, (ox, oy), np, cur_counter)
                delta = new_s - cur_score
                accept = delta > 0 or (delta == 0 and rng.random() < 0.3) or rng.random() < (2.718281828 ** (delta / max(T, 0.01)))
                if accept:
                    cur_pts[idx] = np
                    cur_score = new_s
                    occupied.discard((ox, oy))
                    occupied.add(np)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_pts = list(cur_pts)
                        best_counter = Counter(cur_counter)
                else:
                    for det, sign, old_v in undo:
                        if old_v > 0:
                            cur_counter[det] = old_v
                        else:
                            cur_counter.pop(det, None)
                break
        else:
            i, j = rng.sample(range(n), 2)
            if cur_pts[i] != cur_pts[j]:
                new_s, undo = _fast_swap(cur_pts, i, j, cur_counter)
                delta = new_s - cur_score
                accept = delta > 0 or (delta == 0 and rng.random() < 0.3) or rng.random() < (2.718281828 ** (delta / max(T, 0.01)))
                if accept:
                    cur_pts[i], cur_pts[j] = cur_pts[j], cur_pts[i]
                    cur_score = new_s
                    if cur_score > best_score:
                        best_score = cur_score
                        best_pts = list(cur_pts)
                        best_counter = Counter(cur_counter)
                else:
                    for det, sign, old_v in undo:
                        if old_v > 0:
                            cur_counter[det] = old_v
                        else:
                            cur_counter.pop(det, None)

    return best_pts, best_score, best_counter

def _generate_seeds():
    """Diverse structured seeds."""
    seeds = []
    # 6x5 grid
    seeds.append([(x, y) for x in range(-3, 4) for y in range(-2, 3)])
    # 5x6 grid
    seeds.append([(x, y) for x in range(-2, 3) for y in range(-3, 4)])
    # 10x3
    seeds.append([(x, y) for x in range(-5, 5) for y in range(-1, 2)])
    # 3x10
    seeds.append([(x, y) for x in range(-1, 2) for y in range(-5, 5)])
    # 15x2
    seeds.append([(x, y) for x in range(-7, 8) for y in range(-1, 1)])
    # 2x15
    seeds.append([(x, y) for x in range(-1, 1) for y in range(-7, 8)])
    # Diamond
    pts = []
    for r in range(-7, 8):
        for c in range(-7, 8):
            if abs(r) + abs(c) <= 7 and len(pts) < 30:
                pts.append((r, c))
    if len(pts) == 30:
        seeds.append(pts)
    # Hexagonal-ish
    pts = []
    for r in range(-4, 5):
        for c in range(-4, 5):
            if abs(r) <= 4 and abs(c) <= 4 and len(pts) < 30:
                pts.append((r*2, c*2 + (r % 2)))
    if len(pts) == 30:
        seeds.append(pts)
    # 7x4 grid
    seeds.append([(x, y) for x in range(-3, 4) for y in range(-1, 3)])
    # 4x7 grid
    seeds.append([(x, y) for x in range(-1, 3) for y in range(-3, 4)])
    # 8x4 grid
    seeds.append([(x, y) for x in range(-4, 4) for y in range(-1, 3)])
    # 9x3 grid
    seeds.append([(x, y) for x in range(-4, 5) for y in range(-1, 2)])
    return seeds

def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: score seeds, polish each
    seeds = _generate_seeds()
    for seed in seeds:
        if not _deadline_ok(15):
            break
        if not _valid(seed):
            continue
        s = _score(seed)
        if s >= best_score:
            counter = _full_counter(seed)
            p, _, c = _local_search(seed, counter, max_iter=150, margin=15)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
            # Simulated annealing for harder seeds
            if _deadline_ok(12) and s >= 8:
                pa, sa, ca = _simulated_anneal(seed, counter, max_iter=200, margin=12)
                if sa > best_score:
                    best_score = sa
                    best_pts = list(pa)
    
    # Phase 2: model-and-optimize for various target areas A
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 60, 72]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(12):
            break
        p, s = _construct_for_A(A, rng)
        if p is None:
            continue
        if s >= best_score - 2:
            counter = _full_counter(p)
            pp, _, c = _local_search(p, counter, max_iter=150, margin=12)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(pp)
            # SA on promising target-area constructions
            if _deadline_ok(10) and ps >= best_score - 1:
                pa, sa, ca = _simulated_anneal(pp, c, max_iter=250, margin=10)
                if sa > best_score:
                    best_score = sa
                    best_pts = list(pa)
    
    # Phase 3: random restarts
    for _ in range(40):
        if not _deadline_ok(10):
            break
        pts = []
        used = set()
        for _ in range(30):
            x = rng.randint(-25, 25)
            y = rng.randint(-25, 25)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        s = _score(pts)
        if s >= best_score - 5:
            counter = _full_counter(pts)
            p, _, c = _local_search(pts, counter, max_iter=100, margin=10)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Final polish with SA + local search
    if best_pts is not None and _deadline_ok(8):
        counter = _full_counter(best_pts)
        p, _, c = _simulated_anneal(best_pts, counter, max_iter=300, margin=8)
        ps = max(c.values(), default=0)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
            # High-precision polish: tight local search with ±1 neighborhood
            if _deadline_ok(4):
                c2 = _full_counter(best_pts)
                p2, s2, c3 = _local_search(best_pts, c2, max_iter=200, margin=4)
                if s2 >= best_score:
                    best_pts = list(p2)
                    best_score = s2
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
