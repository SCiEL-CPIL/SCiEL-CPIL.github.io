"""
Combined approach: structured seeds + target-area construction + simulated annealing
with incremental scoring to escape local minima, followed by hill-climb polish.
"""
import time
import random
from collections import Counter
import numpy as np

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
    if len(pts) != 30: return False
    if len(set(pts)) != 30: return False
    for x, y in pts:
        if not (isinstance(x, int) and isinstance(y, int)): return False
        if x < -30 or x > 30 or y < -30 or y > 30: return False
    return True

def _score(pts):
    if not _valid(pts): return 0
    return max(Counter(_doubled_areas(pts)).values(), default=0)

def _full_counter(pts):
    n = len(pts)
    areas = []
    for i in range(n):
        ax, ay = pts[i]
        for j in range(i+1, n):
            bx, by = pts[j]
            dx1, dy1 = bx-ax, by-ay
            for k in range(j+1, n):
                cx, cy = pts[k]
                dx2, dy2 = cx-ax, cy-ay
                det = abs(dx1*dy2 - dy1*dx2)
                if det > 0:
                    areas.append(det)
    return Counter(areas)

def _fast_move(points, idx, old_pt, new_pt, counter):
    """Update counter after moving point idx from old_pt to new_pt.
    Returns (new_counter, new_max)."""
    n = len(points)
    c = Counter(counter)
    ox, oy = old_pt
    nx, ny = new_pt
    others = [i for i in range(n) if i != idx]
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]; bx, by = points[j]
            det = abs((ax-ox)*(by-oy) - (ay-oy)*(bx-ox))
            if det > 0:
                c[det] -= 1
                if c[det] <= 0: del c[det]
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]; bx, by = points[j]
            det = abs((ax-nx)*(by-ny) - (ay-ny)*(bx-nx))
            if det > 0:
                c[det] += 1
    mx = max(c.values(), default=0)
    return c, mx

def _local_search(pts, counter, max_iter=200, margin=10.0):
    """Hill-climb: move one point at a time, accept if score improves."""
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
            if not _deadline_ok(margin): break
            ox, oy = best_pts[idx]
            for dx, dy in nbs:
                nx, ny = ox+dx, oy+dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30: continue
                np = (nx, ny)
                if np in occupied: continue
                new_c, new_s = _fast_move(best_pts, idx, (ox, oy), np, best_counter)
                if new_s > best_score:
                    best_pts[idx] = np
                    best_counter = new_c
                    best_score = new_s
                    occupied.discard((ox, oy))
                    occupied.add(np)
                    improved = True
                    break
            if improved: break
        
        if not improved:
            # Try swap moves
            for _ in range(5):
                if not _deadline_ok(margin): break
                i, j = rng.sample(range(n), 2)
                pi, pj = best_pts[i], best_pts[j]
                for dx1, dy1 in nbs[:15]:
                    for dx2, dy2 in nbs[:15]:
                        ni = (pi[0]+dx1, pi[1]+dy1)
                        nj = (pj[0]+dx2, pj[1]+dy2)
                        if ni[0] < -30 or ni[0] > 30 or ni[1] < -30 or ni[1] > 30: continue
                        if nj[0] < -30 or nj[0] > 30 or nj[1] < -30 or nj[1] > 30: continue
                        if ni in occupied or nj in occupied: continue
                        if ni == nj: continue
                        c1, s1 = _fast_move(best_pts, i, pi, ni, best_counter)
                        c2, s2 = _fast_move(best_pts, j, pj, nj, c1)
                        if s2 > best_score:
                            best_pts[i] = ni
                            best_pts[j] = nj
                            best_counter = c2
                            best_score = s2
                            occupied.discard(pi)
                            occupied.discard(pj)
                            occupied.add(ni)
                            occupied.add(nj)
                            improved = True
                            break
                    if improved: break
                if improved: break
        
        if not improved:
            break
    return best_pts, best_score, best_counter

def _simulated_annealing(pts, counter, time_budget=15.0, margin=10.0):
    """Simulated annealing with incremental scoring to escape local minima."""
    n = len(pts)
    current_pts = list(pts)
    current_counter = Counter(counter)
    current_score = max(current_counter.values(), default=0)
    
    best_pts = list(current_pts)
    best_counter = Counter(current_counter)
    best_score = current_score
    occupied = set(current_pts)
    
    rng = random.Random(hash(tuple(pts)) % (2**32))
    
    nbs_small = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)]
    nbs_large = [(dx, dy) for dx in range(-10, 11) for dy in range(-10, 11) if (dx, dy) != (0, 0)]
    
    start_time = time.time()
    t0 = 300.0
    t_end = 1.0
    cooling = (t0 - t_end) / max(1, int(time_budget * 50))
    
    step = 0
    while time.time() - start_time < time_budget and _deadline_ok(margin):
        temp = t0 - cooling * step
        if temp < t_end:
            temp = t_end
        
        # Choose neighborhood based on temperature
        if temp > 100:
            nbs = nbs_large
        else:
            nbs = nbs_small
        
        # Pick a random point to move
        idx = rng.randint(0, n-1)
        ox, oy = current_pts[idx]
        
        # Pick a random neighbor
        dx, dy = rng.choice(nbs)
        nx, ny = ox+dx, oy+dy
        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            step += 1
            continue
        np = (nx, ny)
        if np in occupied:
            step += 1
            continue
        
        new_c, new_s = _fast_move(current_pts, idx, (ox, oy), np, current_counter)
        delta = new_s - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 0.1)):
            # Accept the move
            current_pts[idx] = np
            current_counter = new_c
            current_score = new_s
            occupied.discard((ox, oy))
            occupied.add(np)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = list(current_pts)
                best_counter = Counter(current_counter)
        else:
            # Reject: restore state (we already computed new_c, so just discard)
            pass
        
        step += 1
    
    return best_pts, best_score, best_counter

def _construct_for_A(A, rng):
    """Construct 30 points with many triples of doubled area A."""
    best_pts = None
    best_score = 0
    
    line_configs = [
        [(0, b) for b in range(-10, 11, 2)],
        [(1, b) for b in range(-10, 11, 2)],
        [(-1, b) for b in range(-10, 11, 2)],
        [(2, b) for b in range(-8, 9, 2)],
        [(-2, b) for b in range(-8, 9, 2)],
    ]
    
    for config in line_configs:
        if not _deadline_ok(10):
            break
        
        candidates = []
        for slope, intercept in config:
            for x in range(-30, 31):
                y = slope * x + intercept
                if -30 <= y <= 30:
                    candidates.append((x, y))
        
        for _ in range(20):
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            candidates.append((x, y))
        
        candidates = list(set(candidates))
        
        if len(candidates) < 30:
            continue
        
        rng.shuffle(candidates)
        chosen = []
        used = set()
        
        for _ in range(5):
            if candidates:
                p = candidates.pop(rng.randint(0, len(candidates)-1))
                chosen.append(p)
                used.add(p)
        
        for _ in range(25):
            if not candidates:
                break
            best_p = None
            best_gain = -1
            for p in candidates[:50]:
                if p in used:
                    continue
                gain = 0
                for i in range(len(chosen)):
                    for j in range(i+1, len(chosen)):
                        ax, ay = chosen[i]
                        bx, by = chosen[j]
                        px, py = p
                        det = abs((ax-px)*(by-py) - (ay-py)*(bx-px))
                        if det == A:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p is not None:
                chosen.append(best_p)
                used.add(best_p)
                if best_p in candidates:
                    candidates.remove(best_p)
        
        while len(chosen) < 30 and candidates:
            p = candidates.pop(rng.randint(0, len(candidates)-1))
            if p not in used:
                chosen.append(p)
                used.add(p)
        
        if len(chosen) < 30:
            continue
        
        s = _score(chosen)
        if s > best_score:
            best_score = s
            best_pts = list(chosen)
    
    if best_pts is None or best_score < 5:
        directions = [
            (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (3, 1),
            (1, 3), (3, 2), (2, 3), (-1, 1)
        ]
        rng.shuffle(directions)
        
        pivot_cands = [(x, y) for x in range(-15, 16, 5) for y in range(-15, 16, 5)]
        rng.shuffle(pivot_cands)
        
        for pivot in pivot_cands[:6]:
            if not _deadline_ok(8):
                break
            px, py = pivot
            candidates = []
            for vx, vy in directions:
                if vx == 0:
                    if A % vy == 0:
                        y = py - A // vy
                        for x in range(-30, 31):
                            if (x, y) != (px, py) and -30 <= y <= 30:
                                candidates.append((x, y))
                elif vy == 0:
                    if A % vx == 0:
                        x = px + A // vx
                        for y in range(-30, 31):
                            if (x, y) != (px, py) and -30 <= x <= 30:
                                candidates.append((x, y))
                else:
                    for y in range(-30, 31):
                        num = A + (y - py) * vx
                        if num % vy == 0:
                            x = px + num // vy
                            if -30 <= x <= 30 and (x, y) != (px, py):
                                candidates.append((x, y))
            
            if len(candidates) < 29:
                continue
            
            rng.shuffle(candidates)
            candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
            
            chosen = [(px, py)]
            used = set(chosen)
            for c in candidates:
                if c not in used:
                    chosen.append(c)
                    used.add(c)
                if len(chosen) == 30:
                    break
            
            if len(chosen) < 30:
                continue
            
            s = _score(chosen)
            if s > best_score:
                best_score = s
                best_pts = list(chosen)
    
    return best_pts, best_score

def _generate_seeds():
    """Diverse structured seeds."""
    seeds = []
    seeds.append([(x, y) for x in range(-3, 4) for y in range(-2, 3)])
    seeds.append([(x, y) for x in range(-2, 3) for y in range(-3, 4)])
    seeds.append([(x, y) for x in range(-5, 5) for y in range(-1, 2)])
    seeds.append([(x, y) for x in range(-1, 2) for y in range(-5, 5)])
    seeds.append([(x, y) for x in range(-7, 8) for y in range(-1, 1)])
    seeds.append([(x, y) for x in range(-1, 1) for y in range(-7, 8)])
    
    pts = []
    for r in range(-7, 8):
        for c in range(-7, 8):
            if abs(r) + abs(c) <= 7 and len(pts) < 30:
                pts.append((r, c))
    if len(pts) == 30: seeds.append(pts)
    
    pts = []
    for r in range(-4, 5):
        for c in range(-4, 5):
            if abs(r) <= 4 and abs(c) <= 4 and len(pts) < 30:
                pts.append((r*2, c*2 + (r % 2)))
    if len(pts) == 30: seeds.append(pts)
    
    # Additional structured seeds
    # Hexagonal packing
    pts = []
    for row in range(-3, 4):
        for col in range(-5, 6):
            x = col + (row % 2) * 0
            y = row
            pts.append((x*2, y*2))
    if len(pts) >= 30:
        pts = pts[:30]
        if _valid(pts):
            seeds.append(pts)
    
    # Lattice with offset
    pts = []
    for i in range(6):
        for j in range(5):
            x = i - 3
            y = j - 2 + (i % 2)
            pts.append((x, y))
    if _valid(pts):
        seeds.append(pts)
    
    return seeds

def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: score seeds, polish each with SA + local search
    seeds = _generate_seeds()
    for seed in seeds:
        if not _deadline_ok(15): break
        if not _valid(seed): continue
        s = _score(seed)
        if s >= best_score:
            counter = _full_counter(seed)
            # Simulated annealing first to escape local minima
            p, sa_score, sa_counter = _simulated_annealing(seed, counter, time_budget=8.0, margin=15)
            # Then local search polish
            pp, ps, c = _local_search(p, sa_counter, max_iter=80, margin=15)
            if ps > best_score:
                best_score = ps
                best_pts = list(pp)
    
    # Phase 2: model-and-optimize for various target areas A
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(8): break
        p, s = _construct_for_A(A, rng)
        if p is None: continue
        if s >= best_score - 5:
            counter = _full_counter(p)
            # SA then local search
            pp, sa_score, sa_counter = _simulated_annealing(p, counter, time_budget=6.0, margin=8)
            ppp, ps, c = _local_search(pp, sa_counter, max_iter=100, margin=8)
            if ps > best_score:
                best_score = ps
                best_pts = list(ppp)
    
    # Phase 3: structured random restarts
    for _ in range(10):
        if not _deadline_ok(6): break
        pts = []
        used = set()
        base_x = rng.randint(-5, 5)
        base_y = rng.randint(-5, 5)
        w = rng.randint(3, 6)
        h = rng.randint(3, 6)
        for i in range(w):
            for j in range(h):
                x = base_x + i
                y = base_y + j
                if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in used:
                    pts.append((x, y))
                    used.add((x, y))
        while len(pts) < 30:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        s = _score(pts)
        if s >= best_score - 5:
            counter = _full_counter(pts)
            pp, sa_score, sa_counter = _simulated_annealing(pts, counter, time_budget=5.0, margin=6)
            ppp, ps, c = _local_search(pp, sa_counter, max_iter=80, margin=6)
            if ps > best_score:
                best_score = ps
                best_pts = list(ppp)
    
    # Final polish
    if best_pts is not None and _deadline_ok(6):
        counter = _full_counter(best_pts)
        p, sa_score, sa_counter = _simulated_annealing(best_pts, counter, time_budget=8.0, margin=6)
        pp, ps, c = _local_search(p, sa_counter, max_iter=150, margin=6)
        if ps >= best_score:
            best_pts = list(pp)
            best_score = ps
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
