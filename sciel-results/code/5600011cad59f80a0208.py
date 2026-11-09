import time
import math
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill
_BOUND = 1000


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _score(points):
    """Returns (num_exact4, num_5plus)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    n4 = 0
    n5 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
    return n4, n5


def _valid(points):
    """True iff no five are collinear."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            c = counts.get(k, 0) + 1
            counts[k] = c
            if c >= 10:
                return False
    return True


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_affine():
    """Shear transform: (x,y) -> (x+y, y)"""
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_diagonal():
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[5] = [1, 5]
    pts[10] = [2, -1]
    pts[15] = [5, 3]
    return pts


def _seed_hexagonal():
    pts = [[0, 0]]
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]:
        pts.append([dx, dy])
    ring2 = [(2,0),(-2,0),(0,2),(0,-2),(2,1),(-2,-1),(1,2),(-1,-2)]
    for p in ring2:
        pts.append(list(p))
    return pts[:16]


def _seed_mixed():
    pts = []
    for i in range(4):
        for j in range(4):
            if i % 2 == 0:
                pts.append([i, j])
            else:
                pts.append([i + 1, j])
    return pts


def _seed_perturbed_grid():
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_diagonal_grid():
    """Diagonal-shifted grid for more line diversity."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i + j, i - j])
    return pts


def _seed_lattice():
    """3x3 lattice plus 7 extra points."""
    pts = [[x, y] for x in range(3) for y in range(3)]
    extras = [[3, 0], [0, 3], [3, 3], [-1, 1], [1, -1], [2, 4], [4, 2]]
    pts.extend(extras)
    return pts


def _seed_3x3_plus():
    """3x3 grid with 7 strategically placed extras."""
    pts = [[x, y] for x in range(3) for y in range(3)]
    extras = [[3, 1], [1, 3], [3, 3], [-1, 2], [2, -1], [4, 0], [0, 4]]
    pts.extend(extras)
    return pts


def _local_search(points, rng, time_budget):
    """Fast greedy local search with line-targeting moves."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5 = _score(cur)
    best_key = (cur_n4, -cur_n5)
    best_pts = [list(p) for p in cur]
    
    while time.time() < time_budget:
        improved = False
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            best_local = None
            best_local_key = None
            
            # Try a larger window of moves around current position
            w = 15
            for dx in range(-w, w + 1):
                if time.time() >= time_budget:
                    break
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue
                    cand = [list(p) for p in cur]
                    cand[i] = np_
                    if not _valid(cand):
                        continue
                    n4, n5 = _score(cand)
                    key = (n4, -n5)
                    if best_local_key is None or key > best_local_key:
                        best_local_key = key
                        best_local = np_
            
            # Also try line-targeting: find pairs of existing points and place i on their line
            if time.time() < time_budget:
                for a in range(n):
                    if a == i:
                        continue
                    for b in range(a + 1, n):
                        if b == i:
                            continue
                        # Check if placing i on line (a,b) gives exactly 4 collinear
                        ax, ay = cur[a]
                        bx, by = cur[b]
                        dx = bx - ax
                        dy = by - ay
                        g = gcd(abs(dx), abs(dy))
                        if g == 0:
                            continue
                        dx_n, dy_n = dx // g, dy // g
                        # Try positions along this line: a + t*(dx_n, dy_n) for small t
                        for t in range(-5, 6):
                            if t == 0:
                                continue
                            nx = ax + t * dx_n
                            ny = ay + t * dy_n
                            if abs(nx) > _BOUND or abs(ny) > _BOUND:
                                continue
                            np_ = [nx, ny]
                            if any(np_ == p for p in cur):
                                continue
                            cand = [list(p) for p in cur]
                            cand[i] = np_
                            if not _valid(cand):
                                continue
                            n4, n5 = _score(cand)
                            key = (n4, -n5)
                            if best_local_key is None or key > best_local_key:
                                best_local_key = key
                                best_local = np_
                            if time.time() >= time_budget:
                                break
            
            if best_local is not None:
                cur_key = (cur_n4, -cur_n5)
                if best_local_key > cur_key:
                    cur[i] = best_local
                    cur_n4, cur_n5 = best_local_key[0], -best_local_key[1]
                    improved = True
                    if (cur_n4, -cur_n5) > best_key:
                        best_key = (cur_n4, -cur_n5)
                        best_pts = [list(p) for p in cur]
        
        if not improved:
            # Random kick to escape local optimum - larger range
            i = rng.randrange(n)
            x0, y0 = cur[i]
            found = False
            for _ in range(300):
                nx = x0 + rng.randint(-15, 15)
                ny = y0 + rng.randint(-15, 15)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue
                cand = [list(p) for p in cur]
                cand[i] = np_
                if not _valid(cand):
                    continue
                n4, n5 = _score(cand)
                key = (n4, -n5)
                cur_key = (cur_n4, -cur_n5)
                if key >= cur_key:
                    cur[i] = np_
                    cur_n4, cur_n5 = n4, n5
                    if (cur_n4, -cur_n5) > best_key:
                        best_key = (cur_n4, -cur_n5)
                        best_pts = [list(p) for p in cur]
                    found = True
                    break
            if not found:
                break
    return best_pts, best_key[0]


def _sa_polish(points, rng, time_budget):
    """Simulated annealing polish with line-targeting moves."""
    cur = [list(p) for p in points]
    cur_n4, cur_n5 = _score(cur)
    cur_score = cur_n4 if cur_n5 == 0 else cur_n4 - 100 * cur_n5
    best_pts = [list(p) for p in cur]
    best_score = cur_score
    
    T = 5.0
    T_min = 0.05
    alpha = 0.9992
    
    while time.time() < time_budget and T > T_min:
        i = rng.randrange(16)
        ox, oy = cur[i]
        
        # 40% random, 40% line-targeting, 20% triple-line targeting
        r = rng.random()
        if r < 0.4:
            nx = ox + rng.randint(-12, 12)
            ny = oy + rng.randint(-12, 12)
        elif r < 0.8:
            # Line-targeting: pick two other points and try to align
            a = rng.randrange(16)
            if a == i:
                a = (a + 1) % 16
            b = rng.randrange(16)
            if b == i or b == a:
                b = (b + 1) % 16
                if b == i:
                    b = (b + 1) % 16
            ax, ay = cur[a]
            bx, by = cur[b]
            dx = bx - ax
            dy = by - ay
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                nx = ox + rng.randint(-5, 5)
                ny = oy + rng.randint(-5, 5)
            else:
                dx_n, dy_n = dx // g, dy // g
                t = rng.randint(-5, 5)
                if t == 0:
                    t = 1
                nx = ax + t * dx_n
                ny = ay + t * dy_n
        else:
            # Triple-line: pick three points, find if any pair's line has a good spot
            a = rng.randrange(16)
            if a == i:
                a = (a + 1) % 16
            b = rng.randrange(16)
            if b == i or b == a:
                b = (b + 1) % 16
                if b == i:
                    b = (b + 1) % 16
            ax, ay = cur[a]
            bx, by = cur[b]
            dx = bx - ax
            dy = by - ay
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                nx = ox + rng.randint(-6, 6)
                ny = oy + rng.randint(-6, 6)
            else:
                dx_n, dy_n = dx // g, dy // g
                t = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
                nx = ax + t * dx_n
                ny = ay + t * dy_n
        
        if abs(nx) > _BOUND or abs(ny) > _BOUND:
            T *= alpha
            continue
        np_ = [nx, ny]
        if any(np_ == p for p in cur):
            T *= alpha
            continue
        cand = [list(p) for p in cur]
        cand[i] = np_
        if not _valid(cand):
            T *= alpha
            continue
        n4, n5 = _score(cand)
        new_score = n4 if n5 == 0 else n4 - 100 * n5
        
        if new_score > cur_score:
            cur = cand
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur]
        else:
            delta = cur_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < math.exp(-delta / T)
            if accept:
                cur = cand
                cur_score = new_score
        T *= alpha
    
    return best_pts, best_score


def _final_polish(points, rng, time_budget):
    """High-precision final polish with tighter tolerances and more iterations."""
    cur = [list(p) for p in points]
    cur_n4, cur_n5 = _score(cur)
    cur_score = cur_n4 if cur_n5 == 0 else cur_n4 - 100 * cur_n5
    best_pts = [list(p) for p in cur]
    best_score = cur_score
    
    T = 2.0
    T_min = 0.01
    alpha = 0.9996  # slower cooling for more iterations at low T
    
    while time.time() < time_budget and T > T_min:
        i = rng.randrange(16)
        ox, oy = cur[i]
        
        # 35% random, 45% line-targeting, 20% wide line-targeting
        r = rng.random()
        if r < 0.35:
            nx = ox + rng.randint(-8, 8)
            ny = oy + rng.randint(-8, 8)
        elif r < 0.8:
            a = rng.randrange(16)
            if a == i:
                a = (a + 1) % 16
            b = rng.randrange(16)
            if b == i or b == a:
                b = (b + 1) % 16
                if b == i:
                    b = (b + 1) % 16
            ax, ay = cur[a]
            bx, by = cur[b]
            dx = bx - ax
            dy = by - ay
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                nx = ox + rng.randint(-3, 3)
                ny = oy + rng.randint(-3, 3)
            else:
                dx_n, dy_n = dx // g, dy // g
                t = rng.randint(-4, 4)
                if t == 0:
                    t = 1
                nx = ax + t * dx_n
                ny = ay + t * dy_n
        else:
            # Wide line-targeting with larger steps
            a = rng.randrange(16)
            if a == i:
                a = (a + 1) % 16
            b = rng.randrange(16)
            if b == i or b == a:
                b = (b + 1) % 16
                if b == i:
                    b = (b + 1) % 16
            ax, ay = cur[a]
            bx, by = cur[b]
            dx = bx - ax
            dy = by - ay
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                nx = ox + rng.randint(-5, 5)
                ny = oy + rng.randint(-5, 5)
            else:
                dx_n, dy_n = dx // g, dy // g
                t = rng.choice([-5, -4, -3, 3, 4, 5])
                nx = ax + t * dx_n
                ny = ay + t * dy_n
        
        if abs(nx) > _BOUND or abs(ny) > _BOUND:
            T *= alpha
            continue
        np_ = [nx, ny]
        if any(np_ == p for p in cur):
            T *= alpha
            continue
        cand = [list(p) for p in cur]
        cand[i] = np_
        if not _valid(cand):
            T *= alpha
            continue
        n4, n5 = _score(cand)
        new_score = n4 if n5 == 0 else n4 - 100 * n5
        
        if new_score > cur_score:
            cur = cand
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur]
        else:
            delta = cur_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < math.exp(-delta / T)
            if accept:
                cur = cand
                cur_score = new_score
        T *= alpha
    
    return best_pts, best_score


def construct():
    t_start = time.time()
    rng = random.Random(12345)
    
    seeds = [
        _seed_grid(),
        _seed_staggered(),
        _seed_affine(),
        _seed_diagonal(),
        _seed_hexagonal(),
        _seed_mixed(),
        _seed_perturbed_grid(),
        _seed_diagonal_grid(),
        _seed_lattice(),
        _seed_3x3_plus(),
    ]
    
    # Filter valid seeds
    valid_seeds = []
    for s in seeds:
        if len(set(tuple(p) for p in s)) == 16 and _valid(s):
            valid_seeds.append(s)
    
    if not valid_seeds:
        valid_seeds = [_seed_grid()]
    
    best_pts = None
    best_score = -9999
    
    # Phase 1: Local search on each seed
    n_seeds = len(valid_seeds)
    # Allocate time: 45% local search, 35% SA polish, 15% final polish, 5% buffer
    ls_budget_total = 60.0
    sa_budget_total = 45.0
    fp_budget_total = 20.0
    per_seed_ls = ls_budget_total / n_seeds
    
    for i, seed in enumerate(valid_seeds):
        if time.time() > _DEADLINE - 25:
            break
        budget = min(per_seed_ls, _DEADLINE - time.time() - 25)
        if budget < 1.0:
            break
        pts, sc = _local_search(seed, rng, t_start + budget)
        if sc > best_score:
            best_score = sc
            best_pts = [list(p) for p in pts]
    
    # Phase 2: SA polish on the best
    if best_pts is not None and time.time() < _DEADLINE - 15:
        sa_budget = min(sa_budget_total, _DEADLINE - time.time() - 15)
        if sa_budget > 2.0:
            pts, sc = _sa_polish(best_pts, rng, t_start + sa_budget)
            if sc > best_score:
                best_score = sc
                best_pts = [list(p) for p in pts]
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        fp_budget = min(fp_budget_total, _DEADLINE - time.time() - 5)
        if fp_budget > 1.0:
            pts, sc = _final_polish(best_pts, rng, t_start + fp_budget)
            if sc > best_score:
                best_score = sc
                best_pts = [list(p) for p in pts]
    
    # Phase 4: One more local search if time remains
    if best_pts is not None and time.time() < _DEADLINE - 3:
        final_ls_budget = min(10.0, _DEADLINE - time.time() - 3)
        if final_ls_budget > 1.0:
            pts, sc = _local_search(best_pts, rng, t_start + final_ls_budget)
            if sc > best_score:
                best_score = sc
                best_pts = [list(p) for p in pts]
    
    if best_pts is None:
        return [[x, y] for x in range(4) for y in range(4)]
    
    # Final validation
    final_pts = [list(p) for p in best_pts]
    if not _valid(final_pts) or len(set(tuple(p) for p in final_pts)) != 16:
        return [[x, y] for x in range(4) for y in range(4)]
    
    return final_pts


if __name__ == "__main__":
    r = construct()
    print(len(r), r[:4])
    print(_score(r))
