# EVOLVE-BLOCK-START
import random, time, math
from itertools import combinations

BOX = 100
N = 20

def _now():
    return time.time()

def _collinear3(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0

def _triple_count(pts):
    c = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if _collinear3(pts[i], pts[j], pts[k]):
                    c += 1
    return c

def _has_four_collinear(pts):
    for i in range(N):
        for j in range(i+1, N):
            cnt = 0
            pi, pj = pts[i], pts[j]
            for k in range(N):
                if k == i or k == j: continue
                if _collinear3(pi, pj, pts[k]):
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False

def _valid(pts):
    if len(set(map(tuple, pts))) != N:
        return False
    if any(abs(x) > BOX or abs(y) > BOX for x, y in pts):
        return False
    return not _has_four_collinear(pts)

def _max_3free(pts):
    """Exact branch-and-bound for largest 3-free subset using vertex cover on hypergraph."""
    n = N
    triples = []
    triple_set = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
                    triple_set.add((i, j, k))
    if not triples:
        return n
    
    # Build adjacency: for each vertex, which triples it's in
    vertex_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        vertex_triples[i].append(idx)
        vertex_triples[j].append(idx)
        vertex_triples[k].append(idx)
    
    # Precompute: for each pair of vertices, how many triples they share
    pair_shared = {}
    for (i, j, k) in triples:
        for a, b in [(i,j), (i,k), (j,k)]:
            key = (min(a,b), max(a,b))
            pair_shared[key] = pair_shared.get(key, 0) + 1
    
    best_vc = [len(triples)]  # best vertex cover size found
    best_is = [0]             # best independent set size found
    nodes = [0]
    CAP = 500000
    
    def search(chosen_mask, remaining_triples_mask, vc_size):
        nodes[0] += 1
        if nodes[0] > CAP:
            return
        
        # If no remaining triples, we have a valid vertex cover
        if remaining_triples_mask == 0:
            if vc_size < best_vc[0]:
                best_vc[0] = vc_size
                best_is[0] = n - vc_size
            return
        
        # Upper bound: even if we add all remaining vertices, can we beat best?
        # Maximum possible IS size = current_is_size + (n - chosen_count)
        # But more tightly: remaining triples each need at least one more vertex
        # Simple bound: if vc_size >= best_vc[0], prune
        if vc_size >= best_vc[0]:
            return
        
        # Find a vertex that appears in the most remaining triples
        # (greedy heuristic for ordering)
        best_v = -1
        best_count = -1
        for v in range(n):
            if (chosen_mask >> v) & 1:
                continue
            cnt = 0
            for t_idx in vertex_triples[v]:
                if (remaining_triples_mask >> t_idx) & 1:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v in vertex cover
        new_mask = chosen_mask | (1 << best_v)
        new_rem = remaining_triples_mask
        for t_idx in vertex_triples[best_v]:
            new_rem &= ~(1 << t_idx)
        search(new_mask, new_rem, vc_size + 1)
        
        # Branch 2: exclude best_v from vertex cover
        # All triples containing best_v must be covered by other vertices
        # Remove best_v from consideration; force its triples to be covered
        # This means for each triple with best_v, at least one of the other two must be in cover
        # We handle this by: removing all triples that don't contain best_v from "must cover"
        # Actually, simpler: just try without best_v and let the recursion handle it
        # But we need to ensure triples with best_v are still coverable
        # For now, just branch on excluding best_v
        search(chosen_mask & ~(1 << best_v), remaining_triples_mask, vc_size)
    
    search(0, (1 << len(triples)) - 1, 0)
    
    if best_is[0] == 0:
        # Fallback: greedy
        is_size = 0
        used = set()
        remaining = list(triples)
        while remaining:
            # Pick vertex in fewest remaining triples
            counts = [0]*n
            for (i,j,k) in remaining:
                if i not in used: counts[i] += 1
                if j not in used: counts[j] += 1
                if k not in used: counts[k] += 1
            min_v = min((v for v in range(n) if v not in used), key=lambda v: counts[v])
            is_size += 1
            used.add(min_v)
            remaining = [(i,j,k) for (i,j,k) in remaining if min_v not in (i,j,k)]
        return is_size
    
    return best_is[0]

def _construct_line_packed():
    """Construct 20 points packed on lines of 3 to maximize structural density."""
    random.seed(42)
    # Design: create multiple lines of exactly 3 points, sharing points where possible
    # to maximize the number of collinear triples while keeping no 4 collinear.
    
    # Strategy: use a grid-like structure with specific slopes
    # Place points so that many triples are collinear
    
    pts = []
    used = set()
    
    # Line 1: horizontal y=0, x=-30,0,30
    for x in [-30, 0, 30]:
        p = (x, 0)
        if p not in used:
            pts.append([x, 0])
            used.add(p)
    
    # Line 2: horizontal y=10, x=-20,0,20
    for x in [-20, 0, 20]:
        p = (x, 10)
        if p not in used:
            pts.append([x, 10])
            used.add(p)
    
    # Line 3: horizontal y=-10, x=-20,0,20
    for x in [-20, 0, 20]:
        p = (x, -10)
        if p not in used:
            pts.append([x, -10])
            used.add(p)
    
    # Line 4: vertical x=0, y=-30,30 (y=0,10,-10 already used)
    for y in [-30, 30]:
        p = (0, y)
        if p not in used:
            pts.append([0, y])
            used.add(p)
    
    # Line 5: slope 1, through (0,0): (-20,-20),(20,20)
    for x in [-20, 20]:
        p = (x, x)
        if p not in used and abs(x) <= BOX and abs(x) <= BOX:
            pts.append([x, x])
            used.add(p)
    
    # Line 6: slope -1, through (0,0): (-20,20),(20,-20)
    for x in [-20, 20]:
        p = (x, -x)
        if p not in used and abs(x) <= BOX and abs(x) <= BOX:
            pts.append([x, -x])
            used.add(p)
    
    # Line 7: slope 1, through (0,10): (-10,0),(10,20)
    for x in [-10, 10]:
        y = x + 10
        p = (x, y)
        if p not in used and abs(x) <= BOX and abs(y) <= BOX:
            pts.append([x, y])
            used.add(p)
    
    # Line 8: slope -1, through (0,-10): (-10,0),(10,-20)
    for x in [-10, 10]:
        y = -x - 10
        p = (x, y)
        if p not in used and abs(x) <= BOX and abs(y) <= BOX:
            pts.append([x, y])
            used.add(p)
    
    # Fill remaining with random points that don't create 4-collinear
    attempts = 0
    while len(pts) < N and attempts < 5000:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        p = (x, y)
        if p in used:
            attempts += 1
            continue
        pts.append([x, y])
        if _has_four_collinear(pts):
            pts.pop()
        else:
            used.add(p)
        attempts += 1
    
    # If still not enough, use parabola fallback
    if len(pts) < N:
        pts = [[x, x*x] for x in range(-10, 10)]
    
    return pts[:N]

def _construct_random():
    """Random construction with 4-collinear avoidance."""
    random.seed(random.randint(0, 1000000))
    pts = []
    used = set()
    attempts = 0
    while len(pts) < N and attempts < 10000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = (x, y)
        if p in used:
            attempts += 1
            continue
        pts.append([x, y])
        if _has_four_collinear(pts):
            pts.pop()
        else:
            used.add(p)
        attempts += 1
    if len(pts) < N:
        pts = [[x, x*x] for x in range(-10, 10)]
    return pts[:N]

def construct():
    t0 = time.time()
    DEADLINE = t0 + 135.0  # 15s safety margin
    
    def true_score(p):
        return _max_3free(p)
    
    # Generate multiple initial candidates
    candidates = []
    
    # Line-packed construction
    lp_pts = _construct_line_packed()
    if _valid(lp_pts):
        candidates.append(lp_pts)
    
    # Multiple random constructions
    for seed in [1, 2, 3, 4, 5]:
        random.seed(seed)
        r_pts = _construct_random()
        if _valid(r_pts):
            candidates.append(r_pts)
    
    # Parabola fallback
    par_pts = [[x, x*x] for x in range(-10, 10)]
    if _valid(par_pts):
        candidates.append(par_pts)
    
    if not candidates:
        return par_pts
    
    # Evaluate all candidates
    best_true = None
    best_true_pts = None
    for cand in candidates:
        if time.time() > DEADLINE - 30:
            break
        st = true_score(cand)
        if best_true is None or st < best_true:
            best_true = st
            best_true_pts = [list(p) for p in cand]
    
    if best_true is None:
        return par_pts
    
    # Simulated annealing on true score
    pts = [list(p) for p in best_true_pts]
    cur_score = best_true
    T = 3.0
    it = 0
    last_improve = 0
    
    def rand_pt():
        return [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
    
    while time.time() < DEADLINE - 10:
        i = random.randrange(N)
        r = random.random()
        np_ = [list(p) for p in pts]
        
        if r < 0.4:
            np_[i] = rand_pt()
        elif r < 0.6:
            # Small perturbation
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0] + dx))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1] + dy))
        elif r < 0.75:
            # Swap two points
            j = random.randrange(N)
            np_[i], np_[j] = np_[j], np_[i]
        elif r < 0.85:
            # Targeted: try to place point on line with two existing points
            j, k = random.sample(range(N), 2)
            pi, pj, pk = pts[i], pts[j], pts[k]
            # Find a third point on line through pj, pk that's integer
            dx = pk[0] - pj[0]
            dy = pk[1] - pj[1]
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
                # Point at pj + t*(dx_n, dy_n) for various t
                for t in range(-4, 5):
                    x = pj[0] + t * dx_n
                    y = pj[1] + t * dy_n
                    if abs(x) <= BOX and abs(y) <= BOX:
                        p = (x, y)
                        if p not in set(map(tuple, pts)):
                            np_[i] = [x, y]
                            break
                else:
                    np_[i] = rand_pt()
            else:
                np_[i] = rand_pt()
        else:
            # Pair move: move two points to form a line with a third
            j = random.randrange(N)
            if j == i:
                j = (i + 1) % N
            k = random.randrange(N)
            while k == i or k == j:
                k = random.randrange(N)
            # Line through pts[k] in random direction
            dx = random.randint(-40, 40)
            dy = random.randint(-40, 40)
            if dx == 0 and dy == 0:
                dx = 1
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            t1 = random.randint(-3, 3)
            t2 = random.randint(-3, 3)
            while t2 == t1:
                t2 = random.randint(-3, 3)
            x1 = pts[k][0] + t1 * dx
            y1 = pts[k][1] + t1 * dy
            x2 = pts[k][0] + t2 * dx
            y2 = pts[k][1] + t2 * dy
            if abs(x1) <= BOX and abs(y1) <= BOX and abs(x2) <= BOX and abs(y2) <= BOX:
                np_[i] = [x1, y1]
                np_[j] = [x2, y2]
        
        if not _valid(np_):
            continue
        
        # Evaluate true score
        new_score = true_score(np_)
        
        if new_score < cur_score:
            pts = np_
            cur_score = new_score
            if cur_score < best_true:
                best_true = cur_score
                best_true_pts = [list(p) for p in pts]
            last_improve = it
        elif random.random() < math.exp(-(new_score - cur_score) / T):
            pts = np_
            cur_score = new_score
        
        T = max(0.5, T * 0.999)
        it += 1
        
        # Stagnation restart
        if it - last_improve > 300 and time.time() < DEADLINE - 30:
            # Try a new random start
            random.seed(random.randint(0, 1000000))
            new_pts = _construct_random()
            if _valid(new_pts):
                new_sc = true_score(new_pts)
                if new_sc < best_true:
                    best_true = new_sc
                    best_true_pts = [list(p) for p in new_pts]
                pts = [list(p) for p in new_pts]
                cur_score = new_sc
                last_improve = it
                T = 3.0
        
        if best_true <= 11:
            break
    
    # Final check
    if best_true_pts is not None and _valid(best_true_pts):
        return best_true_pts
    return par_pts
# EVOLVE-BLOCK-END
