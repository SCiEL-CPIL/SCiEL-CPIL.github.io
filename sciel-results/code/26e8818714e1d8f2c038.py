# EVOLVE-BLOCK-START
import random, time, math
from itertools import combinations

BOX = 100
N = 20

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
    """Exact branch-and-bound for largest 3-collinear-free subset."""
    n = N
    # Build conflict graph: edge between i,j if they're in a collinear triple
    # Actually we need: a set is 3-free iff no 3 points in it are collinear
    # So we want max subset with no collinear triple.
    # Represent: for each point i, mask of points that form a collinear triple with i and some other point
    # Better: build list of collinear triples, then do branch and bound on inclusion
    
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
    
    # For each point, precompute which triples it's in
    point_triples = [[] for _ in range(n)]
    for (i, j, k) in triples:
        point_triples[i].append((j, k))
        point_triples[j].append((i, k))
        point_triples[k].append((i, j))
    
    best = [0]
    nodes = [0]
    CAP = 500000
    
    # For each point, the set of points that would be blocked if we add this point
    # given a current chosen set: point p blocks any pair (a,b) in chosen where (a,b,p) is a triple
    def _can_add(p, chosen_mask):
        for (a, b) in point_triples[p]:
            if (chosen_mask >> a) & 1 and (chosen_mask >> b) & 1:
                return False
        return True
    
    def search(idx, chosen_mask, size):
        nodes[0] += 1
        if nodes[0] > CAP:
            return
        if size > best[0]:
            best[0] = size
        # Upper bound
        rem_count = 0
        for m in range(idx, n):
            if not ((chosen_mask >> m) & 1):
                rem_count += 1
        if size + rem_count <= best[0]:
            return
        if idx >= n:
            return
        # Try adding point idx
        if not ((chosen_mask >> idx) & 1) and _can_add(idx, chosen_mask):
            search(idx + 1, chosen_mask | (1 << idx), size + 1)
        # Try skipping point idx
        search(idx + 1, chosen_mask, size)
    
    search(0, 0, 0)
    return best[0]

def _line_packed_construction(seed):
    """Construct 20 points by explicitly placing them on shared lines to create 3-point collinearities."""
    random.seed(seed)
    pts = []
    used = set()
    
    def in_bounds(p):
        return -BOX <= p[0] <= BOX and -BOX <= p[1] <= BOX
    
    def add_pt(p):
        if in_bounds(p) and tuple(p) not in used:
            pts.append(p)
            used.add(tuple(p))
            return True
        return False
    
    # Phase 1: Place initial points to form a few 3-point lines
    # We want lines that share points (so each point is on multiple lines)
    # Strategy: create a small "grid" of intersection points
    
    # Start with a few base points
    base = [
        [0, 0], [20, 10], [40, 20],  # line y = x/2
        [10, -10], [30, 10], [50, 30],  # line y = x - 20
        [-20, 15], [0, 15], [20, 15],  # horizontal y=15
        [10, -5], [10, 15], [10, 35],  # vertical x=10
    ]
    for p in base:
        add_pt([p[0], p[1]])
    
    # Phase 2: For each pair of existing points, try to find a third integer point
    # on the line through them that isn't already used, and add it
    # This creates new 3-point collinearities
    
    attempts = 0
    max_attempts = 200
    while len(pts) < N and attempts < max_attempts:
        attempts += 1
        # Pick a random pair
        i, j = random.sample(range(len(pts)), 2)
        pi, pj = pts[i], pts[j]
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        if g == 0:
            continue
        dx_n = dx // g
        dy_n = dy // g
        
        # Try extending in both directions
        for t in [1, 2, 3, -1, -2, -3]:
            nx = pi[0] + t * dx_n
            ny = pi[1] + t * dy_n
            if in_bounds([nx, ny]) and (nx, ny) not in used:
                add_pt([nx, ny])
                break
        else:
            # Try from pj
            for t in [1, 2, 3, -1, -2, -3]:
                nx = pj[0] + t * dx_n
                ny = pj[1] + t * dy_n
                if in_bounds([nx, ny]) and (nx, ny) not in used:
                    add_pt([nx, ny])
                    break
    
    # Phase 3: Fill remaining with random points
    while len(pts) < N:
        p = [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
        add_pt(p)
    
    return pts[:N]

def _line_packed_local_search(pts, time_budget):
    """Local search that specifically targets creating new 3-point lines."""
    t0 = time.time()
    cur_triples = _triple_count(pts)
    best_triples = cur_triples
    best_pts = [list(p) for p in pts]
    
    def rand_pt():
        return [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
    
    T = 3.0
    it = 0
    while time.time() - t0 < time_budget:
        it += 1
        np_ = [list(p) for p in pts]
        r = random.random()
        
        if r < 0.30:
            # Place a point on a line through two existing points
            i, j = random.sample(range(N), 2)
            pi, pj = np_[i], np_[j]
            dx = pj[0] - pi[0]
            dy = pj[1] - pi[1]
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g == 0:
                np_[random.randrange(N)] = rand_pt()
            else:
                dx_n = dx // g
                dy_n = dy // g
                t = random.choice([1, 2, 3, -1, -2, -3])
                base = random.choice([pi, pj])
                nx = base[0] + t * dx_n
                ny = base[1] + t * dy_n
                if -BOX <= nx <= BOX and -BOX <= ny <= BOX:
                    k = random.randrange(N)
                    np_[k] = [nx, ny]
                else:
                    np_[random.randrange(N)] = rand_pt()
        elif r < 0.50:
            # Move a point to extend an existing line
            i = random.randrange(N)
            # Find a line through point i and another point, extend it
            j = random.randrange(N)
            if i == j:
                continue
            pi, pj = np_[i], np_[j]
            dx = pj[0] - pi[0]
            dy = pj[1] - pi[1]
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g == 0:
                np_[i] = rand_pt()
            else:
                dx_n = dx // g
                dy_n = dy // g
                t = random.choice([2, 3, -2, -3])
                base = random.choice([pi, pj])
                nx = base[0] + t * dx_n
                ny = base[1] + t * dy_n
                if -BOX <= nx <= BOX and -BOX <= ny <= BOX:
                    np_[i] = [nx, ny]
                else:
                    np_[i] = rand_pt()
        elif r < 0.70:
            # Random reposition
            np_[random.randrange(N)] = rand_pt()
        elif r < 0.85:
            # Swap
            i, j = random.sample(range(N), 2)
            np_[i], np_[j] = np_[j], np_[i]
        else:
            # Small jitter
            i = random.randrange(N)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0] + dx))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1] + dy))
        
        if not _valid(np_):
            continue
        tc = _triple_count(np_)
        if tc > cur_triples:
            pts = np_
            cur_triples = tc
            if tc > best_triples:
                best_triples = tc
                best_pts = [list(p) for p in np_]
        else:
            if random.random() < math.exp(-(cur_triples - tc) / T):
                pts = np_
                cur_triples = tc
        T = max(0.5, T * 0.999)
        
        # Early stop if we've found a very low score
        if best_triples >= 30 and it % 500 == 0:
            st = _max_3free(best_pts)
            if st <= 11:
                break
    
    return best_pts, best_triples

def construct():
    t0 = time.time()
    DEADLINE = t0 + 130.0  # safety margin before 150s kill
    
    # Multiple restarts with different line-packing constructions
    best_true = None
    best_true_pts = None
    best_triples_global = -1
    best_triples_pts = None
    
    seeds = [7, 42, 123, 999, 555, 31415, 27182, 16180, 777, 12345]
    seed_idx = 0
    
    while time.time() < DEADLINE - 15:  # leave 15s for final scoring
        # Construct a line-packed configuration
        pts = _line_packed_construction(seeds[seed_idx % len(seeds)])
        seed_idx += 1
        
        if not _valid(pts):
            continue
        
        tc = _triple_count(pts)
        
        # Local search with remaining time
        time_left = DEADLINE - time.time() - 5
        if time_left < 5:
            break
        
        # Allocate time: more for earlier seeds
        alloc = min(time_left * 0.3, 8.0)
        if seed_idx > 5:
            alloc = min(alloc, 4.0)
        
        searched_pts, searched_triples = _line_packed_local_search(pts, alloc)
        
        if searched_triples > best_triples_global:
            best_triples_global = searched_triples
            best_triples_pts = [list(p) for p in searched_pts]
        
        # Score the best candidate
        if time.time() < DEADLINE - 10:
            st = _max_3free(searched_pts)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in searched_pts]
                if best_true <= 11:
                    break
    
    # Also try the original line-based construction as a fallback
    if time.time() < DEADLINE - 10:
        orig_pts = []
        lines = [
            ((-50,-50),(50,50)), ((-50,50),(50,-50)),
            ((-60,0),(60,0)), ((0,-60),(0,60)),
            ((-40,-60),(40,60)), ((-60,-40),(60,40)),
        ]
        for (x1,y1),(x2,y2) in lines:
            for t in (1,2,3):
                x = x1 + (x2-x1)*t//4
                y = y1 + (y2-y1)*t//4
                orig_pts.append([x, y])
        orig_pts.append([10, 20]); orig_pts.append([-15, 35])
        orig_pts = orig_pts[:N]
        if _valid(orig_pts):
            st = _max_3free(orig_pts)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in orig_pts]
    
    if best_true_pts is not None and _valid(best_true_pts):
        return best_true_pts
    
    # Fallback: parabola
    return [[x, x*x] for x in range(-10,10)]
# EVOLVE-BLOCK-END
