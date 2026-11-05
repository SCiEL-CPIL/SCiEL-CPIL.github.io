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

def _build_triples(pts):
    """Return list of collinear triples as (i,j,k) with i<j<k."""
    triples = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
    return triples

def _max_3free_exact(pts, time_limit=5.0):
    """
    Exact branch-and-bound to find the largest subset with no three collinear.
    Score = size of that subset. We want to minimize this.
    
    Approach: We build the 3-free subset greedily by deciding for each point
    whether to include it. A point can be included only if it doesn't form
    a collinear triple with two already-included points.
    
    We use branch and bound: at each step, try to include the point (if possible),
    and also try to exclude it. The upper bound is current_size + remaining_points.
    """
    t0 = time.time()
    triples = _build_triples(pts)
    
    if not triples:
        return N
    
    # For each pair of points, check if they're in any triple together
    # Build adjacency: for each point, which other points form a triple with it
    # Actually, we need: for each point p, the set of points q such that {p,q} 
    # is part of some triple. And more precisely, for a growing set S, point p
    # can be added if there's no triple fully contained in S ∪ {p}.
    
    # Better: precompute for each triple which points are in it.
    # When we add point p to current set S, we must check that no triple 
    # has all 3 points in S ∪ {p}.
    
    # For efficiency: for each point p, precompute the set of "forbidden pairs"
    # i.e., pairs (a,b) such that {p,a,b} is a triple.
    forbidden_pairs = [set() for _ in range(N)]
    for (i, j, k) in triples:
        forbidden_pairs[i].add((j, k))
        forbidden_pairs[j].add((i, k))
        forbidden_pairs[k].add((i, j))
    
    # Also for each point p, the set of points that are in a triple with p
    triple_partners = [set() for _ in range(N)]
    for (i, j, k) in triples:
        triple_partners[i].update([j, k])
        triple_partners[j].update([i, k])
        triple_partners[k].update([i, j])
    
    best = 0
    # State: current_set (as a frozenset or bitmask), remaining points
    # Use bitmask for efficiency
    # remaining_mask: bitmask of points not yet decided
    # current_mask: bitmask of points in the 3-free subset
    
    # Order points by degree (number of triples they're in) descending
    # Points with more constraints should be decided first
    deg = [0]*N
    for t in triples:
        for v in t:
            deg[v] += 1
    order = sorted(range(N), key=lambda v: -deg[v])
    
    # Convert to bitmask operations
    # We'll use a recursive approach with memoization on (remaining_mask, current_mask)
    # But that's too large. Instead, use branch and bound with pruning.
    
    # Key insight: if a point p is in the current set, then any point q that
    # forms a triple with p and another point r in the current set is forbidden.
    
    def can_add(p, current_mask):
        """Check if point p can be added to the current 3-free set."""
        # For each (a,b) in forbidden_pairs[p], if both a and b are in current_mask,
        # then adding p would create a triple.
        for (a, b) in forbidden_pairs[p]:
            if (current_mask >> a) & 1 and (current_mask >> b) & 1:
                return False
        return True
    
    # Branch and bound
    # We process points in a fixed order to avoid revisiting states
    # Use iterative deepening or just DFS with pruning
    
    global _node_count
    _node_count = 0
    _deadline = t0 + time_limit
    
    def dfs(idx, current_mask, current_size, remaining_mask):
        nonlocal best, _node_count
        _node_count += 1
        if time.time() > _deadline:
            return
        
        # Upper bound
        rem_count = bin(remaining_mask).count('1')
        if current_size + rem_count <= best:
            return
        
        if idx == N:
            if current_size > best:
                best = current_size
            return
        
        p = order[idx]
        
        # Option 1: include p (if possible)
        if can_add(p, current_mask):
            new_mask = current_mask | (1 << p)
            dfs(idx+1, new_mask, current_size+1, remaining_mask & ~(1 << p))
        
        # Option 2: exclude p
        dfs(idx+1, current_mask, current_size, remaining_mask & ~(1 << p))
    
    # Initial upper bound: try a greedy solution
    greedy_mask = 0
    greedy_size = 0
    for p in order:
        if can_add(p, greedy_mask):
            greedy_mask |= (1 << p)
            greedy_size += 1
    best = greedy_size
    
    # Now do branch and bound
    full_mask = (1 << N) - 1
    dfs(0, 0, 0, full_mask)
    
    return best

def _max_3free(pts):
    """Fast scoring: use exact B&B with short time limit, fallback to greedy."""
    t0 = time.time()
    try:
        result = _max_3free_exact(pts, time_limit=3.0)
        return result
    except Exception:
        pass
    
    # Fallback: greedy 3-free subset
    triples = _build_triples(pts)
    if not triples:
        return N
    
    # Greedy: add points that don't create triples
    included = set()
    for i in range(N):
        ok = True
        for (a, b, c) in triples:
            if i in (a,b,c):
                others = [x for x in (a,b,c) if x != i]
                if others[0] in included and others[1] in included:
                    ok = False
                    break
        if ok:
            included.add(i)
    return len(included)

def _construct_grid_packed():
    """
    Construct 20 points on a small grid to maximize collinear triples
    while avoiding 4-collinear.
    
    Use a 5x4 grid subset: pick points from {0,1,2,3,4} x {0,1,2,3}
    This gives many horizontal, vertical, and diagonal triples.
    """
    # Start with a 5x4 grid, pick 20 points
    # Actually 5*4 = 20 exactly!
    # But we need to check no 4 collinear.
    # In a 5x4 grid, rows have 5 points -> 4 collinear! Bad.
    
    # Use a different structure: 4x5 but skip some to avoid 4-in-row
    # Or use a shifted grid.
    
    # Let's use points from a 7x7 grid, picking 20 carefully.
    # Actually, let's just use a random search with structured initialization.
    
    random.seed(123)
    best_pts = None
    best_triples = -1
    
    for attempt in range(500):
        # Generate points with some structure
        # Use a base grid with perturbations
        pts = []
        used = set()
        
        # Place points in clusters on lines
        # Line 1: y=0, x in {-20, -10, 0}
        # Line 2: y=10, x in {-15, -5, 5}
        # Line 3: y=20, x in {-10, 0, 10}
        # Line 4: y=-10, x in {-5, 5, 15}
        # Line 5: y=-20, x in {0, 10, 20}
        # Plus 5 more points
        
        lines = [
            [(x, 0) for x in [-20, -10, 0]],
            [(x, 10) for x in [-15, -5, 5]],
            [(x, 20) for x in [-10, 0, 10]],
            [(x, -10) for x in [-5, 5, 15]],
            [(x, -20) for x in [0, 10, 20]],
        ]
        
        for line in lines:
            for p in line:
                if p not in used and abs(p[0]) <= BOX and abs(p[1]) <= BOX:
                    pts.append(list(p))
                    used.add(p)
        
        # Add 5 more points randomly
        for _ in range(5):
            for _try in range(100):
                x = random.randint(-40, 40)
                y = random.randint(-40, 40)
                p = (x, y)
                if p in used:
                    continue
                pts.append([x, y])
                if _has_four_collinear(pts):
                    pts.pop()
                else:
                    used.add(p)
                    break
        
        if len(pts) != N:
            continue
        
        if not _valid(pts):
            continue
        
        tc = _triple_count(pts)
        if tc > best_triples:
            best_triples = tc
            best_pts = [list(p) for p in pts]
    
    if best_pts is None:
        # Fallback
        best_pts = [[x, x*x] for x in range(-10, 10)]
    
    return best_pts

def construct():
    t0 = time.time()
    random.seed(7)
    
    def rand_pt():
        return [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
    
    # Start with grid-packed construction
    pts = _construct_grid_packed()
    
    best_true = None
    best_true_pts = None
    if _valid(pts):
        st = _max_3free(pts)
        best_true = st
        best_true_pts = [list(p) for p in pts]
    else:
        pts = [[x, x*x] for x in range(-10,10)]
        best_true = _max_3free(pts)
        best_true_pts = [list(p) for p in pts]
    
    cur_triples = _triple_count(pts) if _valid(pts) else -1
    best_triples = cur_triples
    best_triple_pts = [list(p) for p in pts]
    
    T = 2.0
    end = t0 + 130.0  # Leave 20s safety margin
    it = 0
    last_score_check = t0
    while time.time() < end:
        i = random.randrange(N)
        r = random.random()
        np_ = [list(p) for p in pts]
        if r < 0.4:
            np_[i] = rand_pt()
        elif r < 0.65:
            j = random.randrange(N)
            np_[i], np_[j] = np_[j], np_[i]
        elif r < 0.85:
            dx = random.randint(-5,5); dy = random.randint(-5,5)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0]+dx))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1]+dy))
        else:
            # Try to move point onto a line with two existing points
            # Pick two other points, try to place i on their line
            others = [k for k in range(N) if k != i]
            a, b = random.sample(others, 2)
            pa, pb = np_[a], np_[b]
            # Line through pa, pb: find integer points on it within box
            # Parametric: pa + t*(pb-pa)
            dx = pb[0]-pa[0]
            dy = pb[1]-pa[1]
            # Find gcd to get integer steps
            g = math.gcd(abs(dx), abs(dy)) if dx != 0 or dy != 0 else 1
            if g > 0:
                sx, sy = dx//g, dy//g
                # Try a few t values
                for t in range(-3, 4):
                    nx = pa[0] + t*sx
                    ny = pa[1] + t*sy
                    if -BOX <= nx <= BOX and -BOX <= ny <= BOX:
                        np_[i] = [nx, ny]
                        break
        
        if not _valid(np_):
            continue
        
        tc = _triple_count(np_)
        if tc > cur_triples:
            pts = np_; cur_triples = tc
            if tc > best_triples:
                best_triples = tc
                best_triple_pts = [list(p) for p in np_]
        else:
            if random.random() < math.exp(-(cur_triples - tc)/T):
                pts = np_; cur_triples = tc
        
        T = max(0.3, T*0.9995)
        it += 1
        
        # Periodically check true score
        if time.time() - last_score_check > 5.0 and _valid(pts):
            last_score_check = time.time()
            st = _max_3free(pts)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in pts]
            if best_true is not None and best_true <= 11:
                break
    
    # Final check on candidates
    for cand in (best_triple_pts, best_true_pts, pts):
        if _valid(cand):
            st = _max_3free(cand)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in cand]
    
    if best_true_pts is not None and _valid(best_true_pts):
        return best_true_pts
    return [[x, x*x] for x in range(-10,10)]
# EVOLVE-BLOCK-END
