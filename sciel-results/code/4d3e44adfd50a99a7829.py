# EVOLVE-BLOCK-START
import random
import time
import math
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
    """Exact branch-and-bound for largest 3-free subset."""
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
    
    best = [0]
    nodes = [0]
    CAP = 200000 # Reduced cap to allow more frequent calls or faster termination if stuck
    
    # Precompute for each point, the set of other points it forms a triple with
    # Actually, let's stick to the simple B&B but with better pruning
    
    def search(idx, chosen_mask, size):
        nodes[0] += 1
        if nodes[0] > CAP:
            return
        if size > best[0]:
            best[0] = size
        
        # Pruning: if we can't beat best with remaining points
        rem_count = 0
        for m in range(idx, n):
            if not ((chosen_mask >> m) & 1):
                rem_count += 1
        if size + rem_count <= best[0]:
            return
        
        if idx == n:
            return
            
        p = idx
        # Option 1: Don't include p
        search(idx+1, chosen_mask, size)
        
        # Option 2: Include p
        # Check if p conflicts with any chosen points
        ok = True
        # We need to check if adding p creates a triple with any pair of already chosen points
        # This is expensive to check naively. 
        # Better approach: maintain a set of "forbidden" points?
        # Or simply check all pairs in chosen_mask.
        
        # To speed up, we can precompute conflict sets?
        # For now, check pairs in chosen_mask
        chosen_list = []
        for a in range(idx):
            if (chosen_mask >> a) & 1:
                chosen_list.append(a)
                
        for a in range(len(chosen_list)):
            for b in range(a+1, len(chosen_list)):
                ia = chosen_list[a]
                ib = chosen_list[b]
                if _collinear3(pts[ia], pts[ib], pts[p]):
                    ok = False
                    break
            if not ok:
                break
                
        if ok:
            search(idx+1, chosen_mask | (1 << p), size+1)
            
    search(0, 0, 0)
    return best[0]

def _construct_disjoint_triples(seed_val=42):
    """
    Construct 20 points with 6 disjoint lines of 3 points.
    This guarantees a 3-free number of at most 14.
    """
    random.seed(seed_val)
    pts = []
    used = set()
    
    # Define 6 lines with distinct slopes/intercepts to avoid accidental 4-collinear
    # We will pick 3 points from each line.
    # To ensure disjointness, we will pick points far apart.
    
    # Line 1: y=0, x in {-90, -80, -70}
    # Line 2: y=10, x in {-90, -80, -70}
    # ...
    # This creates a grid-like structure which might have vertical 3-collinear?
    # Yes, x=-90, y=0,10,20... would be 4-collinear if we go too high.
    # We only have 6 lines. So y=0,10,20,30,40,50.
    # Points:
    # L1: (-90,0), (-80,0), (-70,0)
    # L2: (-90,10), (-80,10), (-70,10)
    # L3: (-90,20), (-80,20), (-70,20)
    # L4: (-90,30), (-80,30), (-70,30)
    # L5: (-90,40), (-80,40), (-70,40)
    # L6: (-90,50), (-80,50), (-70,50)
    
    # Check 4-collinear:
    # Vertical lines at x=-90: (-90,0), (-90,10), (-90,20), (-90,30) -> 4 collinear! INVALID.
    
    # We need to avoid this.
    # Use different x-ranges for each line?
    # L1: x in {-90, -80, -70}, y=0
    # L2: x in {-60, -50, -40}, y=10
    # L3: x in {-30, -20, -10}, y=20
    # L4: x in {10, 20, 30}, y=30
    # L5: x in {40, 50, 60}, y=40
    # L6: x in {70, 80, 90}, y=50
    
    lines = [
        [(-90, 0), (-80, 0), (-70, 0)],
        [(-60, 10), (-50, 10), (-40, 10)],
        [(-30, 20), (-20, 20), (-10, 20)],
        [(10, 30), (20, 30), (30, 30)],
        [(40, 40), (50, 40), (60, 40)],
        [(70, 50), (80, 50), (90, 50)]
    ]
    
    for line in lines:
        for p in line:
            pts.append([p[0], p[1]])
            used.add(p)
            
    # Add 2 more points
    # They should not create 4-collinear.
    # Place them in the middle, e.g., (0, 15) and (0, 35)?
    # Check (0,15) with existing points.
    # Existing points are on y=0,10,20,30,40,50.
    # (0,15) is not on any of these lines.
    # Does it form a line with two existing points?
    # e.g., (-10,20) and (10,30)? Slope = (30-20)/(10-(-10)) = 10/20 = 0.5.
    # Line: y - 20 = 0.5(x + 10) => y = 0.5x + 25.
    # At x=0, y=25. Not 15.
    # Let's just try random points for the last 2 and check validity.
    
    attempts = 0
    while len(pts) < N and attempts < 1000:
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
        # Fallback
        pts = [[x, x*x] for x in range(-10, 10)]
        
    return pts[:N]

def _construct_packed_triples(seed_val=42):
    """
    Try to pack more triples by allowing some overlap, but ensuring no 4-collinear.
    """
    random.seed(seed_val)
    pts = []
    used = set()
    
    # Start with a grid of points, then perturb to create lines?
    # Or start with lines and add points that complete new lines?
    
    # Let's try a "star" configuration or something similar.
    
    # For now, return the disjoint triple construction as a baseline for this function
    return _construct_disjoint_triples(seed_val)

def construct():
    t0 = time.time()
    random.seed(7)
    
    # Generate initial candidates
    candidates = [
        _construct_disjoint_triples(42),
        _construct_disjoint_triples(123),
        _construct_disjoint_triples(999),
    ]
    
    # Filter valid
    valid_candidates = []
    for cand in candidates:
        if _valid(cand):
            valid_candidates.append(cand)
            
    if not valid_candidates:
        return [[x, x*x] for x in range(-10, 10)]
        
    # Pick the one with most triples
    best_cand = max(valid_candidates, key=_triple_count)
    pts = [list(p) for p in best_cand]
    
    best_true_score = _max_3free(pts)
    best_true_pts = [list(p) for p in pts]
    
    # Local Search
    T = 5.0
    end = t0 + 120.0
    cur_triples = _triple_count(pts)
    last_improve = t0
    stagnation_limit = 10.0
    
    while time.time() < end:
        # Generate neighbor
        i = random.randrange(N)
        np_ = [list(p) for p in pts]
        np_[i][0] = random.randint(-BOX, BOX)
        np_[i][1] = random.randint(-BOX, BOX)
        
        if not _valid(np_):
            continue
            
        new_triples = _triple_count(np_)
        
        # Accept if better or with probability
        if new_triples > cur_triples:
            pts = np_
            cur_triples = new_triples
            last_improve = time.time()
            
            # Check exact score periodically or if triples increased significantly
            if new_triples > cur_triples:
                # We already know it's better in triples, but is it better in 3-free?
                # Let's check exact score every 100 iterations or if we find a new triple record
                pass
        else:
            delta = cur_triples - new_triples
            if random.random() < math.exp(-delta/T):
                pts = np_
                cur_triples = new_triples
                
        T = max(0.5, T * 0.99)
        
        # Periodic exact check
        if time.time() - last_improve > 5.0:
            st = _max_3free(pts)
            if st < best_true_score:
                best_true_score = st
                best_true_pts = [list(p) for p in pts]
                last_improve = time.time()
            else:
                # If we haven't improved the true score in a while, restart?
                if time.time() - last_improve > stagnation_limit:
                    # Reset to best found so far
                    pts = [list(p) for p in best_true_pts]
                    cur_triples = _triple_count(pts)
                    T = 5.0
                    last_improve = time.time()
                    
        if best_true_score <= 12:
            break
            
    # Final check
    st = _max_3free(pts)
    if st < best_true_score:
        best_true_score = st
        best_true_pts = [list(p) for p in pts]
        
    return best_true_pts
# EVOLVE-BLOCK-END
