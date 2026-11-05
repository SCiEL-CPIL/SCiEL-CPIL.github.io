"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START

def _doubled_area(p, q, r):
    """Absolute doubled area (determinant) for three points."""
    ax, ay = p
    bx, by = q
    cx, cy = r
    return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

def _full_count(points):
    """Compute Counter of doubled areas and return (counter, max_count, max_area)."""
    n = len(points)
    cnt = Counter()
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                d = _doubled_area(pi, pj, pk)
                if d > 0:
                    cnt[d] += 1
    if not cnt:
        return cnt, 0, 0
    best_area, best_cnt = max(cnt.items(), key=lambda kv: kv[1])
    return cnt, best_cnt, best_area

def _grid_start():
    """6x5 grid centered at origin."""
    return [(x - 2, y - 2) for x in range(6) for y in range(5)]

def _try_grid_shapes(n=30):
    """Try all grid factorizations and return best points and score."""
    best_pts = None
    best_score = -1
    factors = [(1,30),(2,15),(3,10),(5,6),(6,5),(10,3),(15,2),(30,1)]
    for r, c in factors:
        offx = -c // 2
        offy = -r // 2
        pts = [(offx + x, offy + y) for x in range(c) for y in range(r)]
        if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in pts):
            continue
        _, s, _ = _full_count(pts)
        if s > best_score:
            best_score = s
            best_pts = list(pts)
    return best_pts, best_score

def _random_points(n=30, bounds=30, rng=None):
    """Generate n random distinct integer points in [-bounds, bounds]^2."""
    if rng is None:
        rng = random
    pts = set()
    while len(pts) < n:
        x = rng.randint(-bounds, bounds)
        y = rng.randint(-bounds, bounds)
        pts.add((x, y))
    return list(pts)

def _local_search_targeted(points, deadline, rng=None):
    """
    Local search that targets a specific area value for optimization.
    Uses O(N^2) delta evaluation per candidate move.
    Returns (best_points, best_score).
    """
    if rng is None:
        rng = random
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    
    # Initial full count to determine target
    cnt, current_count, target = _full_count(pts)
    best_overall = current_count
    best_pts = list(pts)
    
    # Precompute pairs excluding each index
    all_pairs = []
    for idx in range(n):
        pairs = []
        for j in range(n):
            if j == idx:
                continue
            for k in range(j + 1, n):
                if k == idx:
                    continue
                pairs.append((j, k))
        all_pairs.append(pairs)
    
    no_improve = 0
    max_no_improve = 50
    iterations = 0
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Re-evaluate to find current best area
            cnt, ba, bc = _full_count(pts)
            if bc > best_overall:
                best_overall = bc
                best_pts = list(pts)
            # Switch target to the current most common area
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 50
            continue
        
        # Pick a random point to move
        idx = rng.randint(0, n - 1)
        old_pt = pts[idx]
        pairs = all_pairs[idx]
        
        improved = False
        # Try up to 8 random positions
        for _ in range(8):
            new_pt = (rng.randint(-30, 30), rng.randint(-30, 30))
            if new_pt == old_pt:
                continue
            if new_pt in pts_set:
                continue
            
            # Calculate delta in count of 'target' area
            old_matches = 0
            new_matches = 0
            for j, k in pairs:
                p_j = pts[j]
                p_k = pts[k]
                if _doubled_area(old_pt, p_j, p_k) == target:
                    old_matches += 1
                if _doubled_area(new_pt, p_j, p_k) == target:
                    new_matches += 1
            
            delta = new_matches - old_matches
            
            if delta > 0:
                # Accept
                current_count += delta
                pts[idx] = new_pt
                pts_set.remove(old_pt)
                pts_set.add(new_pt)
                improved = True
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_pts = list(pts)
                break
            elif delta == 0 and rng.random() < 0.1:
                # Neutral move with some probability to escape local optima
                pts[idx] = new_pt
                pts_set.remove(old_pt)
                pts_set.add(new_pt)
                no_improve += 1
                break
        
        if not improved:
            no_improve += 1
        
        iterations += 1
    
    # Final check
    if current_count > best_overall:
        best_overall = current_count
        best_pts = list(pts)
    
    return best_pts, best_overall

def _local_search_full(points, deadline, rng=None):
    """
    Local search that tracks the full distribution of areas.
    Returns (best_points, best_score).
    """
    if rng is None:
        rng = random
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    
    # Initial full count
    cnt, current_max, _ = _full_count(pts)
    best_overall = current_max
    best_pts = list(pts)
    
    # Precompute pairs excluding each index
    all_pairs = []
    for idx in range(n):
        pairs = []
        for j in range(n):
            if j == idx:
                continue
            for k in range(j + 1, n):
                if k == idx:
                    continue
                pairs.append((j, k))
        all_pairs.append(pairs)
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Re-evaluate full count
            cnt, ba, bc = _full_count(pts)
            if bc > best_overall:
                best_overall = bc
                best_pts = list(pts)
            no_improve = 0
            max_no_improve += 100
            continue
        
        idx = rng.randint(0, n - 1)
        old_pt = pts[idx]
        pairs = all_pairs[idx]
        
        improved = False
        for _ in range(5):
            new_pt = (rng.randint(-30, 30), rng.randint(-30, 30))
            if new_pt == old_pt or new_pt in pts_set:
                continue
            
            # Compute delta for the full counter
            # Remove old contributions
            delta_counts = {}
            for j, k in pairs:
                p_j = pts[j]
                p_k = pts[k]
                old_area = _doubled_area(old_pt, p_j, p_k)
                new_area = _doubled_area(new_pt, p_j, p_k)
                if old_area > 0:
                    delta_counts[old_area] = delta_counts.get(old_area, 0) - 1
                if new_area > 0:
                    delta_counts[new_area] = delta_counts.get(new_area, 0) + 1
            
            # Apply delta to counter
            for area, delta in delta_counts.items():
                cnt[area] += delta
                if cnt[area] <= 0:
                    del cnt[area]
            
            new_max = max(cnt.values()) if cnt else 0
            
            if new_max > current_max:
                current_max = new_max
                pts[idx] = new_pt
                pts_set.remove(old_pt)
                pts_set.add(new_pt)
                improved = True
                no_improve = 0
                if current_max > best_overall:
                    best_overall = current_max
                    best_pts = list(pts)
                break
            else:
                # Revert
                for area, delta in delta_counts.items():
                    cnt[area] -= delta
                    if cnt[area] <= 0:
                        del cnt[area]
                current_max = max(cnt.values()) if cnt else 0
        
        if not improved:
            no_improve += 1
    
    if current_max > best_overall:
        best_overall = current_max
        best_pts = list(pts)
    
    return best_pts, best_overall

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    rng = random.Random(42)
    
    # Start with the best grid configuration
    pts, score = _try_grid_shapes(30)
    if pts is None:
        pts = _grid_start()
        _, score, _ = _full_count(pts)
    
    best_pts = list(pts)
    best_score = score
    
    # Phase 1: Targeted local search (faster, O(N^2) per move)
    if time.time() < deadline - 5:
        search_deadline = min(deadline - 5, start_time + 20)
        search_pts, search_score = _local_search_targeted(best_pts, search_deadline, rng)
        if search_score > best_score:
            best_score = search_score
            best_pts = search_pts
    
    # Phase 2: Full local search (more accurate, but slower)
    if time.time() < deadline - 3:
        search_deadline = min(deadline - 3, time.time() + 15)
        search_pts, search_score = _local_search_full(best_pts, search_deadline, rng)
        if search_score > best_score:
            best_score = search_score
            best_pts = search_pts
    
    # Phase 3: Random restarts with targeted search
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        num_restarts = max(1, int(remaining_time / 3))
        for attempt in range(num_restarts):
            if time.time() > deadline - 1:
                break
            
            rand_pts = _random_points(30, 30, rng)
            search_deadline = min(deadline - 1, time.time() + 2.0)
            search_pts, search_score = _local_search_targeted(rand_pts, search_deadline, rng)
            
            if search_score > best_score:
                best_score = search_score
                best_pts = search_pts
                # If improved, do a bit more search
                if time.time() < deadline - 2:
                    search_deadline2 = min(deadline - 2, time.time() + 1.5)
                    search_pts2, search_score2 = _local_search_full(best_pts, search_deadline2, rng)
                    if search_score2 > best_score:
                        best_score = search_score2
                        best_pts = search_pts2
    
    # Validation
    if len(best_pts) != 30 or len(set(best_pts)) != 30:
        return _grid_start()
    for p in best_pts:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return _grid_start()
    
    return best_pts

# EVOLVE-BLOCK-END
