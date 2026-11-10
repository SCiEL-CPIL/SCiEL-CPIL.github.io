import random
import time
import sys
from itertools import combinations

_START_TIME = None
DEADLINE = 150.0
MARGIN = 15.0

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _deadline_left():
    return DEADLINE - MARGIN - _elapsed()

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def count_collinear_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def build_collinear_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def max_independent_set_3_uniform(points):
    n = len(points)
    if n <= 2:
        return n

    collinear_triples = build_collinear_triples(points)
    
    if not collinear_triples:
        return n

    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j:
                continue
            pair_to_third[i][j] = set()
            
    for i, j, k in collinear_triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)

    best_size = 0
    full_mask = (1 << n) - 1

    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    def solve_recur(current_size, current_mask, remaining_mask):
        nonlocal best_size
        
        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
            return
        
        if current_size + popcount(remaining_mask) <= best_size:
            return

        p = -1
        for i in range(n):
            if remaining_mask & (1 << i):
                p = i
                break
                
        if p == -1:
            return

        solve_recur(current_size, current_mask, remaining_mask & ~(1 << p))

        compatible = True
        cm = current_mask
        while cm:
            lsb = cm & -cm
            r = (lsb.bit_length() - 1)
            if pair_to_third[p][r] & current_mask:
                compatible = False
                break
            cm ^= lsb
            
        if compatible:
            new_current_mask = current_mask | (1 << p)
            new_remaining = remaining_mask & ~(1 << p)
            
            forbidden_mask = 0
            ncm = new_current_mask
            while ncm:
                lsb = ncm & -ncm
                r = (lsb.bit_length() - 1)
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
                ncm ^= lsb
                
            new_remaining &= ~forbidden_mask
            solve_recur(current_size + 1, new_current_mask, new_remaining)

    solve_recur(0, 0, full_mask)
    return best_size

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_score = 100
    best_points = None
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    def generate_structured_initial():
        """Generate initial points using structured patterns that promote 3-collinearity."""
        strategies = []
        
        # Strategy 1: Modular grid with prime spacing
        for mod in [7, 11, 13]:
            ox = random.randint(-50, 50)
            oy = random.randint(-50, 50)
            pts = []
            for i in range(mod):
                for j in range(mod):
                    x = ox + i * 3
                    y = oy + j * 3
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        pts.append((x, y))
            if len(pts) >= 20:
                random.shuffle(pts)
                strategies.append(pts[:20])
        
        # Strategy 2: Two interleaved grids
        for _ in range(3):
            ox1 = random.randint(-60, 60)
            oy1 = random.randint(-60, 60)
            ox2 = ox1 + random.randint(1, 3)
            oy2 = oy1 + random.randint(1, 3)
            pts = []
            for i in range(5):
                for j in range(4):
                    x1 = ox1 + i * 4
                    y1 = oy1 + j * 4
                    x2 = ox2 + i * 4
                    y2 = oy2 + j * 4
                    if -100 <= x1 <= 100 and -100 <= y1 <= 100:
                        pts.append((x1, y1))
                    if -100 <= x2 <= 100 and -100 <= y2 <= 100:
                        pts.append((x2, y2))
            seen = set()
            unique = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            while len(unique) < 20:
                p = random_point()
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            random.shuffle(unique)
            strategies.append(unique[:20])
        
        # Strategy 3: Diagonal lines with controlled spacing
        for _ in range(3):
            base_x = random.randint(-40, 40)
            base_y = random.randint(-40, 40)
            slope_choices = [(1, 1), (1, -1), (2, 1), (1, 2)]
            pts = []
            for dx, dy in slope_choices:
                for t in range(5):
                    x = base_x + t * dx * 3
                    y = base_y + t * dy * 3
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        pts.append((x, y))
            seen = set()
            unique = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            while len(unique) < 20:
                p = random_point()
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            random.shuffle(unique)
            strategies.append(unique[:20])
        
        return strategies

    def repair_no_four_collinear(points):
        """Repair points to ensure no four are collinear."""
        points = points.copy()
        attempts = 0
        while attempts < 200:
            if not has_four_collinear(points):
                return points
            # Find a violating set and move one point
            n = len(points)
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = points[i], points[j]
                    count = 2
                    others = []
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if are_collinear(p1, p2, points[k]):
                            count += 1
                            others.append(k)
                    if count >= 4:
                        # Move a random other point
                        idx = random.choice(others)
                        new_p = random_point()
                        if new_p not in points:
                            points[idx] = new_p
                            found = True
                            break
                if found:
                    break
            if not found:
                idx = random.randint(0, 19)
                new_p = random_point()
                if new_p not in points:
                    points[idx] = new_p
            attempts += 1
        return points

    def local_search(points, max_steps=200):
        """Local search optimizing exact score, with triple count as tiebreaker."""
        current_points = points.copy()
        current_score = max_independent_set_3_uniform(current_points)
        current_tc = count_collinear_triples(current_points)
        
        best_local_score = current_score
        best_local_points = current_points.copy()
        
        for step in range(max_steps):
            if _deadline_left() < 10.0:
                break
                
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            
            # Generate candidate moves: nearby points that might increase collinearity
            candidates = []
            for _ in range(10):
                if random.random() < 0.5:
                    dx = random.randint(-8, 8)
                    dy = random.randint(-8, 8)
                    new_p = (old_p[0] + dx, old_p[1] + dy)
                else:
                    new_p = random_point()
                if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    candidates.append(new_p)
            
            best_candidate = None
            best_candidate_score = current_score
            best_candidate_tc = current_tc
            
            for new_p in candidates:
                if new_p in current_points:
                    continue
                
                candidate_points = current_points.copy()
                candidate_points[idx] = new_p
                
                if has_four_collinear(candidate_points):
                    continue
                
                cand_tc = count_collinear_triples(candidate_points)
                
                # Only evaluate exact score if triple count is promising
                if cand_tc > current_tc + 2 or cand_tc == current_tc:
                    cand_score = max_independent_set_3_uniform(candidate_points)
                    if cand_score < best_candidate_score or (cand_score == best_candidate_score and cand_tc > best_candidate_tc):
                        best_candidate = candidate_points
                        best_candidate_score = cand_score
                        best_candidate_tc = cand_tc
            
            if best_candidate is not None:
                current_points = best_candidate
                current_score = best_candidate_score
                current_tc = best_candidate_tc
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_points = current_points.copy()
            
            # Accept equal-score moves to explore
            elif random.random() < 0.1:
                idx2 = random.randint(0, 19)
                new_p = random_point()
                if new_p not in current_points and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    candidate_points = current_points.copy()
                    candidate_points[idx2] = new_p
                    if not has_four_collinear(candidate_points):
                        current_points = candidate_points
                        current_tc = count_collinear_triples(current_points)
        
        return best_local_points, best_local_score

    # Phase 1: Generate diverse structured initializations
    initial_strategies = generate_structured_initial()
    
    # Evaluate and keep the best few
    evaluated = []
    for init_pts in initial_strategies:
        if _deadline_left() < 20.0:
            break
        repaired = repair_no_four_collinear(init_pts)
        if repaired is None:
            continue
        score = max_independent_set_3_uniform(repaired)
        tc = count_collinear_triples(repaired)
        evaluated.append((score, tc, repaired))
    
    evaluated.sort(key=lambda x: (x[0], -x[1]))
    
    # Phase 2: Local search on top candidates
    for score, tc, pts in evaluated[:5]:
        if _deadline_left() < 20.0:
            break
        improved_pts, improved_score = local_search(pts, max_steps=150)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts.copy()
    
    # Phase 3: Additional random restarts with local search
    for restart in range(30):
        if _deadline_left() < 20.0:
            break
        
        # Random initialization
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        
        points = repair_no_four_collinear(points)
        if points is None:
            continue
        
        # Quick local search
        improved_pts, improved_score = local_search(points, max_steps=50)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts.copy()
    
    # Fallback
    if best_points is None:
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        points = repair_no_four_collinear(points)
        if points is None:
            points = [(i, i * 2) for i in range(20)]
        best_points = points
        best_score = max_independent_set_3_uniform(best_points)
    
    return [list(p) for p in best_points]
