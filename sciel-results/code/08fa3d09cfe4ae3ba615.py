import time
import random
import sys
from math import gcd
from collections import defaultdict

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            if A == 0 and B == 0:
                continue
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
                if lines[key] >= 4:
                    return False
    return True

def _max_independent_set(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    if n == 0:
        return 0

    # Precompute: for each pair (i,j), which other points k are collinear with them
    collinear_with = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            collinear_with[i][j] = mask
            collinear_with[j][i] = mask

    best = 0

    def backtrack(current_mask, candidates_mask):
        nonlocal best
        cand_count = candidates_mask.bit_count()
        cur_count = current_mask.bit_count()
        if cur_count + cand_count <= best:
            return
        if cand_count == 0:
            if cur_count > best:
                best = cur_count
            return

        # Pick point with highest conflict degree among candidates for better pruning
        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        rest_mask = candidates_mask & ~(1 << p_idx)

        # Branch 1: Include p_idx
        blocked_mask = 0
        cm = current_mask
        while cm:
            r_idx = (cm & -cm).bit_length() - 1
            blocked_mask |= collinear_with[p_idx][r_idx]
            cm &= cm - 1

        new_candidates = rest_mask & ~blocked_mask
        backtrack(current_mask | (1 << p_idx), new_candidates)

        # Branch 2: Exclude p_idx
        backtrack(current_mask, rest_mask)

    backtrack(0, (1 << n) - 1)
    return best

def _generate_candidate_pool(size_range=(-50, 50)):
    """Generate a pool of candidate points to search from."""
    points = []
    for x in range(size_range[0], size_range[1] + 1):
        for y in range(size_range[0], size_range[1] + 1):
            points.append((x, y))
    return points

def _precompute_lines(pool):
    """
    Precompute all lines (sets of >=3 collinear points) within the pool.
    Returns a list of sets of point indices.
    """
    n = len(pool)
    # Map point coordinate to index
    coord_to_idx = {p: i for i, p in enumerate(pool)}
    
    lines = []
    seen_lines = set()
    
    # To keep it manageable, we only look for lines of exactly 3 points initially?
    # No, we need to know if a line has >3 points to avoid them.
    # But for packing, we want to select 3 points that form a line.
    # If 4 are collinear, we can't select any 3 from them if we want to avoid 4-collinear in final set?
    # Wait, the constraint is "no four collinear" in the final set of 20.
    # So if we pick a triple (A,B,C) that lies on a line with D, and we later pick D, we fail.
    # So we must ensure that for any line L in the grid, we pick at most 3 points from L.
    # Actually, if we pick 3 points from L, we cannot pick the 4th.
    # If we pick 2 points from L, we can pick the 3rd or 4th, but not both.
    
    # Strategy:
    # 1. Identify all maximal lines in the pool.
    # 2. Select 20 points such that:
    #    a. No line contains 4 or more selected points.
    #    b. Maximize the number of lines containing exactly 3 selected points.
    
    # This is a constraint satisfaction + optimization problem.
    # Given the pool size (100x100 = 10000 points), precomputing all lines is O(N^2) which is 100M pairs.
    # That might be too slow for 150s if done naively in Python.
    
    # Let's restrict the pool to something smaller or use a smarter generation.
    # The current program uses random points in [-50, 50].
    # Let's stick to a pool of ~500-1000 points to make precomputation feasible.
    
    pass

def _construct_via_packing(deadline):
    """
    Strategy:
    1. Define a small candidate pool (e.g., 30x30 grid).
    2. Precompute all lines in this pool.
    3. Use a greedy randomized algorithm to pick 20 points:
       - Maintain a set of 'available' points.
       - At each step, pick a point that allows for the most future 3-collinear triples.
       - Or, pick a triple directly?
       
    Alternative Structural Approach:
    Use a known dense configuration and perturb it?
    Or, use the "Packing" idea more directly:
    Generate all valid triples (lines of 3) in a small grid.
    Run a Max-SAT or Greedy Set Packing to find 20 points maximizing disjoint triples.
    """
    
    # Given time constraints, let's use a smaller grid for the pool, say 20x20 = 400 points.
    # Precomputing lines for 400 points: 400*399/2 = ~80k pairs. Very fast.
    
    pool_size = 25 # 25x25 grid
    offset = -12
    pool = []
    for x in range(offset, offset + pool_size):
        for y in range(offset, offset + pool_size):
            pool.append((x, y))
            
    n_pool = len(pool)
    coord_to_idx = {p: i for i, p in enumerate(pool)}
    
    # Find all lines in the pool
    # A line is defined by a pair of points. We normalize the line equation.
    # We want to group points by line.
    lines_dict = defaultdict(list)
    
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            x1, y1 = pool[i]
            x2, y2 = pool[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            if A == 0 and B == 0:
                continue
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            lines_dict[key].append(i)
            lines_dict[key].append(j) # Wait, this adds duplicates.
            
    # Re-do lines_dict properly
    lines_dict = defaultdict(set)
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            x1, y1 = pool[i]
            x2, y2 = pool[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            if A == 0 and B == 0:
                continue
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            lines_dict[key].add(i)
            lines_dict[key].add(j)
            
    # Filter lines with at least 3 points
    valid_lines = []
    for key, pts in lines_dict.items():
        if len(pts) >= 3:
            valid_lines.append((key, list(pts)))
            
    # Sort lines by length? Longer lines are more "dangerous" (harder to pack 3 from them without risking 4).
    # But we WANT to pack 3 from them.
    # Constraint: In the final set S of 20 points, for every line L, |S intersect L| <= 3.
    # Objective: Maximize number of lines L such that |S intersect L| == 3.
    
    # This is a hard combinatorial problem.
    # Let's use a Greedy Randomized Adaptive Search Procedure (GRASP) or similar.
    
    # State:
    # selected: set of indices
    # line_counts: dict mapping line_key -> count of selected points in that line
    
    best_score = -1
    best_points = None
    
    # Time budget for search
    search_deadline = deadline - 10.0 # Leave 10s for evaluation and safety
    
    # Try multiple random restarts
    for restart in range(100):
        if time.time() > search_deadline:
            break
            
        random.seed(42 + restart)
        selected = set()
        line_counts = defaultdict(int)
        
        # To make it faster, we might not pick points one by one, but try to pick triples?
        # Let's stick to point-by-point but with a heuristic.
        
        # Heuristic: Pick a point that is part of many "available" lines (lines that can still take 1 more point to become a triple, or are already triples).
        # Actually, we want to maximize the count of lines becoming triples.
        
        # Simple Greedy:
        # 1. Start with empty set.
        # 2. For each step, pick a point p from pool not in selected.
        # 3. Score(p) = sum over lines L containing p of (benefit of adding p to L).
        #    Benefit is high if L currently has 2 points (adding p makes it a triple).
        #    Benefit is negative if L currently has 3 points (adding p makes it 4 -> INVALID).
        
        # We need to ensure we never violate the "no 4 collinear" constraint.
        # So if a line has 3 selected points, we CANNOT select any other point on that line.
        
        # Let's implement this greedy.
        
        available_points = list(range(n_pool))
        random.shuffle(available_points)
        
        # We will pick 20 points.
        for step in range(20):
            if time.time() > search_deadline:
                break
                
            best_p = -1
            best_score_p = -1
            best_lines_info = None
            
            # To speed up, we might sample candidates instead of checking all
            # But n_pool is 625. 625 checks per step * 20 steps = 12500 checks. Very fast.
            
            for p in available_points:
                if p in selected:
                    continue
                    
                # Check if adding p is valid
                valid = True
                score_p = 0
                # We need to know which lines p is on.
                # Precompute point_to_lines
                # This requires another pass.
                
                # Let's precompute point_to_lines outside the restart loop?
                # Yes, do it once.
                
            # Precomputation for point_to_lines
            # This is done inside the restart loop currently, which is inefficient.
            # I will restructure the code to precompute once.
            
        # If we broke out of inner loop, check result
        if len(selected) == 20:
             # Check validity
             if _check_no_four_collinear([pool[i] for i in selected]):
                 # Count triples
                 triples = 0
                 for key, pts in valid_lines:
                     cnt = sum(1 for p in pts if p in selected)
                     if cnt == 3:
                         triples += 1
                 if triples > best_score:
                     best_score = triples
                     best_points = [pool[i] for i in selected]
                     
    return best_points, best_score

# Precomputation outside the main loop to save time
POOL_SIZE = 25
OFFSET = -12
POOL = []
for x in range(OFFSET, OFFSET + POOL_SIZE):
    for y in range(OFFSET, OFFSET + POOL_SIZE):
        POOL.append((x, y))

N_POOL = len(POOL)
COORD_TO_IDX = {p: i for i, p in enumerate(POOL)}

# Precompute lines
LINES_DICT = defaultdict(set)
for i in range(N_POOL):
    for j in range(i + 1, N_POOL):
        x1, y1 = POOL[i]
        x2, y2 = POOL[j]
        dx = x2 - x1
        dy = y2 - y1
        A = dy
        B = -dx
        C = dx * y1 - dy * x1
        if A == 0 and B == 0:
            continue
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
        key = (A, B, C)
        LINES_DICT[key].add(i)
        LINES_DICT[key].add(j)

VALID_LINES = []
for key, pts in LINES_DICT.items():
    if len(pts) >= 3:
        VALID_LINES.append((key, list(pts)))

# Precompute point_to_lines
POINT_TO_LINES = defaultdict(list)
for key, pts in VALID_LINES:
    for p in pts:
        POINT_TO_LINES[p].append(key)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_points = None
    best_score = float('inf') # We want to MINIMIZE independence number, but we optimize for MAXIMIZING triples as a proxy.
    # Note: Maximizing triples is a proxy for minimizing independence number.
    # We will evaluate the actual score at the end.
    
    best_triples = -1
    
    # Greedy Randomized Adaptive Search Procedure (GRASP)
    # We run multiple iterations.
    
    for iteration in range(500):
        if time.time() > deadline:
            break
            
        random.seed(iteration * 123 + 456)
        
        selected = set()
        line_counts = defaultdict(int)
        
        # We want to pick 20 points.
        # Heuristic: 
        # At each step, pick a point p that maximizes the number of "potential triples" it completes.
        # A potential triple is a line L where count(L) == 2. Adding p makes it 3.
        # We must avoid lines where count(L) == 3 (would become 4).
        
        # To add randomness, we use a "restricted candidate list" (RCL) based on score.
        
        for step in range(20):
            if time.time() > deadline:
                break
                
            # Calculate scores for all available points
            # Available: not in selected
            # Score(p):
            #   +10 for each line L containing p where count(L) == 2 (completing a triple)
            #   -100 for each line L containing p where count(L) == 3 (violation)
            #   +1 for each line L containing p where count(L) == 1 (building towards triple)
            #   +0.1 for each line L containing p where count(L) == 0 (starting a line)
            
            scores = {}
            for p in range(N_POOL):
                if p in selected:
                    continue
                
                score = 0
                valid = True
                for key in POINT_TO_LINES[p]:
                    cnt = line_counts[key]
                    if cnt >= 3:
                        valid = False
                        break
                    elif cnt == 2:
                        score += 10
                    elif cnt == 1:
                        score += 1
                    else:
                        score += 0.1
                
                if valid:
                    scores[p] = score
            
            if not scores:
                break
                
            # Find max score
            max_score = max(scores.values())
            
            # RCL: points with score >= alpha * max_score
            alpha = 0.8
            threshold = alpha * max_score
            rcl = [p for p, s in scores.items() if s >= threshold]
            
            # Pick randomly from RCL
            p = random.choice(rcl)
            selected.add(p)
            
            # Update line counts
            for key in POINT_TO_LINES[p]:
                line_counts[key] += 1
                
        if len(selected) == 20:
            # Check validity
            pts_list = [POOL[i] for i in selected]
            if _check_no_four_collinear(pts_list):
                # Count triples
                triples = 0
                for key, pts in VALID_LINES:
                    cnt = sum(1 for p in pts if p in selected)
                    if cnt == 3:
                        triples += 1
                
                if triples > best_triples:
                    best_triples = triples
                    best_points = pts_list
                    # Optional: Evaluate actual score here?
                    # It's expensive. Let's do it only for the best few candidates.
                    # Or just rely on triples as proxy and evaluate at the end.
                    # Given 150s limit, and _max_independent_set is fast for n=20,
                    # we can evaluate every time we find a new best triple count.
                    
                    score = _max_independent_set(pts_list)
                    if score < best_score:
                        best_score = score
                        print(f"Iter {iteration}: Triples {triples}, Score {score}", file=sys.stderr)
    
    # Fallback if no valid points found (should not happen)
    if best_points is None:
        # Simple random valid config
        random.seed(0)
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts and _check_no_four_collinear(pts + [p]):
                pts.append(p)
        best_points = pts
        best_score = _max_independent_set(best_points)
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
