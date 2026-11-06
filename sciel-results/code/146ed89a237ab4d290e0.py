import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def get_line_id(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(points):
        n = len(points)
        lines = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                lid = get_line_id(points[i], points[j])
                lines[lid].add(i)
                lines[lid].add(j)
        
        for lid, pts_in_line in lines.items():
            if len(pts_in_line) >= 4:
                return True
        return False

    def max_no_3_collinear(points):
        """
        Exact branch and bound using bitmasks for speed.
        Optimized for small n (20).
        """
        n = len(points)
        if n <= 2:
            return n
            
        # Precompute all lines and their point masks
        all_lines = defaultdict(int) # lid -> bitmask of points
        for i in range(n):
            for j in range(i + 1, n):
                lid = get_line_id(points[i], points[j])
                all_lines[lid] |= (1 << i)
                all_lines[lid] |= (1 << j)
                
        # Precompute line_id for each pair
        line_id_pair = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    line_id_pair[i][j] = get_line_id(points[i], points[j])
                    
        best = 0
        full_mask = (1 << n) - 1
        
        # Order points: maybe sort by "degree" of collinearity to prune earlier?
        # Let's stick to index order for simplicity but maybe try a few orderings if time permits.
        
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = S_mask.bit_count()
            
            # Pruning
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip start_idx
            backtrack(S_mask, start_idx + 1)
            
            # Option 2: Include start_idx
            i = start_idx
            valid = True
            # Check conflicts with points already in S
            temp_mask = S_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                lid = line_id_pair[i][j]
                M = all_lines[lid]
                # Check if any other point k in S is on this line
                if (M & S_mask).bit_count() > 1:
                    valid = False
                    break
                temp_mask ^= lsb
                
            if valid:
                backtrack(S_mask | (1 << i), start_idx + 1)
                
        backtrack(0, 0)
        return best

    def greedy_construct():
        """
        Greedy construction:
        1. Start with 3 random points.
        2. For each next point, evaluate all candidate points in the grid [-100, 100]^2.
           - Candidate must not create a 4-collinear set.
           - Score the candidate by the number of existing pairs (i, j) such that
             the new point p forms a triple with i and j.
             i.e., count of (i, j) in current set where i, j, p are collinear.
           - To avoid dense clustering that might be hard to place later, we might
             add a small penalty for points that are "too close" or just pick randomly
             from the top K candidates.
        """
        pts = []
        # Initial 3 points: random
        while len(pts) < 3:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        
        # Precompute existing lines for fast checking?
        # We will check 4-collinearity by generating lines from current pts + new pt.
        
        for _ in range(17):
            best_candidate = None
            best_score = -1
            top_candidates = []
            
            # Sample a subset of the grid to speed up? 
            # 200x200 = 40000 candidates. Checking 4-collinearity is O(N^2) for each.
            # 40000 * 20^2 * 17 iterations = ~27M ops. Might be slow in Python.
            # Let's restrict candidate pool to a smaller grid or random samples.
            # Actually, we can be smarter. 
            # Let's just sample 1000 random points per step and pick the best.
            
            candidates = set()
            while len(candidates) < 2000:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                candidates.add((x, y))
                
            for p in candidates:
                if p in pts:
                    continue
                new_pts = pts + [p]
                if has_4_collinear(new_pts):
                    continue
                
                # Calculate score: number of collinear triples involving p
                score = 0
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        if are_collinear(pts[i], pts[j], p):
                            score += 1
                            
                if score > best_score:
                    best_score = score
                    best_candidate = p
                    top_candidates = [p]
                elif score == best_score and best_score > 0:
                    top_candidates.append(p)
                    
            if best_candidate is None:
                # Fallback: pick a random valid point
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = (x, y)
                    if p in pts: continue
                    new_pts = pts + [p]
                    if not has_4_collinear(new_pts):
                        best_candidate = p
                        break
            
            pts.append(best_candidate)
            
        return pts

    def local_search(points):
        """
        Refine by swapping points.
        """
        current_score = max_no_3_collinear(points)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Try moving each point to a random new location
            for i in range(len(points)):
                orig = points[i]
                for _ in range(10):
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_p = (new_x, new_y)
                    if new_p in points:
                        continue
                    
                    points[i] = new_p
                    if has_4_collinear(points):
                        points[i] = orig
                        continue
                    
                    new_score = max_no_3_collinear(points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        points[i] = orig
                        
        return points, current_score

    best_config = None
    best_score = 21

    # Run multiple greedy constructions with different random seeds
    for i in range(50):
        if time.time() > deadline - 20:
            break
            
        random.seed(i * 12345 + 67890)
        pts = greedy_construct()
        
        if len(pts) != 20:
            continue
            
        # Validate
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
        if len(set(pts)) != 20:
            continue
        if has_4_collinear(pts):
            continue
            
        # Refine with local search
        refined_pts, score = local_search(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in refined_pts]
            
        if best_score <= 10:
            break

    if best_config is None:
        # Fallback: Random valid config
        while True:
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p in pts: continue
                new_pts = pts + [p]
                if not has_4_collinear(new_pts):
                    pts.append(p)
            if len(pts) == 20:
                return [list(p) for p in pts]
                
    return best_config
