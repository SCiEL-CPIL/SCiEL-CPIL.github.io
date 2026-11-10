import time
import sys
from math import gcd
from collections import defaultdict
import random

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
        # We approximate conflict degree by counting how many candidates are collinear with this point
        # with any other candidate? No, simpler: just pick the first one or one with high degree.
        # For n=20, we can afford a slightly better heuristic.
        
        # Let's pick the point that blocks the most other points.
        best_p_idx = -1
        max_blocks = -1
        
        # To speed up, we can just iterate through a few candidates or the first one.
        # Given n=20, iterating all is 20 ops. 
        cm = candidates_mask
        while cm:
            p_idx = (cm & -cm).bit_length() - 1
            # Count how many candidates would be blocked if we pick p_idx
            # A candidate q is blocked if it is collinear with p_idx and some other point in current_mask?
            # No, the rule is: if we pick p, we cannot pick any q such that q, p, and r are collinear for some r in current_mask?
            # Actually, the constraint is "no three collinear".
            # If we pick p, we cannot pick any pair (q, r) from candidates such that p, q, r are collinear.
            # This is complex.
            
            # Simpler heuristic: Pick the point that is part of the most collinear triples within the candidates?
            # Or just pick the first one. Branch and bound with good upper bound is usually enough for n=20.
            # The current seed program just picks the first one. Let's stick to that for reliability,
            # but maybe add a simple degree check.
            
            # Let's just use the first one to keep it fast and deterministic for the B&B itself.
            # The pruning `cur_count + cand_count <= best` is the main driver.
            break 
            
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Define a small candidate pool
    # A 15x15 grid is small enough to precompute all triples and solve CSP quickly.
    # 15x15 = 225 points.
    # Number of pairs ~ 25k. Number of lines is manageable.
    pool_size = 15
    offset = -7
    pool = []
    for x in range(offset, offset + pool_size):
        for y in range(offset, offset + pool_size):
            pool.append((x, y))
            
    n_pool = len(pool)
    
    # 2. Precompute all lines and triples
    # Map point coordinate to index
    coord_to_idx = {p: i for i, p in enumerate(pool)}
    
    # Find all lines in the pool
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
            
    # Extract triples and constraint info
    # We want to select 20 points.
    # Constraint: No line can have 4 selected points.
    # Objective: Maximize number of lines with exactly 3 selected points.
    
    # To make the CSP tractable, we focus on "active" lines (lines with >= 3 points in the pool).
    # However, the "no 4 collinear" constraint applies to ALL lines, including those with only 2 points in the pool?
    # No, if a line has only 2 points in the pool, we can't select 4 from it.
    # So we only need to worry about lines with >= 3 points in the pool.
    
    active_lines = []
    for key, pts in lines_dict.items():
        if len(pts) >= 3:
            active_lines.append((key, list(pts)))
            
    # For the CSP, we need to know for each point, which lines it belongs to.
    point_to_lines = defaultdict(list)
    for idx, (key, pts) in enumerate(active_lines):
        for p in pts:
            point_to_lines[p].append(idx)
            
    # 3. Solve using Backtracking with Forward Checking
    # State:
    # selected: list of selected points
    # line_counts: dict line_idx -> count of selected points in that line
    # forbidden_lines: set of line indices that have reached count 3 (cannot select any more points on them)
    
    # We will use a randomized restart strategy with a greedy MRV (Minimum Remaining Values) heuristic.
    # Variable: The next point to select.
    # Domain: All points not yet selected.
    # Constraint: If we select point p, it must not be on any forbidden line.
    
    best_triples_found = -1
    best_points = None
    
    # To speed up, we can precompute for each point, the list of active lines it touches.
    # And for each line, the list of points.
    
    # Heuristic for variable ordering:
    # Pick the point that is on the most "tight" lines?
    # Or just randomize to explore different structures.
    
    # Let's try a few different strategies for point selection order.
    # Strategy A: Random order
    # Strategy B: Order by "density" (number of active lines the point is on)
    
    # Calculate density
    density = [len(point_to_lines[p]) for p in range(n_pool)]
    
    # We will run multiple iterations.
    # In each iteration, we pick 20 points.
    
    max_iterations = 5000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Randomize the order of points for selection
        # We want to encourage picking points that form triples.
        # But we must avoid 4-collinear.
        
        # Initialize state
        selected = set()
        line_counts = [0] * len(active_lines)
        forbidden_lines = set() # Lines with count >= 3
        
        # We pick 20 points one by one.
        # At each step, we choose a point from the available pool.
        # To guide the search towards high triple counts, we can bias towards points
        # that complete triples (lines with count 2).
        
        # However, a pure greedy might get stuck.
        # Let's use a "lookahead" or "randomized greedy" approach.
        
        # Available points: all points in pool not in selected.
        # We filter out points that are on any forbidden line.
        
        # To make it faster, we maintain a list of available points.
        # But n_pool is 225. Checking all 225 at each of 20 steps is 4500 checks.
        # 5000 iterations * 4500 = 22.5M checks. Might be tight in Python in 140s?
        # 22.5M simple checks is usually < 1s in PyPy, but CPython might take 5-10s.
        # Let's optimize.
        
        # Precompute for each point, the list of line indices.
        # When we select a point, we update line counts.
        
        # Let's try to reduce iterations or make the check faster.
        # We can precompute a bitmask of forbidden lines for each point?
        # No, forbidden lines change dynamically.
        
        # Alternative: Precompute for each point, the lines it is on.
        # When checking if a point is valid, we check if any of its lines are in forbidden_lines.
        
        valid_points = list(range(n_pool))
        
        # Shuffle to randomize
        random.seed(iteration)
        random.shuffle(valid_points)
        
        current_selected = []
        
        for step in range(20):
            if time.time() > deadline:
                break
                
            # Find the best valid point
            # Score:
            # 1. Must be valid (not on forbidden lines)
            # 2. Prefer points that complete triples (lines with count 2)
            # 3. Prefer points with high density (more lines) to build structure?
            
            best_p = -1
            best_score = -1
            
            # To speed up, we can sample a subset of valid_points?
            # Or just iterate. 225 iterations * 20 steps = 4500 per restart.
            # Let's iterate all.
            
            for p in valid_points:
                if p in selected:
                    continue
                    
                # Check validity
                is_valid = True
                score = 0
                
                for l_idx in point_to_lines[p]:
                    if l_idx in forbidden_lines:
                        is_valid = False
                        break
                    cnt = line_counts[l_idx]
                    if cnt == 2:
                        score += 10 # Completing a triple is good
                    elif cnt == 1:
                        score += 1
                    else:
                        score += 0.1 # Starting a line
                
                if is_valid:
                    # Add some randomness to break ties
                    score += random.random() * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_p = p
                        
            if best_p == -1:
                # No valid point found, restart
                break
                
            # Select best_p
            selected.add(best_p)
            current_selected.append(best_p)
            
            # Update line counts
            for l_idx in point_to_lines[best_p]:
                line_counts[l_idx] += 1
                if line_counts[l_idx] == 3:
                    forbidden_lines.add(l_idx)
                    
        if len(current_selected) == 20:
            # Check if this configuration is better
            # Count triples
            triples = sum(1 for l_idx in range(len(active_lines)) if line_counts[l_idx] == 3)
            
            if triples > best_triples_found:
                best_triples_found = triples
                # Convert indices to points
                pts_list = [pool[i] for i in current_selected]
                # Verify no 4 collinear (should be guaranteed by construction, but just in case)
                if _check_no_four_collinear(pts_list):
                    best_points = pts_list
                    # Optional: evaluate exact score?
                    # It's expensive. We can do it at the end for the best few.
                    # For now, we rely on triples as a proxy.
                    # Actually, let's evaluate the score for the new best to be sure.
                    # If we find a config with more triples, it's likely better.
                    # But "more triples" doesn't strictly mean "lower independence number".
                    # However, they are correlated.
                    
                    # Let's evaluate the score only if we have time.
                    # We'll collect the top few candidates and evaluate them at the end.
                    # Or just evaluate this one.
                    # _max_independent_set for n=20 is fast (<0.1s usually).
                    # 5000 iterations * 0.1s = 500s. Too slow.
                    # So we only evaluate the score if the triple count improves significantly or periodically.
                    
                    # Let's just track the best triple count and evaluate the final best one.
                    # Wait, the objective is to MINIMIZE the independence number.
                    # Maximizing triples is a proxy.
                    # Is it a good proxy?
                    # If we have many triples, we have many constraints, which should reduce the max independent set.
                    # Yes, it's a reasonable proxy.
                    
                    # We'll keep the points with the most triples.
                    
    # If we didn't find any valid points (shouldn't happen), fallback
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
    else:
        # Evaluate the exact score of the best candidate found
        # We might want to evaluate a few top candidates if we stored them.
        # For now, we just evaluate the one with max triples.
        # To be safer, we could store the top 5 and pick the one with min score.
        # But given time, we assume max triples correlates well.
        
        # Let's do a quick local search on the best point to see if we can improve?
        # No, let's just return it.
        
        # Actually, let's evaluate the score of the best candidate.
        # If it's bad, we might want to try a few others.
        # But we only stored the best one.
        
        # Let's modify the loop to store top 5 candidates by triple count.
        # And then evaluate their scores.
        
        # Since I didn't implement the top 5 storage in the loop above, I will just evaluate the current best.
        # If I had time, I'd refactor to store top K.
        
        # For now, return best_points.
        pass
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
