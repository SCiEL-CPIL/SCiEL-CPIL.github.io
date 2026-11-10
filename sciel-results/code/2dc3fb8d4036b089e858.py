"""Minimize largest 3-collinear-free subset for 20 points via stochastic search."""
import time
import random
import sys

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _count_triples(points):
    """Count total number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j+1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
    return count

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    This is the score we want to minimize.
    """
    n = len(points)
    
    # Precompute collinear pairs (edges in conflict graph)
    # Conflict graph: edge (i, j) if there exists k such that i, j, k are collinear?
    # No, the constraint is "no three collinear".
    # This is equivalent to finding the maximum independent set in a 3-uniform hypergraph?
    # Or simply: A subset S is valid if no three points in S are collinear.
    
    # Let's use a simple backtracking with pruning.
    # State: current_set, candidates (indices not yet considered)
    # Pruning: if len(current_set) + len(candidates) <= best, prune.
    
    best = 0
    # Sort points to help pruning? Maybe not necessary for n=20.
    
    # Precompute for each pair (i, j), the set of points collinear with them?
    # Or for each point i, and slope, the points on that line?
    
    # Faster check for validity of a set:
    # When adding point p to set S, check if any pair in S is collinear with p.
    
    def backtrack(current_set, candidates):
        nonlocal best
        if len(candidates) == 0:
            if len(current_set) > best:
                best = len(current_set)
            return
        
        # Pruning
        if len(current_set) + len(candidates) <= best:
            return
            
        # Heuristic: pick a candidate
        # Greedy choice: pick the one that conflicts with the fewest other candidates?
        # Or just pick the first one.
        # For n=20, simple branching is fine if pruning is good.
        
        # Let's try to pick a candidate that is likely to be in the max set.
        # Or just iterate.
        
        for idx in range(len(candidates)):
            p_idx = candidates[idx]
            
            # Check if p_idx can be added to current_set
            valid = True
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    p1 = points[current_set[i]]
                    p2 = points[current_set[j]]
                    p3 = points[p_idx]
                    # Check collinearity
                    dx1 = p2[0] - p1[0]
                    dy1 = p2[1] - p1[1]
                    dx2 = p3[0] - p1[0]
                    dy2 = p3[1] - p1[1]
                    if dx1 * dy2 == dy1 * dx2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_set = current_set + [p_idx]
                new_candidates = candidates[:idx] + candidates[idx+1:]
                backtrack(new_set, new_candidates)
            
            # Also consider NOT adding p_idx
            # But we must be careful not to duplicate states.
            # Standard MIS backtracking:
            # 1. Include p_idx
            # 2. Exclude p_idx (remove from candidates, do not add to set)
            # Wait, the loop above handles "Include". We need to handle "Exclude" separately or in the loop structure.
            
            # Let's restructure backtrack to be cleaner.
            # Actually, the loop above tries to include each candidate.
            # It doesn't explicitly branch on "exclude current candidate and move to next".
            # The standard approach:
            # pick a vertex v.
            # Branch 1: Include v. (Remove v and its neighbors from candidates? No, neighbors are not invalid, just can't be with v. But in 3-uniform, "neighbors" are pairs.)
            # This is complex for 3-uniform hypergraphs.
            
            # Simpler: Just try all subsets? 2^20 is 1M. Too slow for Python in 150s?
            # 1M * 20^2 checks = 400M ops. Might be too slow.
            
            # Let's stick to a simpler heuristic search for the score, or a faster exact solver.
            # Given n=20, we can precompute all collinear triples.
            # A subset is valid if it contains no triple from the list.
            
            # Let's optimize the backtrack function.
            pass

    # Let's implement a faster exact solver using bitmasks or precomputed conflicts.
    # Precompute for each pair (i, j) the list of k such that i, j, k collinear.
    # When we have a set S, it is invalid if there exists i, j in S and k in S such that (i, j, k) is a collinear triple.
    
    # Better: Precompute for each point i, the set of points that form a collinear triple with i and some other point?
    # No, just precompute the set of all collinear triples T.
    # A set S is valid iff S does not contain any t in T.
    
    # Let's use a recursive search that builds the set incrementally.
    # To speed it up, we can use the fact that if we have points i, j in the set, we cannot add any k such that (i, j, k) is collinear.
    
    # Let's restart the exact solver with this logic.
    
    n = len(points)
    # collinear_with[i][j] = set of k such that i, j, k are collinear
    collinear_with = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    collinear_with[i][j].add(k)
                    collinear_with[j][i].add(k)

    best = 0
    
    def backtrack(current_set, candidates):
        nonlocal best
        if len(current_set) + len(candidates) <= best:
            return
            
        if not candidates:
            if len(current_set) > best:
                best = len(current_set)
            return

        # Pick a candidate
        # Heuristic: Pick the one with the most conflicts? Or least?
        # Usually, picking a vertex with high degree helps prune in MIS.
        # Here, "degree" is not well defined.
        # Let's just pick the first one.
        
        p_idx = candidates[0]
        rest_candidates = candidates[1:]
        
        # Branch 1: Include p_idx
        # New candidates: those in rest_candidates that are not collinear with any pair in current_set + {p_idx}
        # Specifically, if we add p_idx, we cannot add any k such that:
        # 1. (current_set[i], current_set[j], k) is collinear (already handled by current_set validity? No, current_set is valid, so no such k exists in candidates if candidates were filtered properly? 
        #    Wait, if current_set is valid, then no two points in it form a line with a third point in the *entire* set? No, they might form a line with a point NOT in current_set.
        #    So when we add p_idx, we must remove from candidates any k that forms a collinear triple with p_idx and any point in current_set.
        #    Also, we must remove any k that forms a collinear triple with two points in current_set? 
        #    If current_set is valid, no two points in current_set form a collinear triple with ANY point? 
        #    No. It's possible that i, j in current_set and k in candidates are collinear. 
        #    But if i, j, k are collinear, then {i, j, k} is a collinear triple. 
        #    Since current_set is valid, it doesn't contain k. 
        #    But if we don't add k, it's fine. 
        #    However, if we add p_idx, do we care about k? 
        #    Only if p_idx, i, k are collinear or p_idx, j, k are collinear.
        
        new_candidates = []
        for k in rest_candidates:
            # Check if k can be added later.
            # k is invalid for future addition if it forms a collinear triple with p_idx and any point in current_set.
            can_add = True
            for c in current_set:
                if k in collinear_with[p_idx][c]:
                    can_add = False
                    break
            if can_add:
                new_candidates.append(k)
        
        backtrack(current_set + [p_idx], new_candidates)
        
        # Branch 2: Exclude p_idx
        backtrack(current_set, rest_candidates)

    backtrack([], list(range(n)))
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: random or simple
    # Let's start with a random set of points
    # Or start with the parabola and perturb?
    # Parabola has score 20. We want lower.
    # Random points likely have many collinear triples? No, random points in a grid have few collinear triples.
    # But we want to MINIMIZE the max independent set.
    # This means we want MANY collinear triples, but NO four collinear.
    # So we want a configuration where points are arranged in many lines of exactly 3 points.
    
    # Strategy:
    # 1. Generate random valid configurations (no 4 collinear).
    # 2. Calculate score (max independent set).
    # 3. Local search: move a point to a random location, if valid and score decreases, accept.
    #    Or simulated annealing.
    
    best_points = None
    best_score = float('inf')
    
    # Helper to generate a random valid point
    def random_point():
        while True:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            return [x, y]

    # Initialize with a random set of 20 points
    # Ensure no 4 collinear
    def generate_initial():
        pts = []
        while len(pts) < 20:
            p = random_point()
            if p not in pts:
                # Check if adding p creates a 4-collinear line
                valid = True
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        # Check if p is on line ij
                        if dx * (p[1] - y1) == dy * (p[0] - x1):
                            # Count how many other points are on this line
                            count = 2
                            for k in range(len(pts)):
                                if k == i or k == j:
                                    continue
                                x3, y3 = pts[k]
                                if dx * (y3 - y1) == dy * (x3 - x1):
                                    count += 1
                            if count >= 3: # Adding p makes 4
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    pts.append(p)
        return pts

    current_points = generate_initial()
    current_score = _max_independent_set(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        new_p = random_point()
        
        if new_p == old_p:
            continue
            
        # Check if new_p is already in points
        if new_p in current_points:
            continue
            
        # Check validity of new configuration
        # Temporarily replace
        current_points[idx] = new_p
        
        if not _check_no_four_collinear(current_points):
            current_points[idx] = old_p
            continue
            
        new_score = _max_independent_set(current_points)
        
        # Accept if better, or with probability exp(-(delta)/T)
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            current_points[idx] = old_p
            
        T *= alpha
        
    return best_points

# Run construct to test? No, just define it.
# But we need to make sure it returns exactly 20 points.
# The generate_initial ensures 20 distinct points.
# The moves ensure distinct points.
# The validity check ensures no 4 collinear.

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
