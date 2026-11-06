import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 135  # 15s margin for safety
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    # Helper: Count total collinear triples in a set of points
    def count_triples(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count
        
    # Helper: Check if any 4 points are collinear
    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Check if there is a 4th point on the line defined by i, j, k
                    # We can just check all other points
                    line_points = [points[i], points[j], points[k]]
                    for m in range(n):
                        if m != i and m != j and m != k:
                            if are_collinear(points[i], points[j], points[m]):
                                return True
        return False
        
    # Helper: Exact branch and bound for largest subset with no 3 collinear
    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinearity graph
        # adj[i] is set of indices j such that i, j, and some k are collinear? 
        # Actually, we need to know if adding point i to a set S creates a collinear triple.
        # A triple (i, j, k) is bad if i, j, k are collinear.
        # Let's precompute for each pair (i, j), the set of points k that are collinear with i and j.
        # If we have a current set S, and we consider adding i, we must ensure that for any j, k in S, 
        # i is not collinear with j and k.
        # This is equivalent to: for any j in S, i is not in the "forbidden" set defined by j and any other k in S?
        # Simpler: Precompute bad_triples. A set S is valid if it contains no bad triple.
        
        # For speed, let's use a recursive backtracking with pruning.
        # State: current set S, index i (next candidate to consider)
        # Pruning: if len(S) + (n - i) <= best_found, prune.
        
        best = 0
        # Sort points? Maybe by degree of collinearity to prune faster?
        # Let's just use indices 0..n-1
        
        # Precompute collinearity matrix or sets
        # collinear_with[i] = set of j such that i, j are part of some collinear triple?
        # Actually, we just need to check if adding i to S creates a triple.
        # i creates a triple with j, k in S if i, j, k are collinear.
        # So for each i, and each pair (j, k) in S, check collinearity.
        # To speed this up, for each i, precompute a bitmask or set of pairs (j, k) that form a collinear triple with i.
        
        bad_pairs_for = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                for k in range(n):
                    if i == k or j == k: continue
                    if are_collinear(points[i], points[j], points[k]):
                        bad_pairs_for[i].add((j, k))
                        # Note: (j,k) and (k,j) are the same pair for the purpose of "j and k are in S"
        
        def backtrack(S, start_idx):
            nonlocal best
            current_len = len(S)
            # Prune: even if we take all remaining, can we beat best?
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip point start_idx
            backtrack(S, start_idx + 1)
            
            # Option 2: Include point start_idx
            # Check if adding start_idx to S is valid
            # It is valid if there is no pair (j, k) in S such that (j, k) is in bad_pairs_for[start_idx]
            # Note: bad_pairs_for[start_idx] contains pairs (j, k) where j != k.
            # We need to check if any such pair is fully contained in S.
            
            valid = True
            # To make this faster, we can precompute for each i, the list of pairs.
            # But checking against S (a list/set) is O(|S|^2) if done naively.
            # Since n=20, |S| <= 20, this is fast enough.
            
            for j in S:
                for k in S:
                    if j == k: continue
                    # Check if (j, k) is a bad pair for start_idx
                    if (j, k) in bad_pairs_for[start_idx]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(start_idx)
                backtrack(S, start_idx + 1)
                S.pop()
                
        backtrack([], 0)
        return best

    # Initial Configuration: Random points, then repair
    def generate_initial_config():
        pts = []
        attempts = 0
        while len(pts) < 20 and attempts < 1000:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            # Check distinctness
            if p in pts:
                attempts += 1
                continue
            # Check no 4 collinear with existing points
            # If we add p, does it create a 4-collinear set?
            # This happens if there are 3 existing points collinear with p.
            # Or if there are 2 existing points collinear with p, and we already had another pair? No.
            # We just need to ensure no 4 points are collinear in the new set.
            # A new 4-collinear set must include p.
            # So we need to check if there are 3 points in pts that are collinear with p.
            # Let's find all points in pts collinear with p.
            collinear_pts = []
            for q in pts:
                if are_collinear(p, q, [0,0]): # Dummy check, just check collinearity with any other?
                    pass
                # Actually, we just check if p is collinear with any pair (q1, q2) in pts?
                # No, if p is collinear with q1 and q2, that's a triple.
                # If p is collinear with q1, q2, q3, that's a 4-set.
                # So we need to count how many points in pts are collinear with p.
                # If count >= 3, adding p creates a 4-collinear set.
                # Wait, if q1, q2, q3 are collinear, and p is on that line, then {p, q1, q2, q3} is 4 collinear.
                # So we just need to check if p lies on a line that already has 3 points?
                # No, if p lies on a line with 2 points, that's fine (triple).
                # If p lies on a line with 3 points, that's bad (quadruple).
                
                # Let's just collect all points collinear with p.
                # If any 3 of them are mutually collinear (which they are if they are all on the same line as p), 
                # then we have 4 points on that line.
                # So: count points q in pts such that p and q define a line that contains another point r in pts?
                # Simpler: 
                # For each pair (q1, q2) in pts, check if p is collinear with them.
                # If yes, then {p, q1, q2} is a triple.
                # We need to know if there is a 4th point.
                # If we add p, the only new lines are those through p.
                # A line through p and some q.
                # If there are 2 other points on that line, we have 4 points.
                
                pass
            
            # Let's use a simpler check for the initial generation:
            # Just generate random points, and if has_4_collinear(new_pts), reject.
            new_pts = pts + [p]
            if not has_4_collinear(new_pts):
                pts = new_pts
            else:
                attempts += 1
        # If we didn't get 20, fill with fixed offsets
        while len(pts) < 20:
            x = len(pts) * 2
            y = len(pts) * 3
            p = [x, y]
            if p not in pts and not has_4_collinear(pts + [p]):
                pts.append(p)
            else:
                # Force add, will repair later
                pts.append(p)
        return pts

    def repair(points):
        """Remove or move points to resolve 4-collinearities."""
        pts = list(points)
        # While there is a 4-collinear set, move one of them to a random location
        for _ in range(100):
            if not has_4_collinear(pts):
                break
            # Find a 4-collinear set
            found = False
            for i in range(20):
                for j in range(i+1, 20):
                    for k in range(j+1, 20):
                        for m in range(k+1, 20):
                            if are_collinear(pts[i], pts[j], pts[k]) and are_collinear(pts[i], pts[j], pts[m]):
                                # Move pts[m] to a random location
                                new_x = random.randint(-100, 100)
                                new_y = random.randint(-100, 100)
                                pts[m] = [new_x, new_y]
                                found = True
                                break
                        if found: break
                    if found: break
                if found: break
            if not found:
                break
        return pts

    def optimize(points):
        """Hill climbing to increase number of collinear triples."""
        pts = [list(p) for p in points]
        current_triples = count_triples(pts)
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_p = pts[idx][:]
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            pts[idx] = [new_x, new_y]
            
            # Check validity: no 4 collinear
            if has_4_collinear(pts):
                pts[idx] = old_p
                continue
                
            # Check distinctness
            if len(set(map(tuple, pts))) < 20:
                pts[idx] = old_p
                continue
                
            new_triples = count_triples(pts)
            
            # Accept if better (more triples is better for minimizing max subset size? 
            # Generally, more triples constrain the no-3-collinear subset more)
            if new_triples >= current_triples:
                current_triples = new_triples
            else:
                # Revert
                pts[idx] = old_p
                
        return pts

    # Main loop
    best_config = None
    best_score = 21  # Max possible score is 20
    
    while time.time() < deadline:
        # 1. Generate
        pts = generate_initial_config()
        
        # 2. Repair
        pts = repair(pts)
        
        # 3. Optimize
        pts = optimize(pts)
        
        # 4. Evaluate
        # Check validity again
        if len(set(map(tuple, pts))) != 20:
            continue
        if has_4_collinear(pts):
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
            
        score = max_no_3_collinear(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
            
        # If we found a very good score (e.g., 10 or lower), we might want to keep searching for even better
        # But let's run a few iterations to see if we can beat 15
        if best_score <= 12:
             # Try a few more optimizations on the best config found so far
             if time.time() < deadline - 5:
                 improved = optimize(best_config)
                 if not has_4_collinear(improved) and len(set(map(tuple, improved))) == 20:
                     s = max_no_3_collinear(improved)
                     if s < best_score:
                         best_score = s
                         best_config = improved

    if best_config is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_config

# EVOLVE-BLOCK-END
