import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_score = 20
    best_config = None
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    # Helper: Check if a set of points has no 4 collinear
    def has_4_collinear(points):
        n = len(points)
        # For each pair, count collinear points
        for i in range(n):
            for j in range(i + 1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count >= 2: # Total 4 points
                    return True
        return False

    # Helper: Calculate max subset size with no 3 collinear (Optimized)
    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinear masks
        # coll[i][j] is bitmask of k such that i, j, k are collinear
        coll = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll[i][j] = mask
                coll[j][i] = mask

        best = 0
        
        def backtrack(A_mask, C_mask, depth):
            nonlocal best
            # Pruning
            if depth + C_mask.bit_count() <= best:
                return
            
            if C_mask == 0:
                if depth > best:
                    best = depth
                return
            
            # Heuristic: Pick a point from C that is "most constrained" or just first
            # Let's pick the first set bit
            p = (C_mask & -C_mask).bit_length() - 1
            
            # Option 1: Exclude p
            backtrack(A_mask, C_mask ^ (1 << p), depth)
            
            # Option 2: Include p
            # Check if p conflicts with A (i.e., if adding p creates a triple with any pair in A)
            # This is equivalent to: Does there exist a1, a2 in A such that p, a1, a2 are collinear?
            # We can check this by:
            # For each a in A, does coll[p][a] intersect A?
            # If yes, then adding p creates a triple (p, a, k) where k is in A and in coll[p][a].
            
            valid = True
            temp_A = A_mask
            while temp_A:
                a_idx = (temp_A & -temp_A).bit_length() - 1
                # If coll[p][a_idx] has any bit set that is also in A_mask, then invalid
                if coll[p][a_idx] & A_mask:
                    valid = False
                    break
                temp_A ^= (1 << a_idx)
                
            if valid:
                # New C_mask: remove p and any q in C that would form a triple with p and some a in A
                # If p, a, q are collinear, q cannot be added later if a is in A.
                # So we remove from C all q such that exists a in A with coll[p][a] having q.
                # new_C_mask = C_mask ^ (1 << p)
                # For each a in A: new_C_mask &= ~coll[p][a]
                
                new_C_mask = C_mask ^ (1 << p)
                temp_A = A_mask
                while temp_A:
                    a_idx = (temp_A & -temp_A).bit_length() - 1
                    new_C_mask &= ~coll[p][a_idx]
                    temp_A ^= (1 << a_idx)
                
                backtrack(A_mask | (1 << p), new_C_mask, depth + 1)

        backtrack(0, (1 << n) - 1, 0)
        return best

    def generate_candidate_greedy():
        pts = []
        # Start with a small random set in general position
        # Pick 5 random points
        while len(pts) < 5:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in pts and not has_4_collinear(pts + [p]):
                # Check no 3 collinear for the initial set? 
                # Actually, we want to maximize triples later, so initial set can have triples?
                # No, "no 4 collinear" is the hard constraint.
                # If we start with 3 collinear, that's fine.
                # But if we start with 3 collinear, we can't add a 4th on that line.
                # Let's just ensure no 4 collinear.
                pts.append(p)
        
        # Greedy expansion
        while len(pts) < 20:
            added = False
            # Find all pairs in pts
            pairs = list(itertools.combinations(range(len(pts)), 2))
            # Shuffle to avoid bias
            random.shuffle(pairs)
            
            for i, j in pairs:
                p1, p2 = pts[i], pts[j]
                # Find integer points on the line p1-p2
                # dx = p2[0] - p1[0], dy = p2[1] - p1[1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # We look for a point k = (p1[0] + t*dx, p1[1] + t*dy)
                # t is integer, t != 0, 1
                # To keep it simple, try t = -2, -1, 2, 3, -3, 4...
                # Or just scan a small range around the segment?
                # The inspiration says "dense, clumpy". So points should be close.
                
                # Let's try t values that keep coordinates in [-100, 100]
                for t in [-2, -3, 2, 3, -4, 4, -1, 1]:
                    if t == 0 or t == 1:
                        continue
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        new_pt = [nx, ny]
                        if new_pt in pts:
                            continue
                        # Check if adding new_pt creates 4 collinear
                        # Check all lines through new_pt
                        valid = True
                        for k in range(len(pts)):
                            if k == i or k == j:
                                continue
                            # Check if new_pt, pts[k], and any other pts[m] are collinear
                            # Actually, just check if new_pt, pts[k], pts[m] are collinear for any m
                            # This is O(N^2) per candidate. N=20, so 400 checks. 20 candidates. 8000 ops. Fast.
                            for m in range(len(pts)):
                                if m == k:
                                    continue
                                if are_collinear(new_pt, pts[k], pts[m]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            pts.append(new_pt)
                            added = True
                            break
                if added:
                    break
            
            if not added:
                # No triple extension found, add a random point
                found = False
                for _ in range(100):
                    p = [random.randint(-10, 10), random.randint(-10, 10)]
                    if p in pts:
                        continue
                    # Check 4 collinear
                    valid = True
                    for k in range(len(pts)):
                        for m in range(len(pts)):
                            if k == m:
                                continue
                            if are_collinear(p, pts[k], pts[m]):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        pts.append(p)
                        found = True
                        break
                if not found:
                    # Fallback: try wider range
                    for _ in range(100):
                        p = [random.randint(-50, 50), random.randint(-50, 50)]
                        if p in pts:
                            continue
                        valid = True
                        for k in range(len(pts)):
                            for m in range(len(pts)):
                                if k == m:
                                    continue
                                if are_collinear(p, pts[k], pts[m]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            pts.append(p)
                            found = True
                            break
                if not found:
                    return None # Failed to generate
            
        return pts

    # Run search
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        pts = generate_candidate_greedy()
        if pts is None:
            continue
        
        score = max_no_3_collinear(pts)
        if score < best_score:
            best_score = score
            best_config = pts
            
    if best_config is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_config
