"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from collections import defaultdict

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _line_key(p1, p2):
    """
    Canonical representation of the line passing through p1 and p2.
    Returns a tuple (A, B, C) for Ax + By + C = 0, normalized.
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    # Normalize
    if A == 0 and B == 0:
        return None
    g = 1
    if A != 0:
        g = abs(A)
        if B != 0:
            g = __import__('math').gcd(g, abs(B))
            g = __import__('math').gcd(g, abs(C))
        else:
            g = __import__('math').gcd(g, abs(C))
    else:
        g = abs(B)
        if C != 0:
            g = __import__('math').gcd(g, abs(C))
            
    if g != 0:
        A //= g
        B //= g
        C //= g
    
    # Ensure consistent sign: first non-zero of A, B, C should be positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    lines = defaultdict(list)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key:
                lines[key].append((i, j))
                
    # Check if any line has 4 or more points
    # We need to reconstruct which points are on the line
    # A simpler way for n=20 is O(n^4) check or using the line map properly.
    # Let's just do the O(n^4) check as it's small enough and less error prone for correctness first.
    for i, j, k, l in combinations(range(n), 4):
        if _are_collinear(points[i], points[j], points[k]) and \
           _are_collinear(points[i], points[j], points[l]):
            return True
    return False

def _max_no_three_collinear(points):
    """
    Find the largest subset with no three collinear using branch and bound.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute conflict pairs: if i, j, k are collinear, then i conflicts with (j, k) etc.
    # For B&B, we need to know if adding p to S creates a triple.
    # This happens if there exist j, k in S such that p, j, k are collinear.
    
    # conflict[p] = list of pairs (j, k) such that p, j, k are collinear.
    # When checking if p can be added to S, we check if any (j, k) in conflict[p] has both j, k in S.
    conflict = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    conflict[i].append((j, k))
                    conflict[j].append((i, k))
                    conflict[k].append((i, j))
                    
    # Sort points by "density" of conflicts to help B&B prune earlier?
    # Or just use fixed order. Let's try sorting by number of conflicts.
    order = sorted(range(n), key=lambda x: len(conflict[x]), reverse=True)
    # But we need to map original indices to new order?
    # It's easier to just use the original indices and pick the "best" candidate dynamically or pre-sort.
    # Let's pre-sort indices by conflict count descending.
    # We will iterate through candidates in this order.
    
    best = [0]
    
    def can_add(p, S_mask):
        for j, k in conflict[p]:
            if (S_mask & (1 << j)) and (S_mask & (1 << k)):
                return False
        return True

    def bnb(S_mask, R_mask, depth):
        # R_mask contains candidates not yet decided.
        # We pick the next candidate from R_mask.
        # To improve pruning, we should pick the candidate with highest conflict with S?
        # Or just the first one in the sorted order that is in R_mask?
        
        if depth + bin(R_mask).count('1') <= best[0]:
            return
            
        if R_mask == 0:
            if depth > best[0]:
                best[0] = depth
            return
            
        # Find the next candidate to consider.
        # Heuristic: Pick the point in R that is "most constrained" by S?
        # Or just pick the first one in our pre-sorted `order` that is in R.
        # Pre-sorting by total conflicts is a good static heuristic.
        
        p = -1
        # Iterate through pre-sorted order to find first one in R
        for idx in order:
            if R_mask & (1 << idx):
                p = idx
                break
                
        if p == -1:
            return
            
        bit = 1 << p
        new_R = R_mask & ~bit
        
        # Branch 1: Include p
        if can_add(p, S_mask):
            bnb(S_mask | bit, new_R, depth + 1)
            
        # Branch 2: Exclude p
        bnb(S_mask, new_R, depth)
        
    bnb(0, (1 << n) - 1, 0)
    return best[0]

def _construct_line_dense(deadline):
    """
    Construct a set of 20 points by greedily extending lines.
    """
    points = []
    line_points = defaultdict(list) # line_key -> list of point indices
    
    # Start with a random triangle or small set
    for _ in range(3):
        while True:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in points:
                break
        points.append(p)
        
    # Update lines for initial points
    for i in range(3):
        for j in range(i+1, 3):
            key = _line_key(points[i], points[j])
            line_points[key].append(i)
            line_points[key].append(j)
            
    while len(points) < 20 and time.time() < deadline:
        # Try to find a point that extends an existing line
        best_p = None
        best_score = -1
        
        # Candidates for new points
        # 1. Points on lines that currently have 2 or 3 points
        # 2. Random points
        
        # Generate candidates from existing lines
        candidates = set()
        for key, pts in line_points.items():
            if 2 <= len(pts) <= 3: # Extend lines of size 2 or 3
                # Get the line equation
                A, B, C = key
                # Find integer points on this line within bounds
                # Ax + By = -C
                # We can parameterize or just scan a small range around existing points?
                # Scanning the whole grid is too slow.
                # Let's just pick random points on the line within bounds.
                # If A != 0: x = ( -C - By ) / A
                # If B != 0: y = ( -C - Ax ) / B
                
                # Try a few random points on the line
                for _ in range(5):
                    if A != 0:
                        # Pick random y
                        y = random.randint(-100, 100)
                        if (-C - B*y) % A == 0:
                            x = (-C - B*y) // A
                            if -100 <= x <= 100:
                                candidates.add([x, y])
                    elif B != 0:
                        x = random.randint(-100, 100)
                        if (-C - A*x) % B == 0:
                            y = (-C - A*x) // B
                            if -100 <= y <= 100:
                                candidates.add([x, y])
                    else:
                        # Degenerate, shouldn't happen
                        pass
        
        # Add some random points to ensure diversity
        for _ in range(10):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            candidates.add([x, y])
            
        # Evaluate candidates
        for p in candidates:
            if p in points:
                continue
                
            new_points = points + [p]
            
            # Check validity: no 4 collinear
            # We can do a quick check: does p create a 4-collinear set?
            # Count how many existing points are collinear with p and any other existing point.
            # Actually, just checking _has_four_collinear on the new set is O(n^4) which is fine for n=20.
            # But we can be smarter: check lines passing through p.
            valid = True
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if _are_collinear(p, points[i], points[j]):
                        # Check if there is a 4th point on this line
                        # We already know p, points[i], points[j] are collinear.
                        # If there is a k such that points[k] is also on this line, we have 4.
                        # We can check against line_points?
                        # Or just check all k.
                        for k in range(len(points)):
                            if k != i and k != j and _are_collinear(p, points[i], points[k]):
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Score: how many new collinear triples does it create?
            # Or how many lines of size 2/3 does it extend?
            # We want to maximize density.
            score = 0
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if _are_collinear(p, points[i], points[j]):
                        score += 10 # Creating a triple is good
                        # If this line already had 2 points (i,j), now it has 3.
                        # If it had 3, now it has 4 (but we checked validity, so it must have been <=2 before? No, we checked no 4-collinear in NEW set. So if it had 3 before, adding p makes 4 -> invalid. So it must have been <=2).
                        # So we are extending a line of size 2 to 3, or 1 to 2 (if i,j was the only other point? No, i,j are distinct).
                        # Actually, if i,j,k are collinear, and p is new.
                        # If p, i, j are collinear, and i, j were the only ones, we create a triple.
                        
            if score > best_score:
                best_score = score
                best_p = p
                
        if best_p is None:
            # Fallback: random
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = [x, y]
                if p not in points:
                    break
            best_p = p
            
        # Add best_p
        points.append(best_p)
        
        # Update line_points
        idx = len(points) - 1
        for i in range(idx):
            key = _line_key(points[idx], points[i])
            if key:
                line_points[key].append(idx)
                line_points[key].append(i)
                
    return points

def _local_search(points, deadline):
    best_points = points[:]
    best_score = _max_no_three_collinear(points)
    current_points = points[:]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(20):
            if time.time() > deadline:
                break
            best_new_point = None
            best_new_score = best_score
            
            for _ in range(5): # Fewer tries per move
                if time.time() > deadline:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = [x, y]
                if new_p in current_points:
                    continue
                
                new_points = current_points[:]
                new_points[idx] = new_p
                if _has_four_collinear(new_points):
                    continue
                
                new_score = _max_no_three_collinear(new_points)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_point = new_p
                    
            if best_new_point is not None:
                current_points[idx] = best_new_point
                if best_new_score < best_score:
                    best_score = best_new_score
                    best_points = current_points[:]
                improved = True
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_score = 20
    
    # Try a few constructive restarts
    for _ in range(5):
        if time.time() > deadline - 20:
            break
            
        points = _construct_line_dense(deadline - 20)
        if points is None or len(points) != 20:
            continue
            
        # Polish with local search
        improved_points, score = _local_search(points, deadline - 10)
        
        if score < best_score:
            best_score = score
            best_points = improved_points[:]
            
        if best_score <= 8: # Theoretical lower bound is unknown, but 8 is very good
            break
            
    if best_points is None:
        # Fallback to random
        while time.time() < deadline:
            points = []
            for _ in range(20):
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = [x, y]
                    if p not in points:
                        break
                points.append(p)
            if not _has_four_collinear(points):
                best_points = points
                best_score = _max_no_three_collinear(points)
                break
                
    return best_points
# EVOLVE-BLOCK-END
