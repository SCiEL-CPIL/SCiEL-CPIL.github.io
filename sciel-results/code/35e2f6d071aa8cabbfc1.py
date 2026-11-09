"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i, j, k, l in combinations(range(n), 4):
        if _are_collinear(points[i], points[j], points[k]) and \
           _are_collinear(points[i], points[j], points[l]) and \
           _are_collinear(points[i], points[k], points[l]):
            return True
    return False

def _count_collinear_triples(points):
    """Count the number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_no_three_collinear(points):
    """
    Find the largest subset with no three collinear using branch and bound.
    Uses a more efficient approach with bitmasks.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute collinear triples as a list of (i,j,k)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    if not triples:
        return n
    
    # For each point, store which other points it's collinear with in a triple
    # conf[p] = set of q such that there exists r with (p,q,r) collinear
    conf = [set() for _ in range(n)]
    for i, j, k in triples:
        conf[i].add(j)
        conf[i].add(k)
        conf[j].add(i)
        conf[j].add(k)
        conf[k].add(i)
        conf[k].add(j)
    
    best = [0]
    
    def can_add(p, S_mask):
        """Check if point p can be added to subset (bitmask S_mask) without creating a collinear triple."""
        # For each pair (q, r) in S_mask, check if (p, q, r) is collinear
        # More efficient: for each q in S_mask, check if there exists r in S_mask with (p,q,r) collinear
        # We need to check all pairs in S that are collinear with p
        # Precompute: for point p, the set of pairs {q,r} collinear with p
        for q in conf[p]:
            if S_mask & (1 << q):
                # q is in S and collinear with p in some triple
                # Check if there's an r in S such that (p, q, r) collinear
                # This means r is in conf[q] and r != p
                for r in conf[q]:
                    if r != p and (S_mask & (1 << r)):
                        return False
        return True
    
    def bnb(S_mask, R_mask, depth):
        """
        S_mask: bitmask of points in current subset
        R_mask: bitmask of remaining candidate points
        depth: number of points in S
        """
        # Upper bound: current size + remaining
        if depth + bin(R_mask).count('1') <= best[0]:
            return
        
        if R_mask == 0:
            if depth > best[0]:
                best[0] = depth
            return
        
        # Pick the most constrained point from R
        # Heuristic: pick point with most conflicts in S
        best_p = -1
        best_conflict = -1
        temp = R_mask
        while temp:
            p = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            # Count conflicts: how many pairs in S would be violated
            conflicts = 0
            for q in conf[p]:
                if S_mask & (1 << q):
                    conflicts += 1
            if conflicts > best_conflict:
                best_conflict = conflicts
                best_p = p
        
        if best_p == -1:
            # No conflicts, take all remaining
            new_depth = depth + bin(R_mask).count('1')
            if new_depth > best[0]:
                best[0] = new_depth
            return
        
        p = best_p
        bit = 1 << p
        new_R = R_mask & ~bit
        
        # Option 1: Add p
        if can_add(p, S_mask):
            bnb(S_mask | bit, new_R, depth + 1)
        
        # Option 2: Don't add p
        bnb(S_mask, new_R, depth)
    
    bnb(0, (1 << n) - 1, 0)
    return best[0]

def _construct_line_dense():
    """
    Construct 20 points by placing them on a small number of shared lines.
    Strategy: use 3-4 lines, each containing 3-4 points (but no line has 4+ total
    from the full set to avoid 4-collinear). We want many triples to make
    3-free subsets small.
    
    Key idea: pick lines such that points are shared across lines, creating
    a dense hypergraph of collinear triples.
    """
    # Use a grid-like structure but perturb to avoid 4-collinear
    # Start with points on a few lines
    
    # Line 1: y = 0, points at x = -30, -10, 10 (3 points)
    # Line 2: y = 20, points at x = -20, 0, 20 (3 points)
    # Line 3: y = 40, points at x = -10, 10, 30 (3 points)
    # Line 4: x = 0, points at y = -20, 0, 20 (but 0,0 and 0,20 already used)
    # Need to be careful about duplicates and 4-collinear
    
    # Better: use a structured approach
    # Place points at intersections of lines, then add extras
    
    # Let's try: 4 lines, each with 5 points, but points shared
    # Actually simpler: just generate and check
    
    points = []
    
    # Create points on several lines, ensuring no 4 are collinear
    # Line A: y = 0, x in {-40, -20, 0, 20} -> 4 points, but we can only use 3
    # Let's use 3 per line
    
    # Line 1: y = 0
    points.append([-40, 0])
    points.append([-10, 0])
    points.append([20, 0])
    
    # Line 2: y = 30
    points.append([-30, 30])
    points.append([0, 30])
    points.append([30, 30])
    
    # Line 3: x = 0
    points.append([0, -30])
    points.append([0, 15])
    # [0, 30] already in line 2, skip
    
    # Line 4: x = -20
    points.append([-20, -20])
    points.append([-20, 10])
    # [-20, 0] not in set yet
    
    # Line 5: x = 20
    points.append([20, -20])
    points.append([20, 10])
    # [20, 0] already in line 1
    
    # Line 6: y = -20
    points.append([-10, -20])
    points.append([10, -20])
    # [-20, -20] and [20, -20] already in set
    
    # That's 14 points. Add 6 more.
    # Line 7: diagonal y = x + 10
    points.append([-30, -20])  # y = x + 10
    points.append([10, 20])    # y = x + 10
    # Check: are any 3 of these collinear with existing?
    
    # Line 8: another diagonal
    points.append([-10, -30])
    points.append([30, 10])
    
    # Need 2 more
    points.append([-50, 15])
    points.append([50, 15])
    
    # Check for duplicates
    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)
    
    # Pad if needed
    while len(unique_points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = [x, y]
        if p not in unique_points:
            unique_points.append(p)
    
    unique_points = unique_points[:20]
    
    # Check validity
    if _has_four_collinear(unique_points):
        return None
    
    return unique_points

def _local_search_line_density(points, deadline):
    """
    Local search that tries to increase line density (more collinear triples)
    while maintaining no-4-collinear constraint.
    """
    best_points = points[:]
    best_score = _max_no_three_collinear(points)
    current_points = points[:]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(20):
            if time.time() > deadline:
                break
            
            best_new = None
            best_new_score = best_score
            old_triples = _count_collinear_triples(current_points)
            
            for attempt in range(50):
                if time.time() > deadline:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = [x, y]
                if new_p in current_points:
                    continue
                
                new_points = current_points[:idx] + [new_p] + current_points[idx+1:]
                if _has_four_collinear(new_points):
                    continue
                
                new_triples = _count_collinear_triples(new_points)
                if new_triples <= old_triples:
                    continue
                
                new_score = _max_no_three_collinear(new_points)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new = new_points[:]
            
            if best_new is not None:
                current_points = best_new
                if best_new_score < best_score:
                    best_score = best_new_score
                    best_points = current_points[:]
                improved = True
    
    return best_points, best_score

def _swap_search(points, deadline):
    """
    Try swapping pairs of points to find better configurations.
    """
    best_points = points[:]
    best_score = _max_no_three_collinear(points)
    current_points = points[:]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(20):
            if time.time() > deadline:
                break
            for j in range(i+1, 20):
                if time.time() > deadline:
                    break
                
                # Try moving point i to a random location
                for attempt in range(10):
                    if time.time() > deadline:
                        break
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    new_p = [x, y]
                    if new_p in current_points:
                        continue
                    
                    new_points = current_points[:]
                    new_points[i] = new_p
                    if _has_four_collinear(new_points):
                        continue
                    
                    new_triples = _count_collinear_triples(new_points)
                    old_triples = _count_collinear_triples(current_points)
                    if new_triples <= old_triples:
                        continue
                    
                    new_score = _max_no_three_collinear(new_points)
                    if new_score < best_score:
                        best_score = new_score
                        best_points = new_points[:]
                        current_points = new_points[:]
                        improved = True
                        break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_score = 20  # Worst possible
    
    # Try structured construction
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        points = _construct_line_dense()
        if points is not None:
            score = _max_no_three_collinear(points)
            if score < best_score:
                best_score = score
                best_points = points[:]
    
    # If structured construction failed, use random valid initial
    if best_points is None:
        while True:
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
                break
        best_points = points
        best_score = _max_no_three_collinear(best_points)
    
    # Local search to improve
    if time.time() < deadline - 10:
        best_points, best_score = _local_search_line_density(best_points, deadline - 10)
    
    # Swap search if time remains
    if time.time() < deadline - 5:
        best_points, best_score = _swap_search(best_points, deadline - 5)
    
    return best_points
# EVOLVE-BLOCK-END
