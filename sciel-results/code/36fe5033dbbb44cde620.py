"""
Strategy: Maximum Weight Independent Set on a Conflict Graph of 4-Point Lines.

1. Enumerate all potential 4-point lines within a small bounding box.
2. Build a conflict graph where edges connect lines that are "incompatible" 
   (they share 3+ points, or their union creates a 5-collinear set with other selected lines).
   Note: To simplify, we primarily conflict lines that share 3+ points (since two 4-point lines 
   sharing 3 points would imply 5 points if the 4th points are distinct and on different lines? 
   Actually, if L1={a,b,c,d} and L2={a,b,c,e}, they share 3. The set {a,b,c,d,e} has 5 points. 
   Are they collinear? a,b,c are on L1 and L2. So a,b,c define the line. d is on L1, e is on L2.
   If L1 != L2, then d is not on L2 and e is not on L1. Thus a,b,c,d,e are NOT all collinear 
   unless L1==L2. Wait. If a,b,c are collinear, and d is on that line, and e is on that line, 
   then all 5 are collinear. 
   So, if two 4-point lines share 3 points, they lie on the SAME geometric line. 
   Therefore, selecting both would mean we have at least 5 points on that geometric line. 
   This violates the "no 5 collinear" constraint.
   So, any two 4-point lines that share 3 or more points are incompatible.
   
   What about lines that share 2 points? 
   L1={a,b,c,d}, L2={a,b,e,f}. They share a,b. 
   The points are {a,b,c,d,e,f}. No 5 are necessarily collinear. 
   However, we must ensure that the resulting set of points doesn't accidentally have 5 collinear 
   from a combination of 3+ lines. This is harder to pre-compute in the graph. 
   But for a first pass, conflicting on 3+ shared points is the strict necessary condition for 
   immediate 5-collinear violation from just those two lines.
   
   We will use a Greedy Set Cover / Independent Set approach:
   - Generate candidate lines (4-subsets of points in a grid) that are collinear.
   - Sort by "density" or potential utility.
   - Greedily select lines that don't conflict (share < 3 points) and build the point set.
   - Refine with local search on the resulting points.
"""

import time
import random
import itertools
from collections import defaultdict

def normalize_line(p1, p2):
    """Return canonical line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    g = __import__('math').gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    c = dy_n * x1 - dx_n * y1
    return (dx_n, dy_n, c)

def count_4_lines_fast(points):
    """Fast count of lines with exactly 4 points. Returns -1 if any 5 collinear."""
    n = len(points)
    if n != 16:
        return 0
    
    line_mask = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = normalize_line(points[i], points[j])
            if key is None:
                continue
            mask = (1 << i) | (1 << j)
            if key in line_mask:
                line_mask[key] |= mask
            else:
                line_mask[key] = mask
    
    count4 = 0
    for key, mask in line_mask.items():
        popcount = bin(mask).count('1')
        if popcount >= 5:
            return -1
        if popcount == 4:
            count4 += 1
    return count4

def generate_candidate_lines(grid_size):
    """
    Generate all 4-point collinear subsets within a [-grid_size, grid_size] box.
    Returns a list of sets of point tuples.
    """
    points = [(x, y) for x in range(-grid_size, grid_size + 1) for y in range(-grid_size, grid_size + 1)]
    
    # Map line key to list of points on that line
    line_points = defaultdict(list)
    
    # To speed up, we can iterate over pairs, but that's O(N^2). 
    # For small grids (e.g., 5x5 = 100 pts), 100^2 = 10k pairs. Manageable.
    # However, we only care about lines with >= 4 points.
    
    n_pts = len(points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            key = normalize_line(points[i], points[j])
            if key is None:
                continue
            line_points[key].append(points[i])
            line_points[key].append(points[j])
            
    # Deduplicate points in each line list (since pairs might add same point twice? No, i<j, so unique pairs. 
    # But multiple pairs on same line add same points. We need unique points per line.)
    candidate_lines = []
    for key, pts in line_points.items():
        unique_pts = set(pts)
        if len(unique_pts) >= 4:
            # We only need the 4-point subsets? 
            # Actually, if a line has 5 points, we can't use it directly as a "4-point line" 
            # because the constraint is "no 5 collinear". 
            # So we must select 4 points from this line. 
            # But if we select 4 points from a line that has 5 available, we must ensure 
            # the 5th point is NOT in our final set.
            # This makes the "Line-based" approach tricky if lines have >4 points in the grid.
            # Strategy: Only consider lines that have EXACTLY 4 points in the grid? 
            # Or handle the exclusion?
            # Simpler: Only use lines with exactly 4 points in the candidate pool.
            # This restricts the search space but avoids complex exclusion logic.
            if len(unique_pts) == 4:
                candidate_lines.append(frozenset(unique_pts))
                
    return candidate_lines

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Baseline: 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in grid]
    best_score = count_4_lines_fast(best_points)
    
    random.seed(42)
    
    # Strategy: 
    # 1. Generate candidate 4-point lines in a small grid (e.g., 0..5 or -2..3)
    # 2. Build conflict graph: Two lines conflict if they share >= 3 points.
    # 3. Find a Max Independent Set of lines.
    # 4. The union of points in these lines is our candidate set.
    # 5. If size < 16, pad with random points. If size > 16, trim? 
    #    Ideally, we want exactly 16 points. 
    #    If we pick k lines, and they are disjoint (share 0-2 points), the number of points 
    #    is roughly 4k - overlaps.
    #    To get 16 points, we might need ~4-5 lines.
    
    # Let's try to construct a set of points directly from a "Line Packing" perspective.
    # We will use a greedy algorithm to pack 4-point lines into a set of 16 points.
    
    def try_line_packing(grid_size, iterations=100):
        """
        Greedily select non-conflicting 4-point lines from a grid.
        Returns the best point set found.
        """
        # Generate all 4-point lines in the grid
        # To keep it fast, we use a small grid.
        # If grid is 5x5 (25 pts), number of lines is small.
        
        # Pre-compute lines for a few grid sizes
        # We'll use a grid centered at 0.
        
        best_local_score = 0
        best_local_points = None
        
        for _ in range(iterations):
            # Shuffle candidate lines to break symmetry
            candidates = generate_candidate_lines(grid_size)
            if not candidates:
                continue
                
            random.shuffle(candidates)
            
            selected_lines = []
            used_points = set()
            conflict_graph = {} # line_index -> set of conflicting line indices
            
            # Build conflict map for speed
            # Two lines conflict if |L1 ∩ L2| >= 3
            # Since we only have 4-point lines, intersection size 3 or 4 means conflict.
            # Intersection size 4 means same line.
            
            # We'll do it on the fly in the greedy loop for simplicity, 
            # but pre-computing is better. Let's pre-compute for this candidate set.
            
            n_lines = len(candidates)
            conflicts = [set() for _ in range(n_lines)]
            
            for i in range(n_lines):
                for j in range(i + 1, n_lines):
                    inter = len(candidates[i] & candidates[j])
                    if inter >= 3:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            
            # Greedy selection: Pick lines that add the most new points while minimizing conflicts?
            # Or just pick first available?
            # To maximize number of 4-lines, we want as many lines as possible.
            # But we also want the points to be distinct and total 16.
            
            # Let's try to pick a set of lines such that their union is <= 16 points.
            # And ideally exactly 16.
            
            selected = []
            current_union = set()
            
            # Sort candidates by some heuristic? Random is fine for diversity.
            for idx, line in enumerate(candidates):
                # Check if adding this line keeps the total points <= 16
                new_union = current_union | line
                if len(new_union) > 16:
                    continue
                    
                # Check if this line conflicts with any already selected line
                is_conflict = False
                for sel_idx in selected:
                    if idx in conflicts[sel_idx]:
                        is_conflict = True
                        break
                
                if not is_conflict:
                    selected.append(idx)
                    current_union = new_union
                    
            # If we have exactly 16 points, great.
            # If less, we can add random points.
            # If we have 0 lines, skip.
            
            if len(selected) == 0:
                continue
                
            # Construct the point list
            pts_list = list(current_union)
            
            # If we have fewer than 16 points, pad with random points from the grid
            # that don't create 5-collinear violations.
            while len(pts_list) < 16:
                # Pick a random point from the grid not in pts_list
                # Check if adding it creates 5-collinear
                # This is expensive to check every time. 
                # Instead, just pick random points and validate the whole set later.
                
                # Pick random point in [-grid_size, grid_size]
                px = random.randint(-grid_size, grid_size)
                py = random.randint(-grid_size, grid_size)
                p = (px, py)
                if p not in pts_list:
                    pts_list.append(p)
                    
            if len(pts_list) != 16:
                continue
                
            # Validate
            pts = [list(p) for p in pts_list]
            score = count_4_lines_fast(pts)
            
            if score > 0 and score > best_local_score:
                best_local_score = score
                best_local_points = pts
                
        return best_local_score, best_local_points

    # Try packing in different grid sizes
    for gs in [4, 5, 6, 8]:
        if time.time() > deadline:
            break
            
        local_score, local_points = try_line_packing(gs, iterations=200)
        
        if local_points and local_score > best_score:
            best_score = local_score
            best_points = local_points
            
            # Refine with local search on points
            current = [p[:] for p in best_points]
            current_score = local_score
            
            # Simple local search to fix any issues or improve
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for idx in range(16):
                    if time.time() > deadline:
                        break
                    old = current[idx][:]
                    # Try small moves
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = old[0] + dx, old[1] + dy
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            current[idx] = [nx, ny]
                            if len(set(tuple(p) for p in current)) < 16:
                                current[idx] = old
                                continue
                            s = count_4_lines_fast(current)
                            if s > current_score:
                                current_score = s
                                improved = True
                                if s > best_score:
                                    best_score = s
                                    best_points = [p[:] for p in current]
                                break
                            else:
                                current[idx] = old
                        if improved:
                            break
                    if improved:
                        break

    # If line packing didn't find anything better, fall back to random restarts as in seed
    if time.time() < deadline:
        for restart in range(50):
            if time.time() > deadline:
                break
            
            size = random.choice([10, 20])
            current = [[random.randint(-size, size), random.randint(-size, size)] for _ in range(16)]
            
            attempts = 0
            while len(set(tuple(p) for p in current)) < 16 and attempts < 100:
                idx = random.randint(0, 15)
                current[idx] = [random.randint(-size, size), random.randint(-size, size)]
                attempts += 1
                
            if len(set(tuple(p) for p in current)) < 16:
                continue
                
            current_score = count_4_lines_fast(current)
            if current_score < 0:
                continue
                
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current]
                
            # Quick local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for idx in range(16):
                    if time.time() > deadline:
                        break
                    old = current[idx][:]
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = old[0] + dx, old[1] + dy
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            current[idx] = [nx, ny]
                            if len(set(tuple(p) for p in current)) < 16:
                                current[idx] = old
                                continue
                            s = count_4_lines_fast(current)
                            if s > current_score:
                                current_score = s
                                improved = True
                                if s > best_score:
                                    best_score = s
                                    best_points = [p[:] for p in current]
                                break
                            else:
                                current[idx] = old
                        if improved:
                            break
                    if improved:
                        break

    return best_points
