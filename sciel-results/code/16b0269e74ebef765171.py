# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using priority-based greedy with local search.
    
    Key structural idea: The Nikodym constraint requires that for every point x,
    at least one of the 50 lines through x must be "clean" (contain no other excluded
    points). This is a hitting set / maximum independent set problem.
    
    We use a priority queue approach: always add the most constrained point first
    (fewest safe directions remaining), which tends to produce larger valid sets.
    Then apply local search: try swapping pairs of excluded points to find better configurations.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 600s limit
    
    p = 7
    q_field = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical: v = (0, 1)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Precompute neighbors: neighbors[pt][d] = list of other points on same line in direction d
    neighbors = [[[] for _ in range(50)] for _ in range(n)]
    for d in range(50):
        for line in direction_lines[d]:
            for i in range(len(line)):
                for j in range(i+1, len(line)):
                    a, b = line[i], line[j]
                    neighbors[a][d].append(b)
                    neighbors[b][d].append(a)
    
    # For each point, precompute which lines it belongs to (for quick lookup)
    # point_lines[pt] = list of (dir_idx, line_idx)
    point_lines = [[] for _ in range(n)]
    for d in range(50):
        for line_idx, line in enumerate(direction_lines[d]):
            for pt in line:
                point_lines[pt].append((d, line_idx))
    
    def count_safe(in_S, pt):
        """Count how many directions are still safe for point pt (no other S points on line)."""
        safe = 0
        for d in range(50):
            line_has_S = False
            for q in neighbors[pt][d]:
                if in_S[q]:
                    line_has_S = True
                    break
            if not line_has_S:
                safe += 1
        return safe
    
    def is_valid_nikodym(in_S):
        """Check if N = All \\ in_S is a valid Nikodym set."""
        for x in range(n):
            found = False
            for d in range(50):
                line_has_S = False
                for q in neighbors[x][d]:
                    if in_S[q]:
                        line_has_S = True
                        break
                if not line_has_S:
                    found = True
                    break
            if not found:
                return False
        return True
    
    def try_add_to_S(in_S, safe_count, pt):
        """
        Try to add pt to S. Update safe_count for affected points.
        Returns (success, changes) where changes is list of (q, d) for rollback.
        safe_count[q] tracks number of safe directions for q.
        """
        changes = []
        valid = True
        
        for d in range(50):
            line = direction_lines[d][point_line[pt][d]]
            for q in line:
                if q == pt:
                    continue
                # q loses direction d
                if safe_count[q] > 0:
                    safe_count[q] -= 1
                    changes.append((q, d))
                    if safe_count[q] == 0:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            in_S[pt] = True
            return True, changes
        else:
            # Rollback
            for q, d in changes:
                safe_count[q] += 1
            return False, []
    
    def local_search(in_S, safe_count, max_iters=10000):
        """
        Local search: try to improve by swapping excluded points.
        A swap: remove pt1 from S, add pt2 to S. This is valid if:
        - Removing pt1 doesn't break validity (it can only help)
        - Adding pt2 is valid given the new state
        Net effect: |S| stays same, but we might enable further additions.
        
        Better: try to ADD a point that was previously blocked.
        Or: try removing a point and see if we can add two others.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try to add each non-S point
            for pt in range(n):
                if in_S[pt]:
                    continue
                
                # Check if we can add pt
                can_add = True
                changes = []
                for d in range(50):
                    line = direction_lines[d][point_line[pt][d]]
                    for q in line:
                        if q == pt:
                            continue
                        if safe_count[q] > 0:
                            safe_count[q] -= 1
                            changes.append((q, d))
                            if safe_count[q] == 0:
                                can_add = False
                                break
                    if not can_add:
                        break
                
                if can_add:
                    in_S[pt] = True
                    improved = True
                else:
                    # Rollback
                    for q, d in changes:
                        safe_count[q] += 1
        
        return in_S, safe_count
    
    def greedy_priority(in_S, safe_count, order):
        """
        Priority-based greedy: process points in given order.
        At each step, among remaining points, pick the one with fewest safe directions.
        """
        remaining = set(order)
        while remaining:
            # Find most constrained point
            best_pt = None
            best_safe = float('inf')
            for pt in remaining:
                sc = safe_count[pt]
                if sc < best_safe:
                    best_safe = sc
                    best_pt = pt
                    if sc <= 1:
                        break
            
            if best_pt is None:
                break
            
            remaining.remove(best_pt)
            
            success, changes = try_add_to_S(in_S, safe_count, best_pt)
            # changes already applied if success
            if not success:
                pass  # already rolled back
        
        return in_S, safe_count
    
    best_S = [False] * n
    best_size = 0
    
    # Multiple strategies with different orderings
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Different ordering strategies
        strategy = r % 4
        order = list(range(n))
        
        if strategy == 0:
            random.shuffle(order)
        elif strategy == 1:
            # Sort by x-coordinate then y-coordinate (structured)
            order.sort(key=lambda i: (all_points[i][0], all_points[i][1], all_points[i][2], all_points[i][3]))
        elif strategy == 2:
            # Sort by "centrality" - points with more neighbors first
            # All points have same number of neighbors in this geometry, so use random with bias
            random.shuffle(order)
            # Reverse to try different order
            order.reverse()
        else:
            # Random
            random.shuffle(order)
        
        in_S = [False] * n
        safe_count = [50] * n
        
        # Phase 1: Priority greedy
        in_S, safe_count = greedy_priority(in_S, safe_count, order)
        
        # Phase 2: Local search - try to add more points
        in_S, safe_count = local_search(in_S, safe_count, max_iters=5000)
        
        current_size = sum(in_S)
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
    
    # Final verification
    if not is_valid_nikodym(best_S):
        # Fallback: use empty S (all points in N)
        best_S = [False] * n
        best_size = 0
    
    # Build result: N = All \\ S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
