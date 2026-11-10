import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin before 630s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines for each direction
    # Direction 0: vertical v = (0,1)
    # Directions 1-49: v = (1, m) for m in F_49
    
    direction_lines = []
    
    # Vertical direction: v = (0, 1)
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
                # Line through (xa,xb, ya,yb) with direction (1, m):
                # x' = xa + t, y' = ya + t*m
                # c = y - x*m (intercept)
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[point_idx][dir_idx] = line_idx within that direction
    point_line = [[None] * 50 for _ in range(n)]
    
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Helper: check if a point has a valid witness
    # A point pt has a valid witness if there exists a direction dir such that
    # all other points on the line through pt in direction dir are in N
    def has_witness(pt_idx, in_N):
        for dir_idx in range(50):
            line_idx = point_line[pt_idx][dir_idx]
            line = direction_lines[dir_idx][line_idx]
            all_others_in_N = True
            for other_idx in line:
                if other_idx != pt_idx and not in_N[other_idx]:
                    all_others_in_N = False
                    break
            if all_others_in_N:
                return True
        return False
    
    # For efficiency, precompute for each point and direction, the list of other points on that line
    # other_points[pt_idx][dir_idx] = list of other point indices on the line
    other_points = [[[] for _ in range(50)] for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for i, pt_idx in enumerate(line):
                for j, other_idx in enumerate(line):
                    if i != j:
                        other_points[pt_idx][dir_idx].append(other_idx)
    
    def has_witness_fast(pt_idx, in_N):
        for dir_idx in range(50):
            others = other_points[pt_idx][dir_idx]
            all_in_N = True
            for other_idx in others:
                if not in_N[other_idx]:
                    all_in_N = False
                    break
            if all_in_N:
                return True
        return False
    
    # Try different baseline constructions and pick the best one
    # Construction 1: exclude points on unital with ya NOT in squares
    # Construction 2: exclude points on unital with ya IN squares
    # Construction 3: exclude all points on unital
    
    squares = set((u * u) % p for u in range(p))
    
    def in_unital(xa, xb, ya, yb):
        return (2 * xa + ya * ya - k * yb * yb) % p == 0
    
    baselines = []
    
    # Baseline 1: ya NOT in squares
    in_N_1 = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_unital(xa, xb, ya, yb) and ya not in squares:
            in_N_1[i] = False
    baselines.append(in_N_1)
    
    # Baseline 2: ya IN squares
    in_N_2 = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_unital(xa, xb, ya, yb) and ya in squares:
            in_N_2[i] = False
    baselines.append(in_N_2)
    
    # Baseline 3: all unital points excluded
    in_N_3 = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_unital(xa, xb, ya, yb):
            in_N_3[i] = False
    baselines.append(in_N_3)
    
    # Pick the baseline with smallest |N| (largest complement)
    best_baseline = min(baselines, key=lambda in_N: sum(in_N))
    in_N = best_baseline[:]
    
    # Validate the baseline
    def is_valid_nikodym(in_N):
        for pt_idx in range(n):
            if not has_witness_fast(pt_idx, in_N):
                return False
        return True
    
    # If baseline is not valid, fall back to full plane
    if not is_valid_nikodym(in_N):
        in_N = [True] * n
    
    # Greedy improvement: try to remove points from N
    # A point can be removed if after removal, all points still have witnesses
    # For efficiency, only check points affected by the removal
    
    def can_remove(pt_idx, in_N):
        # Temporarily remove pt_idx
        in_N[pt_idx] = False
        # Check if all points on lines through pt_idx still have witnesses
        # Only need to check points that were on lines through pt_idx
        affected = set()
        for dir_idx in range(50):
            for other_idx in other_points[pt_idx][dir_idx]:
                affected.add(other_idx)
        
        for aff_idx in affected:
            if not has_witness_fast(aff_idx, in_N):
                in_N[pt_idx] = True
                return False
        
        in_N[pt_idx] = True
        return True
    
    # Greedy: repeatedly try to remove points
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for pt_idx in range(n):
            if time.time() > deadline - 10:
                break
            if in_N[pt_idx]:
                if can_remove(pt_idx, in_N):
                    in_N[pt_idx] = False
                    improved = True
    
    # Convert to array
    result_points = []
    for i in range(n):
        if in_N[i]:
            result_points.append(all_points[i])
    
    result = np.zeros((len(result_points), 2, 2), dtype=np.int64)
    for i, (xa, xb, ya, yb) in enumerate(result_points):
        result[i] = [[xa, xb], [ya, yb]]
    
    return result
